import numpy as np
import tensorflow as tf

from sgfmill import sgf
from sgfmill import sgf_moves

from go_game import go
from go_game import coordinates


def print_legal_moves(legal_moves):
    """Prints a legal_moves numpy array of shape [game_length, num_moves]."""
    moves = legal_moves[:-1].reshape([19, 19])
    moves = 1 - moves
    moves = go.GoEnvironment(19, moves)

    print("Legal moves")
    print(moves)


def print_go_position(board):
    """Prints a board position numpy array with shape [3, board_size, board_size]."""
    assert len(board.shape) == 3

    mask_black = board[0] == 1 if board[2, 0, 0] is 1 else board[1] == 1
    mask_white = board[1] == 1 if board[2, 0, 0] is 1 else board[0] == 1

    new_board = np.copy(go.EMPTY_BOARD)
    new_board[mask_black] = go.BLACK
    new_board[mask_white] = go.WHITE

    to_play = go.BLACK if board[2, 0, 0] == 1 else go.WHITE

    go_game = go.GoEnvironment(board.shape[1], new_board, to_play=to_play)

    print(go_game)


def parse_sgf(filename, board_size, dataset_name):
    """Parses a sgf file to a game dict.

    Args:
        filename: (str), path of the sgf file
        board_size: (int), board size
        dataset_name: (str) optional, name of the dataset
    Returns:
        A dictionary representing a go game with the following fields:
        * positions: (str) of np.array [game_length, board_size, board_size] , encoded game positions,
            stones BLACK: 1 and WHITE: -1
        * p_targets: (int) list, shape: [game_length], index of the played move (incl. pass move)
        * legal_moves: (str) of [game_length, num_moves] np.array, encoded legal_moves at every position
        * to_play: (str) of np.array [game_length], current player at each position, BLACK: 1, WHITE: -1
        * game_length: (int), game length
        * winner: (int), winner of the game, BLACK: 1, WHITE: -1, DRAW: 0
        * dataset_name: (str), either 'kgs' or 'gogod'
        Fields positions, legal_moves, to_play, game_length, winner and dataset_name
             is actually a list of the corresponding type.

        or

        None if one of these to Errors occurs:
        * sgf contains no moves, affects about 1% of files in KGS dataset_name
        * move in the sgf breaks the minigo ko rule, affects about 1% of files in GoGoD dataset_name
    """
    assert dataset_name in ["gogod", "kgs"]

    go.set_board_size(board_size)

    # read the sgf
    sgf_board, plays, sgf_game = read_sgf(filename)

    size = sgf_board.side

    assert size == go.BOARD_SIZE, "Wrong Board Size in SGF"

    # prepare sgf_board and plays
    np_board = np.array(sgf_board.board)
    initial_board = _prep_board(np_board)
    plays = _prep_plays(plays)

    # get first player
    try:
        first_player = _get_first_player(plays)
    except IndexError:
        tf.logging.error("Skipped reading Go game from sgf '{}' because no moves were found!".format(filename))
        return None

    # calculate the number of different possible moves
    num_moves = go.BOARD_SIZE * go.BOARD_SIZE + 1

    # save game_length and winner
    game_length = len(plays)
    winner = _get_winner(sgf_game)

    # create numpy arrays to hold the parsed data
    to_play = np.zeros([game_length], dtype=np.int8)
    positions = np.zeros([game_length, go.BOARD_SIZE, go.BOARD_SIZE], dtype=np.int8)
    p_targets = np.zeros([game_length], dtype=np.int16)
    legal_moves = np.zeros([game_length, num_moves], dtype=np.uint8)

    # initialize go environment
    go_game = go.GoEnvironment(None, initial_board, to_play=first_player)

    for i, play in enumerate(plays):
        # colour: (int) 1 if the current player is BLACK else -1 for WHITE
        # move: (tuple) or (None), coordinate of the next move (from the upper left corner), None if move is pass
        colour, move = play
        to_play[i] = colour

        # create board position [board_size, board_size]
        board = go_game.board
        positions[i] = board

        # create policy target int in [0, board_size * board_size + 1)
        p_target = _generate_p_target(move)
        p_targets[i] = p_target

        # create legal_moves
        legal_move = go_game.all_legal_moves()
        legal_moves[i] = legal_move

        # play move to update board for next iteration
        try:
            go_game.play_move(move, colour, True)
        except go.IllegalMove:
            tf.logging.error("Skipped reading Go game from sgf '{}' because IllegalMove error occurred!"
                             .format(filename))
            return None

    # prepare data for the tf_record reader
    data = {
        'positions': [positions.tostring()],
        'p_targets': p_targets.tolist(),
        'legal_moves': [legal_moves.tostring()],
        'to_play': [to_play.tostring()],
        'game_length': [game_length],
        'winner': [winner],
        'dataset_name': [dataset_name]
    }

    return data


def read_sgf(filename):
    """Parse sgf."""
    with open(filename, "rb") as f:
        sgf_src = f.read()

    try:
        sgf_game = sgf.Sgf_game.from_bytes(sgf_src)
    except ValueError:
        raise Exception("bad sgf file")

    try:
        sgf_board, plays = sgf_moves.get_setup_and_moves(sgf_game)
    except ValueError as e:
        raise Exception(str(e))

    return sgf_board, plays, sgf_game


def _prep_board(board):
    """Create initial board from sgf format for minigo format.

    sgf_mill is a [19, 19] array with stone colors BLACK = 'b' and WHITE = 'w'.
    minigo is a [19, 19] array with stone colors BLACK = 1 and WHITE = -1.
    """
    new_board = np.copy(go.EMPTY_BOARD)

    mask_black = board == 'b'
    mask_white = board == 'w'

    new_board[mask_black] = go.BLACK
    new_board[mask_white] = go.WHITE
    new_board = np.flip(new_board, 0)
    return new_board


def _prep_plays(plays):
    """Flips Coordinates of all moves horizontally.

    sgf_mill has coordinate (0, 0) in the bottom left corner.
    minigo has coordinate (0, 0) in the top left corner.
    """
    flipped_plays = []
    for colour, move in plays:
        colour = go.BLACK if colour == 'b' else go.WHITE
        move = _flip_row_coordinates(move)

        flipped_plays.append((colour, move))

    return flipped_plays


def _flip_row_coordinates(move):
    """Flips a coordinate of a move horizontally.
    Args:
        move: row, col index or None if pass move
    Returns:
        Horizontally flipped move if move was not None
    """
    if move is None:
        return move

    row, col = move
    row = go.BOARD_SIZE - 1 - row

    return row, col


def _get_winner(sgf_game):
    """Reads the games winner from an sgf.
    Args:
        sgf_game: from bytes parsed sgf
    Returns:
        1 if BLACK won
       -1 if WHITE won
        0 if it was a DRAW
    """
    sgf_winner = sgf_game.get_winner()

    if sgf_winner == 'b':
        winner = 1
    elif sgf_winner == 'w':
        winner = -1
    else:
        winner = 0

    return winner


def _get_first_player(plays):
    """Reads fist player.
    Args:
        plays: list of (color, move) tuples
    Returns:
        1 if first player is BLACK
       -1 for WHITE
    """
    first_play = plays[0]
    first_player, _ = first_play

    return first_player


def _generate_p_target(move):
    """Generate the flat index of a move."""
    return coordinates.to_flat(move)
