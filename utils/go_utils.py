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


def parse_sgf(filename, board_size, sort_by_color=False):
    """Parses a sgf file to a game dict.

    Args:
        filename: str, path of the sgf file
        board_size: int, board size
        sort_by_color: bool, if True sort sequences by first black then white
    Returns:
        A dictionary representing a go game with the following fields:
        * positions: str of [game_length, 3, board_size, board_size] np.array, encoded game positions
        * p_targets: [game_length] int list, index of the played move (incl. pass move)
        * v_targets: [game_length] int list, winner of the game, 1 if current player is winning, -1 otherwise
        * legal_moves: str of [game_length, num_moves] np.array, encoded legal_moves at every position
        * game_length: int, game length
        Fields positions, legal_moves, game_length is actually a list of the corresponding type.

        or

        None if one of these to Errors occurs:
        * sgf contains no moves, affects about 1% of files in KGS dataset
        * move in the sgf breaks the minigo ko rule, affects about 1% of files in GoGoD dataset
    """
    go.set_board_size(board_size)

    sgf_board, plays, sgf_game = read_sgf(filename)

    size = sgf_board.side

    assert size == go.BOARD_SIZE, "Wrong Board Size in SGF"

    np_board = np.array(sgf_board.board)

    initial_board = _prep_board(np_board)
    plays = _prep_plays(plays)
    winner = _get_winner(sgf_game)

    try:
        first_player = _get_first_player(plays)
    except IndexError:
        tf.logging.error("Skipped reading Go game from sgf '{}' because no moves were found!".format(filename))
        return None

    num_moves = go.BOARD_SIZE * go.BOARD_SIZE + 1

    game_length = len(plays)
    positions = np.zeros([game_length, 3, go.BOARD_SIZE, go.BOARD_SIZE], dtype=np.uint8)
    p_targets = np.zeros([game_length], dtype=np.int16)
    v_targets = np.zeros([game_length], dtype=np.int8)
    legal_moves = np.zeros([game_length, num_moves], dtype=np.uint8)

    go_game = go.GoEnvironment(None, initial_board, to_play=first_player)

    for i, play in enumerate(plays):
        colour, move = play

        # create board position [3, board_size, board_size]
        board = go_game.board
        np_board = _generate_board(board, colour)
        positions[i] = np_board

        # create policy target int in [0, board_size * board_size + 1)
        p_target = _generate_p_target(move)
        p_targets[i] = p_target

        # create value target: 1 if colour is winning -1 if enemy colour is winning
        v_target = _generate_v_target(winner, colour)
        v_targets[i] = v_target

        # create legal_moves
        legal_move = go_game.all_legal_moves()
        legal_moves[i] = legal_move

        # play move to update board for next iteration
        try:
            go_game.play_move(move, colour, True)
        except go.IllegalMove:
            tf.logging.error("Skipped reading Go game from sgf '{}' because IllegalMove error occurred!".format(filename))
            return None

    if sort_by_color:
        mask_w = positions[:, 2, 0, 0] == 0
        mask = np.argsort(mask_w, kind='mergesort')

        positions = positions[mask]
        p_targets = p_targets[mask]
        v_targets = v_targets[mask]
        legal_moves = legal_moves[mask]

    data = {
        'positions': [positions.tostring()],
        'p_targets': p_targets.tolist(),
        'v_targets': v_targets.tolist(),
        'legal_moves': [legal_moves.tostring()],
        'game_length': [game_length]
    }

    return data


def read_sgf(filename):
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


def _generate_board(board, to_play):
    """Generates output board.

    Args:
        board: [19, 19] numpy array with stone colors
        to_play: color of current player, BLACK = 1 and WHITE = -1
    Returns:
        [3, 19, 19] numpy array:
        * channel 0: current player stones
        * channel 1: enemy player stons
        * channel 2: 1 if current player is BLACK, 0 for WHITE
    """
    np_board = np.zeros([3, go.BOARD_SIZE, go.BOARD_SIZE])

    mask_black = board == go.BLACK
    mask_white = board == go.WHITE

    if to_play == go.BLACK:
        np_board[0][mask_black] = 1
        np_board[1][mask_white] = 1
        np_board[2] = np.ones([go.BOARD_SIZE, go.BOARD_SIZE])
    else:
        np_board[0][mask_white] = 1
        np_board[1][mask_black] = 1

    return np_board


def _generate_p_target(move):
    """Generate the flat index of a move."""
    return coordinates.to_flat(move)


def _generate_v_target(winner, to_play):
    """Generate the winner from the current players PoV.

    Args:
        winner: absolute winner BLACK = 1, WHITE = -1, DRAW = 0
        to_play: color of the current player
    Returns:
        1 if current player is winning
       -1 if enemy is winning
    """
    if to_play == go.BLACK:
        return winner
    return winner * -1
