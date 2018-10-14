from data_generators import base_go_problem
from go_game import go
from utils import sgf_utils

import numpy as np

import argparse
import sys
import os


def remove_bad_files(tmp_dir, board_size):
    # search all sgf files in the gogod dataset
    filenames = base_go_problem.get_gogod_filenames(tmp_dir, board_size)
    print(len(filenames))
    bad_files = find_bad_files(filenames)

    if board_size == 19:
        # search all sgf files in the kgs dataset
        kgs_files = base_go_problem.get_kgs_filenames(tmp_dir)
        print(len(kgs_files))
        kgs_bad = find_bad_files(kgs_files)
        bad_files += kgs_bad

    to_string = "\n".join("- {}".format(file) for file in bad_files)
    question = "Bad Filenames:\n{}".format(to_string)
    question += "\nDo you want to remove {} bad sgf files?".format(len(bad_files))

    remove = query_yes_no(question, None)

    if remove:
        out_file = os.path.join(tmp_dir, "deleted_files.txt")
        with open(out_file, 'w') as f:
            for file in bad_files:
                os.remove(file)
                f.write(file + '\n')


def find_bad_files(filenames):
    bad_files = []
    for filename in filenames:
        out = test_sgf(filename)
        if out is None:
            bad_files.append(filename)
    return bad_files


def test_sgf(filename):
    """Parses a sgf file to a game dict.

    Args:
        filename: str, path of the sgf file
    Returns:
        None if one of these to Errors occurs:
        * sgf contains no moves, affects about 1% of files in KGS dataset
        * move in the sgf breaks the minigo ko rule, affects about 1% of files in GoGoD dataset
    """
    sgf_board, plays, _ = sgf_utils.read_sgf(filename)

    board_size = sgf_board.side
    go.set_board_size(board_size)

    assert board_size == go.BOARD_SIZE, "Wrong Board Size in SGF"

    np_board = np.array(sgf_board.board)

    initial_board = sgf_utils._prep_board(np_board)
    plays = sgf_utils._prep_plays(plays)

    try:
        first_player = sgf_utils._get_first_player(plays)
    except IndexError:
        print("Skipped reading Go game from sgf '{}' because no moves were found!".format(filename))
        return None

    go_game = go.GoEnvironment(None, initial_board, to_play=first_player)

    for i, play in enumerate(plays):
        colour, move = play
        try:
            go_game.play_move(move, colour, True)
        except go.IllegalMove:
            print("Skipped reading Go game from sgf '{}' because IllegalMove error occurred!".format(filename))
            return None

    return True


def query_yes_no(question, default="yes"):
    """Ask a yes/no question via input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")


parser = argparse.ArgumentParser()
parser.add_argument('--tmp_dir',
                    help="Tmp directory containing the sgf files", required=True, type=str)
parser.add_argument('--board_size',
                    help="Board sizes of the sgf files", required=True, type=int)

if __name__ == '__main__':
    args = parser.parse_args()

    tmp_dir = args.tmp_dir
    board_size = args.board_size

    find_bad_files(tmp_dir, board_size)
