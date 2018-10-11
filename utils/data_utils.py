import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import os
import sys
import random
import json

from data_generators import go_problem
from go_game import go
from utils import go_utils, utils


class DatasetStats:
    def __init__(self, problem, hparams):
        # set random seed to make sure shuffle is re-creatable
        random.seed(230)

        self.hparams = hparams

        self.board_size = problem.board_size
        self.multiple_datasets = problem.multiple_datasets
        self.sort_sequence_by_color = problem.sort_sequence_by_color

        gogod_files = go_problem.get_gogod_filenames(hparams.tmp_dir, problem.board_size)
        train_gogod, dev_gogod, test_gogod = go_problem.split_dataset(gogod_files, problem.split_fractions)

        self.files = {
            "train": train_gogod,
            "dev": dev_gogod,
            "test": test_gogod
        }

        if self.multiple_datasets:
            kgs_files = go_problem.get_kgs_filenames(hparams.tmp_dir)
            train_kgs, dev_kgs, test_kgs = go_problem.split_dataset(kgs_files, problem.split_fractions)

            self.files = {
                "train": train_gogod + train_kgs,
                "gogod_dev": dev_gogod,
                "kgs_dev": dev_kgs,
                "gogod_test": test_gogod,
                "kgs_test": test_kgs
            }

    def print_stats(self):
        for k, lengths in self.lengths.items():
            min_length = np.min(lengths)
            max_length = np.max(lengths)
            mean_length = np.mean(lengths)
            string = "{} dataset game_length stats:\n" \
                     "- min:  {:03}\n" \
                     "- max:  {:03}\n" \
                     "- mean: {:03}\n".format(k, min_length, max_length, mean_length)
            print(string)

            self.plot_histogram(lengths, k, max_length - min_length, 1)

    @staticmethod
    def plot_histogram(lengths, dataset_name, num_bins, bin_width):
        # matplotlib histogram
        plt.hist(lengths, color='blue', edgecolor='black', bins=int(num_bins / bin_width))

        # Add labels
        plt.title('Histograms of game lengths in {}'.format(dataset_name))
        plt.xlabel("Game lengths")
        plt.ylabel('Games')
        plt.show()

    def get_sizes(self, mode):
        assert mode in ['rnn', 'cnn'], "Mode must be either 'rnn' or 'cnn'!"

        if not hasattr(self, 'sizes'):
            self.create_or_load_sizes()

        if self.sort_sequence_by_color and mode == "rnn":
            mode += "_sorted"
        return self.sizes[mode]

    def create_or_load_sizes(self):
        hp = self.hparams
        data_dir = hp.data_dir
        min_length = hp.min_length if hasattr(hp, "min_length") else 0

        if hasattr(self, "sizes"):
            return

        modes = ['rnn', 'rnn_sorted', 'cnn']

        json_path = os.path.join(data_dir, 'dataset_params_{:03}.json'.format(min_length))
        if os.path.isfile(json_path):
            tf.logging.info("Skipped creating sizes. Found dataset_params.json in data_dir.")

            self.sizes = {}

            with open(json_path) as f:
                raw_dict = f.read()
                sizes = json.loads(raw_dict)

            for mode in modes:
                self.sizes[mode] = sizes[mode]
        else:
            self.create_sizes()

    def create_sizes(self):
        hp = self.hparams
        data_dir = hp.data_dir
        min_length = hp.min_length if hasattr(hp, "min_length") else 0

        tf.logging.info("Generating dataset_params in data dir with min_length {:03}".format(min_length))

        modes = ['rnn', 'rnn_sorted', 'cnn']

        mode_to_stat = {
            'rnn': lambda x: len(x),
            'rnn_sorted': lambda x: len(x),
            'cnn': lambda x: np.sum(x)
        }

        if hasattr(self, 'sizes'):

            return self.sizes

        self.lengths = {}

        for k, files in self.files.items():
            game_lengths = []

            for file in files:
                game_length = get_game_length(file)
                if min_length <= game_length:
                    game_lengths.append(game_length)
            self.lengths[k] = np.array(game_lengths)

        self.sizes = {}

        for mode in modes:
            tmp = {}
            for k, game_lengths in self.lengths.items():
                tmp[k + "_size"] = mode_to_stat[mode](game_lengths)
            self.sizes[mode] = tmp

        file_out = os.path.join(data_dir, 'dataset_params_{:03}.json'.format(min_length))
        utils.save_dicts_to_json(self.sizes, file_out)

        self.print_stats()


def get_game_length(filename):
    _, plays, _ = go_utils.read_sgf(filename)
    game_length = len(plays)
    return game_length


def example_length(example):
    length = example["game_length"]
    return length


def example_valid_size(example, min_length):
    length = example_length(example)
    return tf.logical_and(
        length >= min_length,
        tf.constant(True),
        )


def remove_bad_files(tmp_dir, board_size):
    # search all sgf files in the gogod dataset
    filenames = go_problem.get_gogod_filenames(tmp_dir, board_size)
    print(len(filenames))
    bad_files = find_bad_files(filenames)

    if board_size == 19:
        # search all sgf files in the kgs dataset
        kgs_files = go_problem.get_kgs_filenames(tmp_dir)
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
    sgf_board, plays, _ = go_utils.read_sgf(filename)

    board_size = sgf_board.side
    go.set_board_size(board_size)

    assert board_size == go.BOARD_SIZE, "Wrong Board Size in SGF"

    np_board = np.array(sgf_board.board)

    initial_board = go_utils._prep_board(np_board)
    plays = go_utils._prep_plays(plays)

    try:
        first_player = go_utils._get_first_player(plays)
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
