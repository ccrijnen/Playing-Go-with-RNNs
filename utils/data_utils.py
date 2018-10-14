import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import os
import random
import json

from data_generators import base_go_problem, go_problem_19
from utils import sgf_utils, utils


class DatasetStats:
    def __init__(self, problem, hparams):
        # set random seed to make sure shuffle is re-creatable
        random.seed(230)

        self.hparams = hparams

        self.board_size = problem.board_size
        self.multiple_datasets = problem.multiple_datasets
        self.sort_sequence_by_color = problem.sort_sequence_by_color

        gogod_files = base_go_problem.get_gogod_filenames(hparams.tmp_dir, problem.board_size)
        train_gogod, dev_gogod, test_gogod = base_go_problem.split_dataset(gogod_files, problem.split_fractions)

        self.files = {
            "train": train_gogod,
            "dev": dev_gogod,
            "test": test_gogod
        }
        self.suffix = ""

        if self.multiple_datasets:
            kgs_files = base_go_problem.get_kgs_filenames(hparams.tmp_dir)
            train_kgs, dev_kgs, test_kgs = base_go_problem.split_dataset(kgs_files, problem.split_fractions)
            if isinstance(problem, go_problem_19.GoProblem19x19):
                self.files = {
                    "train": train_gogod + train_kgs,
                    "dev": dev_gogod + dev_kgs,
                    "test": test_gogod + test_kgs,
                }
                self.suffix = "_multi"

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
        max_length = hp.max_length if hasattr(hp, "max_length") else 0

        max_str = "-{:03}".format(max_length) if max_length else ""

        if hasattr(self, "sizes"):
            return

        modes = ['rnn', 'rnn_sorted', 'cnn']

        json_path = os.path.join(data_dir, 'dataset_params{}_{:03}{}.json'.format(self.suffix, min_length, max_str))
        if os.path.isfile(json_path):
            tf.logging.info("Skipped creating sizes. Found dataset_params{}.json in data_dir.".format(self.suffix))

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
        max_length = hp.max_length if hasattr(hp, "max_length") else 0

        max_str = " and max_length {:03}".format(max_length) if max_length else ""
        tf.logging.info("Generating dataset_params{} in data dir with min_length {:03}{}"
                        .format(self.suffix, min_length, max_str))

        modes = ['rnn', 'rnn_sorted', 'cnn']

        mode_to_stat = {
            'rnn': lambda x: len(x),
            'rnn_sorted': lambda x: 2 * len(x),
            'cnn': lambda x: np.sum(x)
        }

        if hasattr(self, 'sizes'):
            return self.sizes

        self.lengths = {}

        for k, files in self.files.items():
            game_lengths = []

            for file in files:
                game_length = get_game_length(file)
                if (min_length <= game_length <= max_length) or (min_length <= game_length and not max_length):
                    game_lengths.append(game_length)
            self.lengths[k] = np.array(game_lengths)

        self.sizes = {}

        for mode in modes:
            tmp = {}
            for k, game_lengths in self.lengths.items():
                tmp[k + "_size"] = mode_to_stat[mode](game_lengths)
            self.sizes[mode] = tmp

        max_str = "-{:03}".format(max_length) if max_length else ""
        file_out = os.path.join(data_dir, 'dataset_params{}_{:03}{}.json'.format(self.suffix, min_length, max_str))
        utils.save_dicts_to_json(self.sizes, file_out)

        self.print_stats()


def get_game_length(filename):
    _, plays, _ = sgf_utils.read_sgf(filename)
    game_length = len(plays)
    return game_length


def example_length(example):
    length = example["game_length"]
    return length


def example_valid_size(example, min_length, max_length):
    length = example_length(example)
    return tf.logical_and(
        length >= min_length,
        length <= max_length,
        )
