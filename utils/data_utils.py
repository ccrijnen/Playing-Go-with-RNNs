import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import os
import json

from tensor2tensor.data_generators.problem import DatasetSplit
from utils import sgf_utils, utils


class DatasetStats:
    """Calculates the number of train, dev and test examples for a problem with specific hparams and
    saves the number of examples for every mode to data/dataset_params_{problem_suffix}.json.

    If it finds data/dataset_params_{problem_suffix}.json for the Problem, loads the number of examples from the file.
    """
    def __init__(self, problem, hparams):
        # set random seed to make sure shuffle is re-creatable
        self.problem = problem
        self.hparams = hparams
        self.sort_sequence_by_color = problem.sort_sequence_by_color
        self.suffix = "_{}".format(problem.board_size)

        if problem.is_small:
            self.suffix += "_small"

        if problem.use_gogod_data and problem.use_kgs_data:
            self.suffix += "_multi"
        elif problem.use_kgs_data:
            self.suffix += "_kgs"
        elif problem.use_gogod_data:
            self.suffix += "_gogod"

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
        if hasattr(self.hparams, "split") and self.hparams.split_to_min_length and mode == "cnn":
            mode += "_split"

        return self.sizes[mode]

    def create_or_load_sizes(self):
        hp = self.hparams
        data_dir = hp.data_dir
        min_length = hp.min_length if hasattr(hp, "min_length") else 0
        max_length = hp.max_length if hasattr(hp, "max_length") else 0

        max_str = "-{:03}".format(max_length) if max_length else ""

        if hasattr(self, "sizes"):
            return

        modes = ['rnn', 'rnn_sorted', 'cnn', 'cnn_split']

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

        modes = ['rnn', 'rnn_sorted', 'cnn', 'cnn_split']

        mode_to_stat = {
            'rnn': lambda x: len(x),
            'rnn_sorted': lambda x: 2 * len(x),
            'cnn': lambda x: np.sum(x),
            'cnn_split_to_min_length': lambda x: len(x) * min_length
        }

        if hasattr(self, 'sizes'):
            return self.sizes

        splits = {
            'train': DatasetSplit.TRAIN,
            'dev': DatasetSplit.EVAL,
            'test': DatasetSplit.TEST
        }
        self.lengths = {}

        for split, dataset_split in splits.items():
            game_lengths = []

            data_filepattern = self.problem.filepattern(data_dir, dataset_split)
            data_files = sorted(tf.contrib.slim.parallel_reader.get_data_files(data_filepattern))

            for file in data_files:
                record_iterator = tf.python_io.tf_record_iterator(path=file)

                for string_record in record_iterator:
                    example = tf.train.Example()
                    example.ParseFromString(string_record)

                    game_length = int(example.features.feature['game_length'].int64_list.value[0])

                    if (min_length <= game_length <= max_length) or (min_length <= game_length and not max_length):
                        game_lengths.append(game_length)
            self.lengths[split] = np.array(game_lengths)

        self.sizes = {}

        for mode in modes:
            tmp = {}
            for split, game_lengths in self.lengths.items():
                tmp[split + "_size"] = mode_to_stat[mode](game_lengths)
            self.sizes[mode] = tmp

        max_str = "-{:03}".format(max_length) if max_length else ""
        file_out = os.path.join(data_dir, 'dataset_params{}_{:03}{}.json'.format(self.suffix, min_length, max_str))
        utils.save_dicts_to_json(self.sizes, file_out)

        self.print_stats()


def example_length(example):
    length = example["game_length"]
    return length


def example_valid_size(example, min_length, max_length):
    length = example_length(example)
    return tf.logical_and(
        length >= min_length,
        length <= max_length)
