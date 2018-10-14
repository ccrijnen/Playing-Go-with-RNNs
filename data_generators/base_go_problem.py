import tensorflow as tf
import zipfile
import tarfile
import random
import math
import os
import re

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem

from models.base_go_hparams import base_go_hparams
from utils import sgf_utils

_GOGOD_ZIP_NAME = 'GoGoDSpring2018.zip'
_GOGOD_FILENAMES_GLOB = '/*.sgf'
_GOGOD_PREFIX = 'GoGoD/Database/'
_GOGOD_FOLDER = 'GoGoD/'

_KGS_ZIP_NAMES_GLOB = 'KGS*.tar.gz'
_KGS_FILENAMES_GLOB = '*/*.sgf'
_KGS_FOLDER = 'KGS/'


def _unzip(tmp_dir, out_dir, filename, extracted=None, remove=False):
    path = os.path.join(tmp_dir, filename)
    extension = os.path.splitext(filename)[1]

    if extracted:
        check_file = os.path.join(out_dir, extracted)
        if tf.gfile.Exists(check_file):
            tf.logging.info("Skip extracting '{}' because output '{}' already exists".format(filename, check_file))
            return

    tf.gfile.MakeDirs(out_dir)

    if extension == ".zip":
        zipfile.ZipFile(path, 'r').extractall(out_dir)
    else:
        tarfile.open(path, 'r:gz').extractall(out_dir)

    if remove:
        os.remove(path)


def maybe_unzip_gogod(tmp_dir):
    out_dir = os.path.join(tmp_dir, _GOGOD_FOLDER)
    tf.logging.info("Extracting GoGoD zip '{}' in '{}' to '{}'".format(_GOGOD_ZIP_NAME, tmp_dir, out_dir))
    _unzip(tmp_dir, out_dir, _GOGOD_ZIP_NAME, "Database")


def maybe_unzip_kgs(tmp_dir):
    out_dir = os.path.join(tmp_dir, _KGS_FOLDER)

    zipped_files = tf.gfile.Glob(tmp_dir + _KGS_ZIP_NAMES_GLOB)
    for file in zipped_files:
        file = file.split("\\")[-1]

        tf.logging.info("Extracting KGS zip '{}' in '{}' to '{}'".format(file, tmp_dir, out_dir))

        fsplit = re.split('[\-_]', file)[:-2]
        if fsplit[1] in ['2016', '2017', '2018']:
            name_list = [fsplit[0], fsplit[3], fsplit[1], fsplit[2], 'new']
            extracted = '-'.join(name_list).lower()
        elif fsplit[1] in ['2001', '2002', '2003', '2004', '2005']:
            extracted = fsplit[0] + fsplit[1]
        else:
            name_list = [fsplit[0], fsplit[2], fsplit[1]]
            extracted = '-'.join(name_list).lower()

        _unzip(tmp_dir, out_dir, file, extracted)


def get_gogod_filenames(tmp_dir, board_size):
    """Find all GoGoD sgf filenamses.

    Searches at tmp_dir/GoGoD/Database/board_size_path/*.sgf
    where board_size_path is the path to the correct folders for the board_size.

    Args:
        tmp_dir: str, temporary directory
        board_size: int, board size of the go game
    Returns:
        list of str, gogod filenames
    """
    assert board_size in [9, 13, 15, 19, 21]

    filepath = os.path.join(tmp_dir, _GOGOD_PREFIX)

    if board_size == 19:
        dirs = [d for d in tf.gfile.ListDirectory(filepath) if not d.startswith('Non')]
    else:
        filepath = os.path.join(filepath, "Non19x19Boards/")
        dirs = [d for d in tf.gfile.ListDirectory(filepath) if d.startswith(str(board_size))]

    filenames = []
    for d in dirs:
        path = os.path.join(filepath, d)
        files = tf.gfile.Glob(path + _GOGOD_FILENAMES_GLOB)
        filenames.extend(files)

    filenames.sort()

    return filenames


def get_kgs_filenames(tmp_dir):
    """Find all KGS sgf filenames.

    Searches at tmp_dir/KGS/*/*.sgf

    Args:
        tmp_dir: str, temporary directory
    Returns:
        list of str, kgs filenames
    """
    filepath = os.path.join(tmp_dir, _KGS_FOLDER)
    filenames = tf.gfile.Glob(filepath + _KGS_FILENAMES_GLOB)
    filenames.sort()

    return filenames


def split_dataset(filenames, split_fractions):
    """Split dataset into train, dev and test.

    Args:
        filenames: str, paths to split
        split_fractions: dict<DatasetSplit, fraction>
    Return:
        paths split into train, dev and test according to split fractions
    """
    assert sum(split_fractions.values()) == 1, \
        "Sum of the split fractions is not 1 but {}!".format(sum(split_fractions.values()) == 1)

    random.shuffle(filenames)

    training_fraction = split_fractions[problem.DatasetSplit.TRAIN]
    test_fraction = split_fractions[problem.DatasetSplit.TEST]

    split_index1 = int(math.floor(len(filenames) * training_fraction))
    split_index2 = int(math.floor(len(filenames) * (1 - test_fraction)))

    train_split = filenames[:split_index1]
    dev_split = filenames[split_index1:split_index2]
    test_split = filenames[split_index2:]

    return train_split, dev_split, test_split


class GoProblem(problem.Problem):
    """Abstract Go Problem."""
    @property
    def board_size(self):
        """Board Size of the Go games."""
        raise NotImplementedError

    @property
    def num_moves(self):
        """Equivalent to num_classes."""
        return self.board_size * self.board_size + 1

    @property
    def multiple_datasets(self):
        raise NotImplementedError

    @property
    def sort_sequence_by_color(self):
        """
        Set to True if you want to order the sequence of positions, targets and legal_moves in a game
        by move color resulting in this order:
            [black_1, ..., black_N, white_1, ..., white_K]

        Set to False to preserve the sequence order from the sgf.
        """
        raise NotImplementedError

    @property
    def train_shards(self):
        return 8

    @property
    def dev_shards(self):
        return 1

    @property
    def test_shards(self):
        return 1

    @property
    def split_fractions(self):
        return {
            problem.DatasetSplit.TRAIN: 0.8,
            problem.DatasetSplit.EVAL: 0.1,
            problem.DatasetSplit.TEST: 0.1
        }

    def add_sizes(self, hparams):
        raise NotImplementedError

    def generator(self, datasets):
        """Go game generator from sgf format.

        Args:
            datasets: (tuple), dataset name and path to sgf files to generate go games from

        Yields:
            A dictionary representing a go game with the following fields:
            * positions: str of [game_length, 3, board_size, board_size] np.array, encoded game positions
            * p_targets: [game_length] int list, index of the played move (incl. pass move)
            * v_targets: [game_length] int list, winner of the game, 1 if current player is winning, -1 otherwise
            * legal_moves: str of [game_length, num_moves] np.array, encoded legal_moves at every position
            * game_length: int, game length
            Fields positions, legal_moves, game_length is actually a list of the corresponding type.
        """
        for dataset_name, filenames in datasets:
            for file in filenames:
                yield sgf_utils.parse_sgf(file, self.board_size, dataset_name)

    def generate_data(self, data_dir, tmp_dir, task_id=-1):
        """Generates sharded Train, Dev and Test splits of the KGS and GoGoD Datasets.

        Assumes GoGoD zip from https://gogodonline.co.uk/ and
        KGS tar.gz from https://u-go.net/gamerecords/ are already downloaded and in the tmp_dir!

        Uses split fractions defined in self.split_fractions and
        num shards per split defined in self.{train/dev/test]_shards_{gogod/kgs}.
        Args:
            data_dir: str, final data directory.
            tmp_dir: str, directory containing KGS and GoGoD zips
            task_id: int, task id.
        """
        # set random seed to make sure shuffle is recreatable
        random.seed(230)

        # unzip gogod and kgs zips if not already done
        maybe_unzip_gogod(tmp_dir)
        maybe_unzip_kgs(tmp_dir)

        # search all sgf files in the gogod dataset and shuffle the filnames
        filenames_gogod = get_gogod_filenames(tmp_dir, self.board_size)

        # split gogod filenames into train, dev and test
        train_gogod, dev_gogod, test_gogod = split_dataset(filenames_gogod, self.split_fractions)
        tf.logging.info("Split GoGoD data into train: {}, dev: {}, test: {} files!"
                        .format(len(train_gogod), len(dev_gogod), len(test_gogod)))

        train_data = [("gogod", train_gogod)]
        dev_data = [("gogod", dev_gogod)]
        test_data = [("gogod", test_gogod)]

        if self.multiple_datasets and self.board_size == 19:
            # search all sgf files in the kgs dataset and shuffle the filnames
            filenames_kgs = get_kgs_filenames(tmp_dir)

            # split kgs filenames into train, dev and test
            train_kgs, dev_kgs, test_kgs = split_dataset(filenames_kgs, self.split_fractions)
            tf.logging.info("Split KGS data into train: {}, dev: {}, test: {} files!"
                            .format(len(train_kgs), len(dev_kgs), len(test_kgs)))

            train_data.append(("kgs", train_kgs))
            dev_data.append(("kgs", dev_kgs))
            test_data.append(("kgs", test_kgs))

        # generate sharded TFRecord files of the train sgf's and shuffle
        tf.logging.info("Generating GoGoD and KGS train data")
        train_gen = self.generator(train_data)
        train_paths = self.training_filepaths(data_dir, self.train_shards, shuffled=False)
        generator_utils.generate_files(train_gen, train_paths)
        generator_utils.shuffle_dataset(train_paths)

        # generate sharded TFRecord files of the dev sgf's and shuffle
        tf.logging.info("Generating GoGoD and KGS dev data")
        dev_gen = self.generator(dev_data)
        dev_paths = self.dev_filepaths(data_dir, self.dev_shards, shuffled=False)
        generator_utils.generate_files(dev_gen, dev_paths)
        generator_utils.shuffle_dataset(dev_paths)

        # generate sharded TFRecord files of the test sgf's and shuffle
        tf.logging.info("Generating GoGoD and KGS test data")
        test_gen = self.generator(test_data)
        test_paths = self.test_filepaths(data_dir, self.test_shards, shuffled=False)
        generator_utils.generate_files(test_gen, test_paths)
        generator_utils.shuffle_dataset(test_paths)

    def example_reading_spec(self):
        data_fields = {
            'positions': tf.FixedLenFeature((), tf.string),
            'p_targets': tf.VarLenFeature(tf.int64),
            'legal_moves': tf.FixedLenFeature((), tf.string),
            'game_length': tf.FixedLenFeature((), tf.int64),
            'dataset_name': tf.FixedLenFeature((), tf.string),
            'to_play': tf.FixedLenFeature((), tf.string),
            'winner': tf.FixedLenFeature((), tf.int64),
        }
        data_items_to_decoders = {
            'inputs': NumpyHandler('positions', [-1, self.board_size, self.board_size], dtype=tf.int8),
            'p_targets': tf.contrib.slim.tfexample_decoder.Tensor('p_targets'),
            'legal_moves': NumpyHandler('legal_moves', [-1, self.num_moves], dtype=tf.uint8),
            'game_length': tf.contrib.slim.tfexample_decoder.Tensor('game_length'),
            'dataset_name': tf.contrib.slim.tfexample_decoder.Tensor('dataset_name'),
            'to_play': NumpyHandler('to_play', [-1], dtype=tf.int8),
            'winner': tf.contrib.slim.tfexample_decoder.Tensor('winner'),
        }
        return data_fields, data_items_to_decoders

    def get_hparams(self, model_hparams=None):
        """Returns problem_hparams."""
        if model_hparams is None:
            model_hparams = base_go_hparams()
        if self._hparams is not None:
            return self._hparams

        model_hparams.add_hparam("max_length", 2 * self.board_size * self.board_size)
        model_hparams.add_hparam("multiple_datasets", self.multiple_datasets)

        ret = self.hparams(model_hparams, model_hparams)
        if ret is not None:
            raise ValueError("The Problem subclass hp function should mutate "
                             "the defaults passed in and return None.")

        self._hparams = model_hparams
        return self._hparams

    def hparams(self, defaults, model_hparams):
        hp = model_hparams

        hp.add_hparam("board_size", self.board_size)
        hp.add_hparam("num_moves", self.num_moves)

        self.add_sizes(model_hparams)


class NumpyHandler(tf.contrib.slim.tfexample_decoder.ItemHandler):
    def __init__(self,
                 np_key,
                 shape,
                 dtype=tf.uint8):
        """Initializes the Numpy Handler.
       Args:
            np_key: the name of the TF-Example feature in which the encoded numpy array key
                is stored.
            shape: the output shape of the numpy array key as 1-D `Tensor`
                [game_length, 3, board_size, board_size] for go position
                [game_length, board_size * board_size + 1] for legal_moves.
            dtype: numpy array key will be decoded at this bit depth.
                    See tf.decode_raw
        """
        super(NumpyHandler, self).__init__([np_key])
        self._np_key = np_key
        self._shape = shape
        self._dtype = dtype

    def tensors_to_item(self, keys_to_tensors):
        """See base class."""
        np_buffer = keys_to_tensors[self._np_key]

        return self._decode(np_buffer)

    def _decode(self, np_buffer):
        """Decodes the numpy array buffer.
        Args:
            np_buffer: The tensor representing the encoded numpy array.
        Returns:
            A tensor that represents decoded numpy array key of self._shape.
        """
        array = tf.decode_raw(np_buffer, out_type=self._dtype)
        array = tf.reshape(array, self._shape)

        return array
