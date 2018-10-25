import tensorflow as tf
import zipfile
import tarfile
import random
import math
import os
import re

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.utils import data_reader

from models.go_hparams import base_go_hparams
from utils import data_utils, sgf_utils

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
        path, file = os.path.split(file)

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
    def train_shards(self):
        raise NotImplementedError

    @property
    def is_small(self):
        raise NotImplementedError

    @property
    def is_recurrent(self):
        raise NotImplementedError

    def generate_dataset(self, tmp_dir, unzip=True):
        raise NotImplementedError

    @property
    def num_moves(self):
        """Equivalent to num_classes."""
        return self.board_size * self.board_size + 1

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

    @property
    def use_gogod_data(self):
        return self._use_gogod_data

    @use_gogod_data.setter
    def use_gogod_data(self, use_gogod_data):
        self._use_gogod_data = use_gogod_data

    @property
    def use_kgs_data(self):
        return self._use_kgs_data

    @use_kgs_data.setter
    def use_kgs_data(self, use_kgs_data):
        if self.board_size == 19:
            self._use_kgs_data = use_kgs_data
        else:
            self._use_kgs_data = False

    @property
    def sort_sequence_by_color(self):
        return self._sort_sequence_by_color

    @sort_sequence_by_color.setter
    def sort_sequence_by_color(self, sort_sequence_by_color):
        self._sort_sequence_by_color = sort_sequence_by_color

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

    def get_gogod_dataset(self, tmp_dir, unzip=True):
        # unzip gogod zips if not already done
        if unzip:
            maybe_unzip_gogod(tmp_dir)

        # search all sgf files in the gogod dataset and shuffle the filnames
        filenames_gogod = get_gogod_filenames(tmp_dir, self.board_size)

        # split gogod filenames into train, dev and test
        train_gogod, dev_gogod, test_gogod = split_dataset(filenames_gogod, self.split_fractions)
        tf.logging.info("Split GoGoD data into train: {}, dev: {}, test: {} files!"
                        .format(len(train_gogod), len(dev_gogod), len(test_gogod)))

        return {
            "train": [("gogod", train_gogod)],
            "dev": [("gogod", dev_gogod)],
            "test": [("gogod", test_gogod)]
        }

    def get_kgs_dataset(self, tmp_dir, unzip=True):
        # unzip kgs zips if not already done
        if unzip:
            maybe_unzip_kgs(tmp_dir)

        # search all sgf files in the kgs dataset and shuffle the filnames
        filenames_kgs = get_kgs_filenames(tmp_dir)

        # split kgs filenames into train, dev and test
        train_kgs, dev_kgs, test_kgs = split_dataset(filenames_kgs, self.split_fractions)
        tf.logging.info("Split KGS data into train: {}, dev: {}, test: {} files!"
                        .format(len(train_kgs), len(dev_kgs), len(test_kgs)))

        return {
            "train": [("kgs", train_kgs)],
            "dev": [("kgs", dev_kgs)],
            "test": [("kgs", test_kgs)]
        }

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
        data = self.generate_dataset(tmp_dir)

        for k, v in data.items():
            if v == []:
                raise ValueError("No {} files found!".format(k))

        # generate sharded TFRecord files of the train sgf's and shuffle
        tf.logging.info("Generating GoGoD and KGS train data")
        train_gen = self.generator(data["train"])
        train_paths = self.training_filepaths(data_dir, self.train_shards, shuffled=False)
        generator_utils.generate_files(train_gen, train_paths)
        generator_utils.shuffle_dataset(train_paths)

        # generate sharded TFRecord files of the dev sgf's and shuffle
        tf.logging.info("Generating GoGoD and KGS dev data")
        dev_gen = self.generator(data["dev"])
        dev_paths = self.dev_filepaths(data_dir, self.dev_shards, shuffled=False)
        generator_utils.generate_files(dev_gen, dev_paths)
        generator_utils.shuffle_dataset(dev_paths)

        # generate sharded TFRecord files of the test sgf's and shuffle
        tf.logging.info("Generating GoGoD and KGS test data")
        test_gen = self.generator(data["test"])
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

    def get_hparams(self, hparams=None):
        """Returns problem_hparams."""
        if hparams is None:
            hparams = base_go_hparams()
        if self._hparams is not None:
            return self._hparams

        hparams.add_hparam("max_length", 2 * self.board_size * self.board_size)
        hparams.add_hparam("board_size", self.board_size)
        hparams.add_hparam("num_moves", self.num_moves)

        if hasattr(hparams, "sort_sequence_by_color"):
            if self.is_recurrent:
                self.sort_sequence_by_color = hparams.sort_sequence_by_color
            else:
                self.sort_sequence_by_color = False
        else:
            self.sort_sequence_by_color = False

        if hasattr(hparams, "use_gogod_data"):
            self.use_gogod_data = hparams.use_gogod_data
        else:
            self.use_gogod_data = False

        if hasattr(hparams, "use_kgs_data"):
            self.use_kgs_data = hparams.use_kgs_data
        else:
            self.use_kgs_data = False

        ret = self.add_hparams(hparams)
        if ret is not None:
            raise ValueError("The Problem subclass hp function should mutate "
                             "the defaults passed in and return None.")

        if self.sort_sequence_by_color:
            hparams.min_length = hparams.min_length // 2
            hparams.max_length = hparams.max_length // 2

        self._hparams = hparams
        return self._hparams

    def add_hparams(self, hparams):
        stats = data_utils.DatasetStats(self, hparams)
        if self.is_recurrent:
            sizes = stats.get_sizes('rnn')
        else:
            sizes = stats.get_sizes('cnn')

        for k, v in sizes.items():
            hparams.add_hparam(k, v)

    def dataset(self,
                mode,
                data_dir=None,
                num_threads=None,
                output_buffer_size=None,
                shuffle_files=None,
                hparams=None,
                preprocess=True,
                dataset_split=None,
                shard=None,
                partition_id=0,
                num_partitions=1,
                max_records=-1,
                only_last=False):
        """Build a Dataset for this problem.
        Args:
            mode: tf.estimator.ModeKeys; determines which files to read from.
            data_dir: directory that contains data files.
            num_threads: int, number of threads to use for decode and preprocess
                Dataset.map calls.
            output_buffer_size: int, how many elements to prefetch at end of pipeline.
            shuffle_files: whether to shuffle input files. Default behavior (i.e. when
                shuffle_files=None) is to shuffle if mode == TRAIN.
            hparams: tf.contrib.training.HParams; hp to be passed to
                Problem.preprocess_example and Problem.hp. If None, will use a
                default set that is a no-op.
            preprocess: bool, whether to map the Dataset through
                Problem.preprocess_example.
            dataset_split: DatasetSplit, which split to read data
                from (TRAIN:"-train", EVAL:"-dev", "test":"-test"). Defaults to mode.
            shard: int, if provided, will only read data from the specified shard.
            partition_id: integer - which partition of the dataset to read from
            num_partitions: how many partitions in the dataset
            max_records: int, number of records to truncate to.
            only_last: bool, whether we should include only files from last epoch.
        Returns:
            Dataset containing dict<feature name, Tensor>.
        Raises:
            ValueError: if num_partitions is greater than the number of data files.
        """
        is_training = mode == tf.estimator.ModeKeys.TRAIN
        shuffle_files = shuffle_files or shuffle_files is None and is_training

        dataset_split = dataset_split or mode
        assert data_dir

        if hparams is None:
            hparams = base_go_hparams()

        if not hasattr(hparams, "data_dir"):
            hparams.add_hparam("data_dir", data_dir)
        if not hparams.data_dir:
            hparams.data_dir = data_dir
        # Construct the Problem's hp so that items within it are accessible
        _ = self.get_hparams(hparams)

        data_filepattern = self.filepattern(data_dir, dataset_split, shard=shard)
        if only_last:
            imprv_data_filepattern = data_filepattern + r"10.[\d+]"
        else:
            imprv_data_filepattern = data_filepattern
        tf.logging.info("Reading data files from %s", data_filepattern)
        try:
            data_files = sorted(tf.contrib.slim.parallel_reader.get_data_files(
                imprv_data_filepattern))
        except ValueError:
            data_files = sorted(tf.contrib.slim.parallel_reader.get_data_files(
                data_filepattern))

        # Functions used in dataset transforms below. `filenames` can be either a
        # `tf.string` tensor or `tf.data.Dataset` containing one or more filenames.
        def _load_records_and_preprocess(filenames):
            """Reads files from a string tensor or a dataset of filenames."""
            # Load records from file(s) with an 8MiB read buffer.
            _dataset = tf.data.TFRecordDataset(filenames, buffer_size=8 * 1024 * 1024)
            # Decode.
            _dataset = _dataset.map(self.decode_example, num_parallel_calls=num_threads)
            # Preprocess if requested.
            # Note that preprocessing should happen per-file as order may matter.
            if preprocess:
                _dataset = self.preprocess(_dataset, mode, hparams, interleave=shuffle_files)
            return _dataset

        if len(data_files) < num_partitions:
            raise ValueError(
                "number of data files (%d) must be at least the number of hosts (%d)"
                % (len(data_files), num_partitions))
        data_files = [f for (i, f) in enumerate(data_files)
                      if i % num_partitions == partition_id]
        tf.logging.info(
            "partition: %d num_data_files: %d" % (partition_id, len(data_files)))
        if shuffle_files:
            random.shuffle(data_files)

        dataset = tf.data.Dataset.from_tensor_slices(tf.constant(data_files))
        # Create data-set from files by parsing, pre-processing and interleaving.
        if shuffle_files:
            dataset = dataset.apply(
                tf.contrib.data.parallel_interleave(
                    _load_records_and_preprocess, sloppy=True, cycle_length=8))
        else:
            dataset = _load_records_and_preprocess(dataset)

        dataset = dataset.take(max_records)
        if output_buffer_size:
            dataset = dataset.prefetch(output_buffer_size)

        return dataset

    def input_fn(self,
                 mode,
                 hparams,
                 data_dir=None,
                 params=None,
                 force_repeat=False,
                 prevent_repeat=False,
                 dataset_kwargs=None):
        """Builds input pipeline for problem.
        Args:
            mode: tf.estimator.ModeKeys
            hparams: HParams, model hp
            data_dir: str, data directory; if None, will use hp.data_dir
            params: dict, may include "batch_size"
            force_repeat: bool, whether to repeat the data even if not training
            prevent_repeat: bool, whether to not repeat when in training mode.
                Overrides force_repeat.
            dataset_kwargs: dict, if passed, will pass as kwargs to self.dataset
                method when called
        Returns:
            (features_dict<str name, Tensor feature>, Tensor targets)
        """
        is_training = mode == tf.estimator.ModeKeys.TRAIN
        num_threads = problem.cpu_count() if is_training else 1

        def gpu_valid_size(example):
            return data_utils.example_valid_size(example, hparams.min_length, hparams.max_length)

        # Read and preprocess
        data_dir = data_dir or (hasattr(hparams, "data_dir") and hparams.data_dir)

        dataset_kwargs = dataset_kwargs or {}
        dataset_kwargs.update({
            "mode": mode,
            "data_dir": data_dir,
            "num_threads": num_threads,
            "hparams": hparams
        })

        dataset = self.dataset(**dataset_kwargs)
        if (force_repeat or is_training) and not prevent_repeat:
            # Repeat and skip a random number of records
            dataset = dataset.repeat()

        dataset = dataset.map(
            data_reader.cast_ints_to_int32, num_parallel_calls=num_threads)

        dataset = dataset.filter(gpu_valid_size)

        dataset = dataset.apply(
            tf.contrib.data.bucket_by_sequence_length(
                data_utils.example_length, [], [hparams.batch_size]))

        def prepare_for_output(example):
            problem._summarize_features(example, 1)
            return example

        dataset = dataset.map(prepare_for_output, num_parallel_calls=num_threads)
        dataset = dataset.prefetch(2)

        if mode == tf.estimator.ModeKeys.PREDICT:
            # This is because of a bug in the Estimator that short-circuits prediction
            # if it doesn't see a QueueRunner. DummyQueueRunner implements the
            # minimal expected interface but does nothing.
            tf.add_to_collection(tf.GraphKeys.QUEUE_RUNNERS,
                                 data_reader.DummyQueueRunner())

        return dataset


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
