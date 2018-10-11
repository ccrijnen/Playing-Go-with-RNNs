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

from utils import go_utils, data_utils
from models.base_go_hparams import base_go_hparams

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


def _random_augmentation(example, board_size):
    """Perform a random rotation/flip on the example.

    example["inputs"] needs to be of shape [game_length, 3, board_size, board_size].
    example["p_targets"] needs to be of shape [game_length] containing ints in [0, num_moves).
    example["legal_moves"] needs to be of shape [game_length, num_moves].

    1/8 chance to do on of:
    * do nothing
    * rotate 90° counter clockwise
    * rotate 180° counter clockwise
    * rotate 270° counter clockwise
    * flip along vertical axis
    * flip along horizontal axis
    * flip along diagonal axis from the upper left
    * flip along diagonal axis from the upper right

    Args:
        example: dict, go game
        board_size: int, board size
    Return:
        Randomly augmented example
    """
    flat_board = board_size * board_size
    num_moves = flat_board + 1

    inputs = example["inputs"]
    legal_moves = example["legal_moves"]
    p_targets = example["p_targets"]

    inputs = tf.convert_to_tensor(inputs, name='inputs')
    legal_moves = tf.convert_to_tensor(legal_moves, name='legal_moves')
    p_targets = tf.convert_to_tensor(p_targets, name='p_targets')

    rand_k = tf.random_uniform([], int(0), int(8), tf.int64, name="rand_k")

    for name, array in [["inputs", inputs], ["legal_moves", legal_moves], ["p_targets", p_targets]]:
        split = name != "inputs"
        split_p = name == "p_targets"

        if split:
            if split_p:
                # convert p_target index to one hot
                array = tf.one_hot(array, num_moves)

            # split array into a flat board and one int representing the pass move
            array, rest = tf.split(array, [flat_board, 1], 1)
            # reshape to boards
            array = tf.reshape(array, [-1, board_size, board_size])

            # need to transpose last 2 axes
            perm = [0, 2, 1]
        else:
            # need to transpose last 2 axes
            perm = [0, 1, 3, 2]

        scope = ""

        def _no_aug():
            nonlocal scope
            scope = "no_augmentation"
            return array

        def _rot90():
            nonlocal scope
            scope = "rot_90_counter"
            return tf.transpose(tf.reverse(array, [-1]), perm)

        def _rot180():
            nonlocal scope
            scope = "rot_180_counter"
            return tf.reverse(array, [-1, -2])

        def _rot270():
            nonlocal scope
            scope = "rot_270_counter"
            return tf.reverse(tf.transpose(array, perm), [-1])

        def _flip_left_right():
            nonlocal scope
            scope = "flip_left_right"
            return tf.reverse(array, [-1])

        def _flip_up_down():
            nonlocal scope
            scope = "flip_up_down"
            return tf.reverse(array, [-2])

        def _flip_diagonal_upper_left():
            nonlocal scope
            scope = "flip_diagonal_upper_left"
            return tf.transpose(array, perm)

        def _flip_diagonal_upper_right():
            nonlocal scope
            scope = "flip_diagonal_upper_right"
            return tf.transpose(tf.reverse(array, [-1, -2]), perm)

        cases = [
            (tf.equal(rand_k, 0), _no_aug),
            (tf.equal(rand_k, 1), _rot90),
            (tf.equal(rand_k, 2), _rot180),
            (tf.equal(rand_k, 3), _rot270),
            (tf.equal(rand_k, 4), _flip_left_right),
            (tf.equal(rand_k, 5), _flip_up_down),
            (tf.equal(rand_k, 6), _flip_diagonal_upper_left),
            (tf.equal(rand_k, 7), _flip_diagonal_upper_right)
        ]

        result = tf.case(cases, name=scope)

        if split:
            # reassemble the original shape from combined result tensor and rest tensor
            result = tf.reshape(result, [-1, flat_board])
            result = tf.concat([result, rest], 1)
            if split_p:
                result = tf.argmax(result, 1)

        example[name] = result

    return example


def split_exmaple(example):
    inputs = example["inputs"]
    legal_moves = example["legal_moves"]
    p_targets = example["p_targets"]
    v_targets = example["v_targets"]

    mask_black = tf.equal(inputs[:, 2, 0, 0], 1)

    game_length = tf.boolean_mask(inputs, mask_black)
    game_length = tf.shape(game_length)[0]

    example1 = {
        "inputs": tf.boolean_mask(inputs, mask_black),
        "legal_moves": tf.boolean_mask(legal_moves, mask_black),
        "p_targets": tf.boolean_mask(p_targets, mask_black),
        "v_targets": tf.boolean_mask(v_targets, mask_black),
        "game_lengt": game_length
    }

    mask_white = tf.equal(inputs[:, 2, 0, 0], 0)

    game_length = tf.boolean_mask(inputs, mask_white)
    game_length = tf.shape(game_length)[0]

    example2 = {
        "inputs": tf.boolean_mask(inputs, mask_white),
        "legal_moves": tf.boolean_mask(legal_moves, mask_white),
        "p_targets": tf.boolean_mask(p_targets, mask_white),
        "v_targets": tf.boolean_mask(v_targets, mask_white),
        "game_lengt": game_length
    }

    return example


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
        return False

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
    def split_fractions(self):
        return {
            problem.DatasetSplit.TRAIN: 0.8,
            problem.DatasetSplit.EVAL: 0.1,
            problem.DatasetSplit.TEST: 0.1
        }

    def add_sizes(self, hparams):
        raise NotImplementedError

    def generator(self, paths):
        """Go game generator from sgf format.

        Args:
            paths: str, paths to sgf files to generate go games from

        Yields:
            A dictionary representing a go game with the following fields:
            * positions: str of [game_length, 3, board_size, board_size] np.array, encoded game positions
            * p_targets: [game_length] int list, index of the played move (incl. pass move)
            * v_targets: [game_length] int list, winner of the game, 1 if current player is winning, -1 otherwise
            * legal_moves: str of [game_length, num_moves] np.array, encoded legal_moves at every position
            * game_length: int, game length
            Fields positions, legal_moves, game_length is actually a list of the corresponding type.
        """
        for path in paths:
            yield go_utils.parse_sgf(path, self.board_size, self.sort_sequence_by_color)

    def example_reading_spec(self):
        data_fields = {
            'positions': tf.FixedLenFeature((), tf.string),
            'p_targets': tf.VarLenFeature(tf.int64),
            'v_targets': tf.VarLenFeature(tf.int64),
            'legal_moves': tf.FixedLenFeature((), tf.string),
            'game_length': tf.FixedLenFeature((), tf.int64)
        }
        data_items_to_decoders = {
            'inputs': NumpyHandler('positions', [-1, 3, self.board_size, self.board_size]),
            'p_targets': tf.contrib.slim.tfexample_decoder.Tensor('p_targets'),
            'v_targets': tf.contrib.slim.tfexample_decoder.Tensor('v_targets'),
            'legal_moves': NumpyHandler('legal_moves', [-1, self.num_moves]),
            'game_length': tf.contrib.slim.tfexample_decoder.Tensor('game_length')
        }
        return data_fields, data_items_to_decoders

    def preprocess_example(self, example, mode, hparams):
        example["inputs"].set_shape([None, 3, self.board_size, self.board_size])
        example["legal_moves"].set_shape([None, self.num_moves])
        example["p_targets"].set_shape([None])
        example["v_targets"].set_shape([None])

        if mode == tf.estimator.ModeKeys.TRAIN:
            example = _random_augmentation(example, self.board_size)

        example["inputs"] = tf.cast(example["inputs"], tf.float32)
        example["legal_moves"] = tf.cast(example["legal_moves"], tf.float32)
        example["v_targets"] = tf.cast(example["v_targets"], tf.float32)

        return example

    def get_hparams(self, model_hparams=None):
        """Returns problem_hparams."""
        if model_hparams is None:
            model_hparams = base_go_hparams()
        if self._hparams is not None:
            return self._hparams

        ret = self.hparams(model_hparams, model_hparams)
        if ret is not None:
            raise ValueError("The Problem subclass hparams function should mutate "
                             "the defaults passed in and return None.")

        self._hparams = model_hparams
        return self._hparams

    def hparams(self, defaults, model_hparams):
        hp = model_hparams

        hp.add_hparam("board_size", self.board_size)
        hp.add_hparam("num_moves", self.num_moves)

        hp.max_length = self.board_size * self.board_size * 2

        self.add_sizes(model_hparams)


class GoProblem19x19(GoProblem):
    """Go Problem for 19x19 go games."""
    @property
    def board_size(self):
        return 19

    @property
    def train_shards(self):
        return 80

    @property
    def dev_shards_gogod(self):
        return 5

    @property
    def dev_shards_kgs(self):
        return 5

    @property
    def test_shards_gogod(self):
        return 5

    @property
    def test_shards_kgs(self):
        return 5

    @property
    def sort_sequence_by_color(self):
        return True

    @property
    def multiple_datasets(self):
        return True

    def dataset_filename(self):
        return "go_problem_19"

    @property
    def batch_size_means_tokens(self):
        """Do we specify hparams.batch_size in tokens per datashard per batch.
        This is generally done for text problems.
        If False, we assume that batch sizes are specified in examples per
        datashard per batch.
        Returns:
          a boolean
        """
        return True

    def add_sizes(self, hparams):
        stats = data_utils.DatasetStats(self, hparams)
        sizes = stats.get_sizes('rnn')

        for k, v in sizes.items():
            hparams.add_hparam(k, v)

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

        # search all sgf files in the kgs dataset and shuffle the filnames
        filenames_kgs = get_kgs_filenames(tmp_dir)

        # split gogod filenames into train, dev and test
        train_gogod, dev_gogod, test_gogod = split_dataset(filenames_gogod, self.split_fractions)
        tf.logging.info("Split GoGoD data into train: {}, dev: {}, test: {} files!".format(len(train_gogod),
                                                                                           len(dev_gogod),
                                                                                           len(test_gogod)))

        # split kgs filenames into train, dev and test
        train_kgs, dev_kgs, test_kgs = split_dataset(filenames_kgs, self.split_fractions)
        tf.logging.info("Split KGS data into train: {}, dev: {}, test: {} files!".format(len(train_kgs),
                                                                                         len(dev_kgs),
                                                                                         len(test_kgs)))

        # merge train data from both datasets into one
        train_data = train_gogod + train_kgs

        # generate sharded TFRecord files of the train sgf's and shuffle
        tf.logging.info("Generating GoGoD and KGS train data")
        train_gen = self.generator(train_data)
        train_paths = self.training_filepaths(data_dir, self.train_shards, shuffled=False)
        generator_utils.generate_files(train_gen, train_paths)
        generator_utils.shuffle_dataset(train_paths)

        # generate sharded TFRecord files of the gogod dev sgf's
        tf.logging.info("Generating GoGoD dev data")
        dev_gen_gogod = self.generator(dev_gogod)
        dev_paths_gogod = self.dev_filepaths_gogod(data_dir, self.dev_shards_gogod, shuffled=True)
        generator_utils.generate_files(dev_gen_gogod, dev_paths_gogod)

        # generate sharded TFRecord files of the gogod test sgf's
        tf.logging.info("Generating GoGoD test data")
        test_gen_gogod = self.generator(test_gogod)
        test_paths_gogod = self.test_filepaths_gogod(data_dir, self.test_shards_gogod, shuffled=True)
        generator_utils.generate_files(test_gen_gogod, test_paths_gogod)

        # generate sharded TFRecord files of the kgs dev sgf's
        tf.logging.info("Generating KGS dev data")
        dev_gen_kgs = self.generator(dev_kgs)
        dev_paths_kgs = self.dev_filepaths_kgs(data_dir, self.dev_shards_kgs, shuffled=True)
        generator_utils.generate_files(dev_gen_kgs, dev_paths_kgs)

        # generate sharded TFRecord files of the kgs test sgf's
        tf.logging.info("Generating KGS test data")
        test_gen_kgs = self.generator(test_kgs)
        test_paths_kgs = self.test_filepaths_kgs(data_dir, self.test_shards_kgs, shuffled=True)
        generator_utils.generate_files(test_gen_kgs, test_paths_kgs)

    def training_filepaths(self, data_dir, num_shards, shuffled, file_suffix=""):
        file_basename = self.dataset_filename() + file_suffix
        if not shuffled:
            file_basename += generator_utils.UNSHUFFLED_SUFFIX
        return generator_utils.train_data_filenames(file_basename, data_dir, num_shards)

    def dev_filepaths_gogod(self, data_dir, num_shards, shuffled):
        return self.dev_filepaths(data_dir, num_shards, shuffled, "-gogod")

    def dev_filepaths_kgs(self, data_dir, num_shards, shuffled):
        return self.dev_filepaths(data_dir, num_shards, shuffled, "-kgs")

    def dev_filepaths(self, data_dir, num_shards, shuffled, file_suffix=""):
        file_basename = self.dataset_filename() + file_suffix
        if not shuffled:
            file_basename += generator_utils.UNSHUFFLED_SUFFIX
        return generator_utils.dev_data_filenames(file_basename, data_dir, num_shards)

    def test_filepaths_gogod(self, data_dir, num_shards, shuffled):
        return self.test_filepaths(data_dir, num_shards, shuffled, "-gogod")

    def test_filepaths_kgs(self, data_dir, num_shards, shuffled):
        return self.test_filepaths(data_dir, num_shards, shuffled, "-kgs")

    def test_filepaths(self, data_dir, num_shards, shuffled, file_suffix=""):
        file_basename = self.dataset_filename() + file_suffix
        if not shuffled:
            file_basename += generator_utils.UNSHUFFLED_SUFFIX
        return generator_utils.test_data_filenames(file_basename, data_dir, num_shards)

    def filepattern(self, data_dir, mode, shard=None, dataset_suffix=""):
        """Get filepattern for data files for mode.
        Matches mode to a suffix.
        * DatasetSplit.TRAIN: train
        * DatasetSplit.EVAL: dev
        * DatasetSplit.TEST: test
        * tf.estimator.ModeKeys.PREDICT: dev
        Args:
            data_dir: str, data directory
            mode: DatasetSplit
            shard: int, if provided, will only read data from the specified shard.
            dataset_suffix: str, if provided, will only read from the specified dataset
        Returns:
            filepattern str
        """
        path = os.path.join(data_dir, self.dataset_filename())
        shard_str = "-%05d" % shard if shard is not None else ""
        if mode == problem.DatasetSplit.TRAIN:
            suffix = "train"
        elif mode in [problem.DatasetSplit.EVAL, tf.estimator.ModeKeys.PREDICT]:
            suffix = "dev"
        else:
            assert mode == problem.DatasetSplit.TEST
            suffix = "test"

        return "{}{}-{}{}*".format(path, dataset_suffix, suffix, shard_str)

    def dataset(self,
                mode,
                data_dir=None,
                num_threads=None,
                output_buffer_size=None,
                shuffle_files=None,
                hparams=None,
                preprocess=True,
                dataset_split=None,
                dataset_suffix="",
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
            hparams: tf.contrib.training.HParams; hparams to be passed to
                Problem.preprocess_example and Problem.hparams. If None, will use a
                default set that is a no-op.
            preprocess: bool, whether to map the Dataset through
                Problem.preprocess_example.
            dataset_split: DatasetSplit, which split to read data
                from (TRAIN:"-train", EVAL:"-dev", "test":"-test"). Defaults to mode.
            dataset_suffix: str, if provided, will only read from the specified dataset
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
            hparams = problem.default_model_hparams()

        if not hasattr(hparams, "data_dir"):
            hparams.add_hparam("data_dir", data_dir)
        if not hparams.data_dir:
            hparams.data_dir = data_dir
        # Construct the Problem's hparams so that items within it are accessible
        _ = self.get_hparams(hparams)

        data_filepattern = self.filepattern(data_dir, dataset_split, shard=shard, dataset_suffix=dataset_suffix)
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
                 force_repeat=False,
                 prevent_repeat=False,
                 dataset_suffix="",
                 dataset_kwargs=None):
        """Builds input pipeline for problem.
        Args:
            mode: tf.estimator.ModeKeys
            hparams: HParams, model hparams
            data_dir: str, data directory; if None, will use hparams.data_dir
            force_repeat: bool, whether to repeat the data even if not training
            prevent_repeat: bool, whether to not repeat when in training mode.
                Overrides force_repeat.
            dataset_suffix: str, if provided, will only read from the specified dataset
            dataset_kwargs: dict, if passed, will pass as kwargs to self.dataset
                method when called
        Returns:
            (features_dict<str name, Tensor feature>, Tensor targets)
        """
        is_training = mode == tf.estimator.ModeKeys.TRAIN
        num_threads = problem.cpu_count() if is_training else 1

        def gpu_valid_size(example):
            return data_utils.example_valid_size(example, hparams.min_length)

        # Read and preprocess
        data_dir = data_dir or (hasattr(hparams, "data_dir") and hparams.data_dir)

        dataset_kwargs = dataset_kwargs or {}
        dataset_kwargs.update({
            "mode": mode,
            "data_dir": data_dir,
            "num_threads": num_threads,
            "hparams": hparams,
            "dataset_suffix": dataset_suffix
        })

        dataset = self.dataset(**dataset_kwargs)
        if (force_repeat or is_training) and not prevent_repeat:
            # Repeat and skip a random number of records
            dataset = dataset.repeat()

        if is_training:
            data_files = tf.contrib.slim.parallel_reader.get_data_files(
                self.filepattern(data_dir, mode))
            #  In continuous_train_and_eval when switching between train and
            #  eval, this input_fn method gets called multiple times and it
            #  would give you the exact same samples from the last call
            #  (because the Graph seed is set). So this skip gives you some
            #  shuffling.
            dataset = problem.skip_random_fraction(dataset, data_files[0])

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
