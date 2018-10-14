import tensorflow as tf
import random


from tensor2tensor.data_generators import problem
from tensor2tensor.utils import data_reader

from utils import data_utils
from data_generators import base_go_problem, go_preprocessing


class GoProblem19x19(base_go_problem.GoProblem):
    """Go Problem for 19x19 go games."""
    @property
    def board_size(self):
        return 19

    @property
    def sort_sequence_by_color(self):
        return True

    @property
    def multiple_datasets(self):
        return True

    def dataset_filename(self):
        return "go_problem_19"

    def preprocess_example(self, example, mode, hparams):
        example = go_preprocessing.format_test(example, self.board_size)

        example["inputs"].set_shape([None, 3, self.board_size, self.board_size])
        example["legal_moves"].set_shape([None, self.num_moves])
        example["p_targets"].set_shape([None])
        example["v_targets"].set_shape([None])

        example["inputs"] = tf.cast(example["inputs"], tf.float32)
        example["legal_moves"] = tf.cast(example["legal_moves"], tf.float32)
        example["v_targets"] = tf.cast(example["v_targets"], tf.float32)

        if self.sort_sequence_by_color:
            examples = go_preprocessing.split_exmaple_by_color(example)
            dataset = tf.data.Dataset.from_tensors(examples[0])

            for ex in examples[1:]:
                if mode == tf.estimator.ModeKeys.TRAIN:
                    ex = go_preprocessing.random_augmentation(ex, self.board_size)
                dat = tf.data.Dataset.from_tensors(ex)
                dataset = dataset.concatenate(dat)

            return dataset
        else:
            if mode == tf.estimator.ModeKeys.TRAIN:
                example = go_preprocessing.random_augmentation(example, self.board_size)

            return example

    def add_sizes(self, hparams):
        stats = data_utils.DatasetStats(self, hparams)
        sizes = stats.get_sizes('rnn')

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
            hparams = problem.default_model_hparams()

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
