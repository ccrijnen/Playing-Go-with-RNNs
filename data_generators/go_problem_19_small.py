import tensorflow as tf
import random

from data_generators import base_go_problem, go_preprocessing


class GoProblem19Small(base_go_problem.GoProblem):
    """Small Go Problem for 19x19 go games (~1000 games)."""
    @property
    def board_size(self):
        return 19

    def dataset_filename(self):
        fn = "go_problem_19_small"

        if self.use_kgs_data and self.use_gogod_data:
            fn += "_multi"
        elif self.use_kgs_data:
            fn += "_kgs"
        elif self.use_gogod_data:
            fn += "_gogod"

        return fn

    @property
    def train_shards(self):
        return 1

    @property
    def is_small(self):
        return True

    def generate_dataset(self, tmp_dir, unzip=True):
        # set random seed to make sure shuffle is recreatable
        random.seed(230)

        data = {
            "train": [],
            "dev": [],
            "test": []
        }
        sizes = {
            "train": 400,
            "dev": 50,
            "test": 50
        }

        if self.use_gogod_data:
            data_gogod = self.get_gogod_dataset(tmp_dir, unzip)
            for k in data:
                size = sizes[k]
                _data = data_gogod[k][0]
                data[k] += [(_data[0], _data[1][:size])]
        if self.board_size == 19:
            data_kgs = self.get_kgs_dataset(tmp_dir, unzip)
            for k in data:
                size = sizes[k]
                _data = data_kgs[k][0]
                data[k] += [(_data[0], _data[1][:size])]
        return data


class GoProblem19SmallRnn(GoProblem19Small):
    @property
    def is_recurrent(self):
        return True

    def preprocess_example(self, example, mode, hparams):
        if hasattr(hparams, "split_to_min_length") and hparams.split_to_min_length:
            # game_length = example["game_length"]
            # min_length = tf.constant(hparams.min_length, tf.int64)
            # example["game_length"] = tf.cond(game_length < hparams.min_length, lambda: game_length, lambda: min_length)
            example["game_length"] = tf.constant(hparams.min_length, tf.int64)
            example["inputs"] = example["inputs"][:hparams.min_length]
            example["to_play"] = example["to_play"][:hparams.min_length]
            example["p_targets"] = example["p_targets"][:hparams.min_length]
            example["legal_moves"] = example["legal_moves"][:hparams.min_length]

        example = go_preprocessing.format_example_rnn(example)

        example["inputs"].set_shape([None, 3, self.board_size, self.board_size])
        example["legal_moves"].set_shape([None, self.num_moves])
        example["p_targets"].set_shape([None])
        example["v_targets"].set_shape([None])

        example["inputs"] = tf.cast(example["inputs"], tf.float32)
        example["legal_moves"] = tf.cast(example["legal_moves"], tf.float32)
        example["v_targets"] = tf.cast(example["v_targets"], tf.float32)

        if self.sort_sequence_by_color:
            examples = go_preprocessing.split_exmaple_by_color(example)
            example.pop("to_play")

            if mode == tf.estimator.ModeKeys.TRAIN:
                examples[0] = go_preprocessing.random_augmentation(examples[0], self.board_size)

            dataset = tf.data.Dataset.from_tensors(examples[0])

            for ex in examples[1:]:
                if mode == tf.estimator.ModeKeys.TRAIN:
                    ex = go_preprocessing.random_augmentation(ex, self.board_size)
                dat = tf.data.Dataset.from_tensors(ex)
                dataset = dataset.concatenate(dat)
            return dataset
        else:
            example.pop("to_play")
            if mode == tf.estimator.ModeKeys.TRAIN:
                example = go_preprocessing.random_augmentation(example, self.board_size)
            return example


class GoProblem19SmallCnn(GoProblem19Small):
    @property
    def is_recurrent(self):
        return False

    def preprocess_example(self, example, mode, hparams):
        if hasattr(hparams, "split_to_min_length") and hparams.split_to_min_length:
            # game_length = example["game_length"]
            # min_length = tf.constant(hparams.min_length, tf.int64)
            # example["game_length"] = tf.cond(game_length < hparams.min_length, lambda: game_length, lambda: min_length)
            example["game_length"] = tf.constant(hparams.min_length, tf.int64)
            example["inputs"] = example["inputs"][:hparams.min_length]
            example["to_play"] = example["to_play"][:hparams.min_length]
            example["p_targets"] = example["p_targets"][:hparams.min_length]
            example["legal_moves"] = example["legal_moves"][:hparams.min_length]

        example = go_preprocessing.format_example_cnn(example, hparams)
        example.pop("to_play")

        example["inputs"].set_shape([None, hparams.history_length * 2 + 1, self.board_size, self.board_size])
        example["legal_moves"].set_shape([None, self.num_moves])

        example["inputs"] = tf.cast(example["inputs"], tf.float32)
        example["legal_moves"] = tf.cast(example["legal_moves"], tf.float32)
        example["v_targets"] = tf.cast(example["v_targets"], tf.float32)

        dataset = go_preprocessing.build_dataset_cnn(example)

        if mode == tf.estimator.ModeKeys.TRAIN:
            def _augment(ex):
                ex = go_preprocessing.random_augmentation(ex, self.board_size, "cnn")
                return ex
            dataset = dataset.map(_augment)

        return dataset
