import tensorflow as tf
import numpy as np
import os

from utils import utils
from tqdm import trange
from tensor2tensor.data_generators.problem import DatasetSplit
from data_generators import go_problem
from models.base_go_model import GoModel


class GoTrainer:
    def __init__(self, problem, model, hparams):
        assert isinstance(problem, go_problem.GoProblem)
        self.hparams = hparams
        self.multiple_datasets = problem.multiple_datasets
        if self.multiple_datasets:
            assert isinstance(problem, go_problem.GoProblem19x19)

        self.problem = problem

        assert isinstance(model, GoModel)
        self.model = model

    def train_epoch(self, sess, model_spec, num_steps, writer):
        """Train the model on `num_steps` batches
        Args:
            sess: (tf.Session) current session
            model_spec: (dict) contains the graph operations or nodes needed for training
            num_steps: (int) train for this number of batches
            writer: (tf.summary.FileWriter) writer for summaries
        """
        hp = self.hparams

        # Get relevant graph operations or nodes needed for training
        loss = model_spec['loss']

        train_op = model_spec['train_op']
        update_metrics = model_spec['update_metrics']
        metrics = model_spec['metrics']
        summary_op = model_spec['summary_op']
        global_step = tf.train.get_global_step()

        # Load the training dataset into the pipeline and initialize the metrics local variables
        sess.run(model_spec['iterator_init_op'])
        sess.run(model_spec['metrics_init_op'])

        # Use tqdm for progress bar
        t = trange(num_steps)
        for i in t:
            # Evaluate summaries for tensorboard only once in a while
            if i % hp.save_summary_steps == 0:
                # Perform a mini-batch update
                _, _, loss_val, summaries, global_step_val = sess.run([train_op, update_metrics, loss,
                                                                       summary_op, global_step])
                # Write summaries for tensorboard
                writer.add_summary(summaries, global_step_val)
            else:
                _, _, loss_val = sess.run([train_op, update_metrics, loss])
            # Log the loss in the tqdm progress bar
            t.set_postfix(loss='{:05.3f}'.format(loss_val))

        metrics_values = {k: v[0] for k, v in metrics.items()}
        metrics_val = sess.run(metrics_values)
        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_val.items())
        tf.logging.info("- Train metrics: " + metrics_string)

    def evaluate_epoch(self, sess, model_spec, num_steps, writer=None):
        """Train the model on `num_steps` batches.
        Args:
            sess: (tf.Session) current session
            model_spec: (dict) contains the graph operations or nodes needed for training
            num_steps: (int) train for this number of batches
            writer: (tf.summary.FileWriter) writer for summaries. Is None if we don't log anything
        """
        hp = self.hparams

        update_metrics = model_spec['update_metrics']
        eval_metrics = model_spec['metrics']
        global_step = tf.train.get_global_step()

        # Load the evaluation dataset into the pipeline and initialize the metrics init op
        sess.run(model_spec['iterator_init_op'])
        sess.run(model_spec['metrics_init_op'])

        # compute metrics over the dataset
        for _ in range(num_steps):
            sess.run(update_metrics)

        # Get the values of the metrics
        metrics_values = {k: v[0] for k, v in eval_metrics.items()}
        metrics_val = sess.run(metrics_values)
        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_val.items())
        tf.logging.info("- Eval metrics: " + metrics_string)

        # Add summaries manually to writer at global_step_val
        if writer is not None:
            global_step_val = sess.run(global_step)
            for tag, val in metrics_val.items():
                summ = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=val)])
                writer.add_summary(summ, global_step_val)

        return metrics_val

    def train_and_evaluate(self, restore_from=None):
        """Train the model and evaluate every epoch.
        Args:
            restore_from: (string) directory or file containing weights to restore the graph
        """
        hp = self.hparams
        experiment_dir = hp.experiment_dir

        tf.logging.info("Starting training for {} epoch(s)".format(hp.num_epochs))

        split = DatasetSplit.TRAIN
        mode = self._split_to_mode(split)
        with tf.variable_scope('train_datasets'):
            train_datasets = self._get_datasets(split)
        train_model_spec = self._get_model_spec(train_datasets, mode)

        split = DatasetSplit.EVAL
        mode = self._split_to_mode(split)
        with tf.variable_scope('eval_datasets'):
            eval_datasets = self._get_datasets(split)
        eval_model_specs = self._get_model_spec(eval_datasets, mode)

        # Initialize tf.Saver instances to save weights during training
        last_saver = tf.train.Saver()  # will keep last 5 epochs
        best_saver = tf.train.Saver(max_to_keep=1)  # only keep 1 best checkpoint (best on eval)
        begin_at_epoch = 0

        with tf.Session() as sess:
            # Initialize model variables
            sess.run(train_model_spec['variable_init_op'])

            # Reload weights from directory if specified
            if restore_from is not None:
                tf.logging.info("Restoring parameters from {}".format(restore_from))
                if os.path.isdir(restore_from):
                    restore_from = tf.train.latest_checkpoint(restore_from)
                    begin_at_epoch = int(restore_from.split('-')[-1])
                last_saver.restore(sess, restore_from)

            # For tensorboard (takes care of writing summaries to files)
            train_writer = tf.summary.FileWriter(os.path.join(experiment_dir, 'train_summaries'), sess.graph)

            if self.multiple_datasets:
                eval_writers = {
                    "gogod": tf.summary.FileWriter(os.path.join(experiment_dir, 'eval_summaries_gogod'), sess.graph),
                    "kgs": tf.summary.FileWriter(os.path.join(experiment_dir, 'eval_summaries_kgs'), sess.graph)
                }
            else:
                eval_writers = tf.summary.FileWriter(os.path.join(experiment_dir, 'eval_summaries'), sess.graph)

            best_accs = BestAccs(self.multiple_datasets)

            for epoch in range(begin_at_epoch, begin_at_epoch + hp.num_epochs):
                # Run one epoch
                tf.logging.info("Epoch {}/{}".format(epoch + 1, begin_at_epoch + hp.num_epochs))
                # Compute number of batches in one epoch (one full pass over the training set)
                num_steps = self._num_steps(hp.train_size)
                self.train_epoch(sess, train_model_spec, num_steps, train_writer)

                # Save weights
                last_save_path = os.path.join(experiment_dir, 'last_weights', 'after-epoch')
                tf.gfile.MakeDirs(os.path.join(experiment_dir, 'last_weights'))
                last_saver.save(sess, last_save_path, global_step=epoch + 1)

                # Evaluate for one epoch on validation set
                if isinstance(eval_model_specs, dict):
                    num_steps = {
                        "gogod": self._num_steps(hp.gogod_dev_size),
                        "kgs": self._num_steps(hp.kgs_dev_size)
                    }

                    metrics = {}
                    for dataset_name, eval_model_spec in eval_model_specs.items():
                        eval_writer = eval_writers[dataset_name]
                        temp_metrics = self.evaluate_epoch(sess, eval_model_spec, num_steps[dataset_name], eval_writer)
                        for metric_name, value in temp_metrics.items():
                            metrics[dataset_name + "_" + metric_name] = value
                else:
                    num_steps = self._num_steps(hp.dev_size)
                    eval_model_spec = eval_model_specs
                    metrics = self.evaluate_epoch(sess, eval_model_spec, num_steps, eval_writers)

                # If best_eval, best_save_path
                cond = best_accs.get_condition(metrics)
                if cond:
                    # Store new best accuracy
                    best_accs.update_accs(metrics)
                    # Save weights
                    best_save_path = os.path.join(experiment_dir, 'best_weights', 'after-epoch')
                    tf.gfile.MakeDirs(os.path.join(experiment_dir, 'best_weights'))
                    best_save_path = best_saver.save(sess, best_save_path, global_step=epoch + 1)
                    tf.logging.info("- Found new best accuracy, saving in {}".format(best_save_path))
                    # Save best eval metrics in a json file in the model directory
                    best_json_path = os.path.join(experiment_dir, "metrics_eval_best_weights.json")
                    utils.save_dict_to_json(metrics, best_json_path)

                # Save latest eval metrics in a json file in the model directory
                last_json_path = os.path.join(experiment_dir, "metrics_eval_last_weights.json")
                utils.save_dict_to_json(metrics, last_json_path)

    def test(self, restore_from):
        """Test the model
        Args:
            restore_from: (string) directory or file containing weights to restore the graph
        """
        hp = self.hparams
        experiment_dir = hp.experiment_dir

        split = DatasetSplit.TEST
        mode = self._split_to_mode(split)
        with tf.variable_scope('test_datasets'):
            datasets = self._get_datasets(split)
        model_specs = self._get_model_spec(datasets, mode)

        # Initialize tf.Saver
        saver = tf.train.Saver()

        with tf.Session() as sess:
            # Initialize the lookup table
            if isinstance(model_specs, dict):
                for _, model_spec in model_specs.items():
                    sess.run(model_spec['variable_init_op'])
            else:
                sess.run(model_specs['variable_init_op'])

            # Reload weights from the weights subdirectory
            save_path = os.path.join(experiment_dir, restore_from)
            if os.path.isdir(save_path):
                save_path = tf.train.latest_checkpoint(save_path)
            saver.restore(sess, save_path)

            # Evaluate
            if isinstance(model_specs, dict):
                num_steps = {
                    "gogod": self._num_steps(hp.gogod_test_size),
                    "kgs": self._num_steps(hp.kgs_test_size)
                }

                metrics = {}
                for dataset_name, test_model_spec in model_specs.items():
                    temp_metrics = self.evaluate_epoch(sess, test_model_spec, num_steps[dataset_name])
                    for metric_name, value in temp_metrics.items():
                        metrics[dataset_name + "_" + metric_name] = value
            else:
                num_steps = self._num_steps(hp.test_size)
                test_model_spec = model_specs
                metrics = self.evaluate_epoch(sess, test_model_spec, num_steps)

            metrics_name = '_'.join(restore_from.split('/'))
            save_path = os.path.join(experiment_dir, "metrics_test_{}.json".format(metrics_name))
            utils.save_dict_to_json(metrics, save_path)

    def _num_steps(self, size):
        hp = self.hparams
        return (size + hp.batch_size - 1) // hp.batch_size

    @staticmethod
    def _split_to_mode(split):
        split_to_mode = {
            DatasetSplit.TRAIN: tf.estimator.ModeKeys.TRAIN,
            DatasetSplit.EVAL: tf.estimator.ModeKeys.EVAL,
            DatasetSplit.TEST: tf.estimator.ModeKeys.EVAL
        }
        return split_to_mode[split]

    def _get_datasets(self, dataset_split):
        problem = self.problem
        hp = self.hparams

        mode = self._split_to_mode(dataset_split)

        dataset_kwargs = {
            "dataset_split": dataset_split
        }

        if self.multiple_datasets and not dataset_split == DatasetSplit.TRAIN:
            gogod_dataset = problem.input_fn(mode, hp, dataset_suffix="-gogod", dataset_kwargs=dataset_kwargs,
                                             prevent_repeat=True)
            kgs_dataset = problem.input_fn(mode, hp, dataset_suffix="-kgs", dataset_kwargs=dataset_kwargs,
                                           prevent_repeat=True)
            datasets = {
                "gogod": gogod_dataset,
                "kgs": kgs_dataset
            }
        else:
            datasets = problem.input_fn(mode, hp, dataset_kwargs=dataset_kwargs, prevent_repeat=True)

        return datasets

    def _get_model_spec(self, datasets, mode):
        model = self.model

        tf.logging.info("Creating the model...")

        if isinstance(datasets, dict):
            model_specs = {}
            for k, dataset in datasets.items():
                iterator = dataset.make_initializable_iterator()
                init_op = iterator.initializer

                features = iterator.get_next()
                model_spec = model.model_fn(features, mode)
                model_spec["iterator_init_op"] = init_op

                model_specs[k] = model_spec

            tf.logging.info("- done.")
            return model_specs
        else:
            iterator = datasets.make_initializable_iterator()
            init_op = iterator.initializer

            features = iterator.get_next()
            model_spec = model.model_fn(features, mode)
            model_spec["iterator_init_op"] = init_op

            tf.logging.info("- done.")
            return model_spec


class BestAccs:
    def __init__(self, multiple_datasets):
        self.multiple_datasets = multiple_datasets

        self.best_eval_p_acc1 = 0.0
        self.best_eval_v_loss1 = np.inf

        if multiple_datasets:
            self.best_eval_p_acc2 = 0.0
            self.best_eval_v_loss2 = np.inf

    def _multiple_cond(self, metrics):
        eval_p_acc1 = metrics['gogod_policy_accuracy']
        eval_v_loss1 = metrics['gogod_value_loss']
        cond1 = eval_p_acc1 >= self.best_eval_p_acc1 and eval_v_loss1 <= self.best_eval_v_loss1

        eval_p_acc2 = metrics['kgs_policy_accuracy']
        eval_v_loss2 = metrics['kgs_value_loss']
        cond2 = eval_p_acc2 >= self.best_eval_p_acc2 and eval_v_loss2 <= self.best_eval_v_loss2

        return cond1 and cond2

    def _single_cond(self, metrics):
        eval_p_acc1 = metrics['policy_accuracy']
        eval_v_loss1 = metrics['value_loss']

        return eval_p_acc1 >= self.best_eval_p_acc1 and eval_v_loss1 <= self.best_eval_v_loss1

    def get_condition(self, metrics):
        if self.multiple_datasets:
            return self._multiple_cond(metrics)
        else:
            return self._single_cond(metrics)

    def update_accs(self, metrics):
        self.best_eval_p_acc1 = metrics.get('policy_accuracy') or metrics.get('gogod_policy_accuracy')
        self.best_eval_v_loss1 = metrics.get('value_accuracy') or metrics.get('gogod_value_accuracy')

        if self.multiple_datasets:
            self.best_eval_p_acc2 = metrics['kgs_policy_accuracy']
            self.best_eval_v_loss2 = metrics['kgs_value_accuracy']
