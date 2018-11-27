import argparse
import random
import os

import tensorflow as tf
import numpy as np

import trainer
from data_generators import get_problem_class
from models import get_model_class
from hparams import get_hparams
from utils import utils

parser = argparse.ArgumentParser()

parser.add_argument('--problem', type=str, required=True,
                    help="Problem to use")
parser.add_argument('--model', type=str, required=True,
                    help="Model to use")
parser.add_argument('--hparams', type=str, required=True,
                    help="Hyper parameters to use")

parser.add_argument('--experiment_dir', type=str, required=True,
                    help="Directory to save summaries and log to")
parser.add_argument('--restore_dir',  type=str, default=None,
                    help="Optional, directory containing weights to reload before training")
parser.add_argument('--test_dir',  type=str, default='best_weights',
                    help="Optional, directory containing weights to use for the test dataset")

parser.add_argument('--overwrite_results', action='store_true', default=False,
                    help="Set this flag to overwrite previous results if they already exist "
                         "in the experiment dir")
parser.add_argument('--skip_generate_data', action='store_true', default=False,
                    help="Set this flag to not generate the tf record files if they already exist")


def set_random_seed(seed):
    tf.set_random_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def main(problem,
         model,
         hparams,
         experiment_dir,
         restore_dir,
         test_dir,
         overwrite_results,
         skip_generate_data):
    set_random_seed(230)
    tf.gfile.MakeDirs(experiment_dir)

    # Check that we are not overwriting some previous experiment
    if not overwrite_results:
        model_dir_has_best_weights = os.path.isdir(os.path.join(experiment_dir, "best_weights"))
        overwriting = model_dir_has_best_weights and restore_dir is None
        assert not overwriting, "Weights found in model_dir, aborting to avoid overwrite"

    utils.set_logger(os.path.join(experiment_dir, 'train.log'))

    # initialize the GoTrainer
    my_trainer = trainer.GoTrainer(problem, model, hparams, experiment_dir, skip_generate_data)
    # train and evaluate network on dev split
    my_trainer.train_and_evaluate(restore_from=restore_dir)

    utils.set_logger(os.path.join(experiment_dir, 'test.log'))
    # evaluate the network on test split
    my_trainer.test(test_dir)


if __name__ == '__main__':
    """Parse command line arguments and start main function."""
    args = parser.parse_args()

    model_name = args.model
    _model = get_model_class(model_name)

    problem_name = args.problem
    _problem = get_problem_class(problem_name)

    hp_name = args.hparams
    _hp = get_hparams(hp_name)

    _experiment_dir = args.experiment_dir
    _restore_dir = args.restore_dir
    _test_dir = args.test_dir

    _overwrite_results = args.overwrite_results
    _skip_generate_data = args.skip_generate_data

    main(_problem, _model, _hp, _experiment_dir, _restore_dir, _test_dir, _overwrite_results, _skip_generate_data)
