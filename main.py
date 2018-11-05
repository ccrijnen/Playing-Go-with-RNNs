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
parser.add_argument('--restore_dir',  type=str, default='best_weights',
                    help="Optional, directory containing weights to reload before training")

parser.add_argument('--overwrite_results', action='store_true', default=False,
                    help="TODO")
parser.add_argument('--skip_generate_data', action='store_true', default=False,
                    help="TODO")


def set_random_seed(seed):
    tf.set_random_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def main(problem,
         model,
         hparams,
         experiment_dir,
         restore_dir,
         overwrite_results,
         skip_generate_data):
    set_random_seed(230)
    tf.gfile.MakeDirs(experiment_dir)

    if not overwrite_results:
        # Check that we are not overwriting some previous experiment
        # Comment these lines if you are developing your _model and don't care about overwritting
        model_dir_has_best_weights = os.path.isdir(os.path.join(experiment_dir, "best_weights"))
        overwriting = model_dir_has_best_weights and restore_dir is None
        assert not overwriting, "Weights found in model_dir, aborting to avoid overwrite"

    hp = hparams()
    hp.add_hparam("experiment_dir", experiment_dir)

    utils.set_logger(os.path.join(experiment_dir, 'train.log'))

    problem = problem()
    hp = problem.get_hparams(hp)

    if not skip_generate_data:
        tf.logging.info("Creating the datasets...")
        problem.generate_data(hp.data_dir, hp.tmp_dir)
        tf.logging.info("- done")

    model = model(hp)

    my_trainer = trainer.GoTrainer(problem, model, hp)
    my_trainer.train_and_evaluate()

    utils.set_logger(os.path.join(experiment_dir, 'test.log'))
    my_trainer.test(restore_dir)


if __name__ == '__main__':
    args = parser.parse_args()

    model_name = args.model
    _model = get_model_class(model_name)

    problem_name = args.problem
    _problem = get_problem_class(problem_name)

    hp_name = args.hparams
    _hp = get_hparams(hp_name)

    _experiment_dir = args.experiment_dir
    _restore_dir = args.restore_dir

    _overwrite_results = args.overwrite_results
    _skip_generate_data = args.skip_generate_data

    main(_problem, _model, _hp, _experiment_dir, _restore_dir, _overwrite_results, _skip_generate_data)
