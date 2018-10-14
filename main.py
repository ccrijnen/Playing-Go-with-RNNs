import argparse
import random
import os

import tensorflow as tf
import numpy as np

import trainer
from models.base_go_hparams import base_go_hparams
from data_generators import go_problem_19
from models import lstm_model
from utils import utils

parser = argparse.ArgumentParser()
parser.add_argument('--experiment_dir',
                    help="Directory containing params.json", required=True)
parser.add_argument('--restore_dir', default=None,
                    help="Optional, directory containing weights to reload before training")


def set_random_seed(seed):
    tf.set_random_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def main(args):
    set_random_seed(230)
    tf.gfile.MakeDirs(args.experiment_dir)

    # Check that we are not overwriting some previous experiment
    # Comment these lines if you are developing your model and don't care about overwritting
    # model_dir_has_best_weights = os.path.isdir(os.path.join(args.experiment_dir, "best_weights"))
    # overwritting = model_dir_has_best_weights and args.restore_dir is None
    # assert not overwritting, "Weights found in model_dir, aborting to avoid overwrite"

    hp = base_go_hparams()
    hp.add_hparam("experiment_dir", args.experiment_dir)

    utils.set_logger(os.path.join(args.experiment_dir, 'train.log'))

    prob = go_problem_19.GoProblem19x19()

    # tf.logging.info("Creating the datasets...")
    # prob.generate_data(hp.data_dir, hp.tmp_dir)
    # tf.logging.info("- done")

    hp = prob.get_hparams(hp)

    model = lstm_model.ConvLSTMModel(hp)
    my_trainer = trainer.GoTrainer(prob, model, hp)

    my_trainer.train_and_evaluate()


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
