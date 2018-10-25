import argparse
import random
import os

import tensorflow as tf
import numpy as np

import trainer
from models import go_hparams
from data_generators import go_problem_19
from models import go_models_rnn, go_models_cnn
from utils import utils

parser = argparse.ArgumentParser()
parser.add_argument('--experiment_dir', type=str, required=True,
                    help="Directory to save summaries and log to")
parser.add_argument('--restore_dir',  type=str, default='best_weights',
                    help="Optional, directory containing weights to reload before training")


def set_random_seed(seed):
    tf.set_random_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def main():
    args = parser.parse_args()

    set_random_seed(230)
    tf.gfile.MakeDirs(args.experiment_dir)

    # Check that we are not overwriting some previous experiment
    # Comment these lines if you are developing your model and don't care about overwritting
    model_dir_has_best_weights = os.path.isdir(os.path.join(args.experiment_dir, "best_weights"))
    overwriting = model_dir_has_best_weights and args.restore_dir is None
    assert not overwriting, "Weights found in model_dir, aborting to avoid overwrite"

    hp = go_hparams.go_params_19_cnn()
    hp.add_hparam("experiment_dir", args.experiment_dir)

    utils.set_logger(os.path.join(args.experiment_dir, 'train.log'))

    # prob = go_problem_19.GoProblem19Rnn()
    prob = go_problem_19.GoProblem19Cnn()
    hp = prob.get_hparams(hp)

    tf.logging.info("Creating the datasets...")
    prob.generate_data(hp.data_dir, hp.tmp_dir)
    tf.logging.info("- done")

    # model = go_models_rnn.ConvLSTMModel(hp)
    model = go_models_cnn.AlphaZeroModel(hp)

    my_trainer = trainer.GoTrainer(prob, model, hp)
    my_trainer.train_and_evaluate()

    my_trainer.test(args.restore_dir)


if __name__ == '__main__':
    main()
