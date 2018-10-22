import argparse
import random
import os

import tensorflow as tf
import numpy as np

import trainer
from data_generators import go_problem_19_toy
from models import go_models_rnn, go_models_cnn, go_hparams
from utils import utils


parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str,
                    help="Mode of the model to run; must be either cnn or rnn!", required=True)


def set_random_seed(seed):
    tf.set_random_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def run_rnn(hp):
    prob = go_problem_19_toy.GoProblem19ToyRnn()
    hp = prob.get_hparams(hp)

    model = go_models_rnn.ConvLSTMModel(hp)

    my_trainer = trainer.GoTrainer(prob, model, hp)
    my_trainer.train_and_evaluate()

    my_trainer.test('best_weights')


def run_cnn(hp):
    prob = go_problem_19_toy.GoProblem19ToyCnn()
    hp = prob.get_hparams(hp)

    model = go_models_cnn.AlphaZeroModel(hp)

    my_trainer = trainer.GoTrainer(prob, model, hp)
    my_trainer.train_and_evaluate()

    my_trainer.test('best_weights')


def main():
    args = parser.parse_args()
    mode = args.mode
    assert mode in ["rnn", "cnn"]

    experiment_dir = "./experiments/toy_problem/"

    set_random_seed(230)

    if mode == "rnn":
        hp = go_hparams.go_params_19_rnn_sorted()
        path = os.path.join(experiment_dir, "rnn")
        tf.gfile.MakeDirs(path)
        hp.experiment_dir = path

        utils.set_logger(os.path.join(path, 'train.log'))

        run_rnn(hp)
    elif mode == "cnn":
        hp = go_hparams.go_params_19_cnn()
        path = os.path.join(experiment_dir, "cnn")
        tf.gfile.MakeDirs(path)
        hp.experiment_dir = path

        utils.set_logger(os.path.join(path, 'train.log'))

        run_cnn(hp)


if __name__ == '__main__':
    main()
