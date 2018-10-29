import argparse
import random
import os

import tensorflow as tf
import numpy as np

import trainer
from data_generators import go_problem_19_small
from models import go_models_rnn, go_models_cnn, go_hparams
from utils import utils


parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str,
                    help="Mode of the model to run; must be either cnn or rnn!", required=True)


def set_random_seed(seed):
    tf.set_random_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def conv_lstm_trainer(hp):
    prob = go_problem_19_small.GoProblem19SmallRnn()
    hp = prob.get_hparams(hp)

    model = go_models_rnn.ConvLSTMModel(hp)

    my_trainer = trainer.GoTrainer(prob, model, hp)
    return my_trainer


def my_conv_lstm_trainer(hp):
    prob = go_problem_19_small.GoProblem19SmallRnn()
    hp = prob.get_hparams(hp)

    model = go_models_rnn.MyConvLSTMModel(hp)

    my_trainer = trainer.GoTrainer(prob, model, hp)
    return my_trainer


def lstm_trainer(hp):
    prob = go_problem_19_small.GoProblem19SmallRnn()
    hp = prob.get_hparams(hp)

    model = go_models_rnn.LSTMModel(hp)

    my_trainer = trainer.GoTrainer(prob, model, hp)
    return my_trainer


def cnn_trainer(hp):
    prob = go_problem_19_small.GoProblem19SmallCnn()
    hp = prob.get_hparams(hp)

    model = go_models_cnn.AlphaZeroModel(hp)

    my_trainer = trainer.GoTrainer(prob, model, hp)
    return my_trainer


def main():
    args = parser.parse_args()
    mode = args.mode
    assert mode in ["cnn", "conv_lstm", "my_conv_lstm", "lstm"]

    experiment_dir = "./experiments/small_problem/"

    set_random_seed(230)

    if mode == "cnn":
        hp = go_hparams.go_params_19_cnn()
        path = os.path.join(experiment_dir, "cnn")
        tf.gfile.MakeDirs(path)
        hp.experiment_dir = path

        my_trainer = cnn_trainer(hp)
    elif mode == "conv_lstm":
        hp = go_hparams.go_params_19_rnn_sorted()
        path = os.path.join(experiment_dir, "conv_lstm")
        tf.gfile.MakeDirs(path)
        hp.experiment_dir = path

        my_trainer = conv_lstm_trainer(hp)
    elif mode == "my_conv_lstm":
        hp = go_hparams.go_params_19_rnn_sorted()
        path = os.path.join(experiment_dir, "my_conv_lstm")
        tf.gfile.MakeDirs(path)
        hp.experiment_dir = path

        my_trainer = my_conv_lstm_trainer(hp)
    elif mode == "lstm":
        hp = go_hparams.go_params_19_rnn_sorted()
        path = os.path.join(experiment_dir, "lstm")
        tf.gfile.MakeDirs(path)
        hp.experiment_dir = path

        my_trainer = lstm_trainer(hp)

    utils.set_logger(os.path.join(path, 'train.log'))
    my_trainer.train_and_evaluate()

    utils.set_logger(os.path.join(path, 'test.log'))
    my_trainer.test('best_weights')


if __name__ == '__main__':
    main()
