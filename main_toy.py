import argparse
import random
import os

import tensorflow as tf
import numpy as np

import trainer
from data_generators import go_problem_19_toy
from models import go_models_rnn, go_models_cnn
from utils import utils


parser = argparse.ArgumentParser()
parser.add_argument('--mode',
                    help="Mode of the model to run; must be either cnn or rnn!", required=True)


HPARAMS_RNN = tf.contrib.training.HParams(
    data_dir="./data/",
    tmp_dir="./data/tmp/",
    experiment_dir="./experiments/toy_problem/",

    use_gogod_data=True,
    use_kgs_data=True,

    # If this is True and the problem is recurrent it will split the game
    # sequence into two sequences, one for all black moves and one for all
    # white moves
    sort_sequence_by_color=True,

    # During training, we drop sequences whose inputs and targets are shorter
    # than min_length
    min_length=50,

    batch_size=2,

    num_filters=256,
    num_res_blocks=8,

    reg_strength=1e-4,
    value_loss_weight=0.01,

    sgd_momentum=0.9,

    lr_boundaries=[200000, 400000, 600000, 700000],
    lr_rates=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5],

    num_epochs=1,
    eval_every=1000,
    save_summary_steps=1000,
)

HPARAMS_CNN = tf.contrib.training.HParams(
    data_dir="./data/",
    tmp_dir="./data/tmp/",
    experiment_dir="./experiments/toy_problem/",

    use_gogod_data=True,
    use_kgs_data=True,

    # If this is True and the problem is recurrent it will split the game
    # sequence into two sequences, one for all black moves and one for all
    # white moves
    sort_sequence_by_color=False,

    # During training, we drop sequences whose inputs and targets are shorter
    # than min_length
    min_length=50,

    history_length=8,

    batch_size=8,

    num_filters=256,
    num_res_blocks=8,

    reg_strength=1e-4,
    value_loss_weight=0.01,

    sgd_momentum=0.9,

    lr_boundaries=[200000, 400000, 600000, 700000],
    lr_rates=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5],

    num_epochs=1,
    eval_every=1000,
    save_summary_steps=1000,
)


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


def run_cnn(hp):
    prob = go_problem_19_toy.GoProblem19ToyCnn()
    hp = prob.get_hparams(hp)

    model = go_models_cnn.AlphaZeroModel(hp)

    my_trainer = trainer.GoTrainer(prob, model, hp)
    my_trainer.train_and_evaluate()


def main():
    args = parser.parse_args()
    mode = args.mode
    assert mode in ["rnn", "cnn"]

    set_random_seed(230)

    if mode == "rnn":
        hp = HPARAMS_RNN
        path = os.path.join(hp.experiment_dir, "rnn")
        tf.gfile.MakeDirs(path)
        hp.experiment_dir = path

        utils.set_logger(os.path.join(hp.experiment_dir, 'train.log'))

        run_rnn(hp)
    elif mode == "cnn":
        hp = HPARAMS_CNN
        path = os.path.join(hp.experiment_dir, "cnn")
        tf.gfile.MakeDirs(path)
        hp.experiment_dir = path

        utils.set_logger(os.path.join(hp.experiment_dir, 'train.log'))

        run_cnn(hp)


if __name__ == '__main__':
    main()
