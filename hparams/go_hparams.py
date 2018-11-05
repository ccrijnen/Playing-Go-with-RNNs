import tensorflow as tf


def base_go_hparams():
    return tf.contrib.training.HParams(
        data_dir="./data",
        tmp_dir="./data/tmp/",

        use_gogod_data=True,
        use_kgs_data=False,

        # If this is True and the _problem is recurrent it will split the game
        # sequence into two sequences, one for all black moves and one for all
        # white moves
        sort_sequence_by_color=False,

        # During training, we drop sequences whose inputs and targets are shorter
        # than min_length
        min_length=50,

        # When using a cnn _problem it will keep this amount of positions as history
        # resulting in a history_length*2+1 x board_size x board_size input
        history_length=0,

        truncated_length=10,

        # trainer settings
        num_epochs=1,
        batch_size=1,
        eval_every=150,
        save_summary_steps=15,

        # _model settings
        num_filters=256,
        num_res_blocks=8,

        reg_strength=1e-4,
        value_loss_weight=0.01,

        sgd_momentum=0.9,

        lr_boundaries=[275000, 550000, 825000, 962500],
        lr_rates=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
    )


def go_params_19():
    hp = base_go_hparams()

    hp_dict = {
        'use_gogod_data': True,
        'use_kgs_data': True,
    }

    hp.override_from_dict(hp_dict)

    return hp


def go_params_19_rnn():
    hp = go_params_19()

    return hp


def go_params_19_rnn_sorted():
    hp = go_params_19()

    hp_dict = {
        'sort_sequence_by_color': True,
        'batch_size': 1
    }

    hp.override_from_dict(hp_dict)

    return hp


def go_params_19_cnn():
    hp = go_params_19()

    hp_dict = {
        'use_gogod_data': True,
        'use_kgs_data': True,
        'history_length': 8,
        'batch_size': 32,
        'eval_every': 1000,
        'save_summary_steps': 100
    }

    hp.override_from_dict(hp_dict)

    return hp
