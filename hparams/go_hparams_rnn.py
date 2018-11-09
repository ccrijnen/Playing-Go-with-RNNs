import tensorflow as tf


def base_go_hparams_rnn():
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

        # model settings
        num_filters=256,
        num_res_blocks=8,

        # num filters to reshape the dense output to
        num_dense_filter=2,

        # trainer settings
        num_epochs=1,
        batch_size=1,
        eval_every=150,
        save_summary_steps=15,

        # optimizer settings
        reg_strength=1e-4,
        value_loss_weight=0.01,
        sgd_momentum=0.9,

        lr_boundaries=[43750, 87500, 131250, 153125],
        lr_rates=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
    )


def go_hparams_19_rnn():
    hp = base_go_hparams_rnn()

    hp_dict = {
        'use_gogod_data': True,
        'use_kgs_data': True,
    }

    hp.override_from_dict(hp_dict)

    return hp


def go_hparams_19_rnn_sorted():
    hp = go_hparams_19_rnn()

    hp_dict = {
        'sort_sequence_by_color': True,
    }

    hp.override_from_dict(hp_dict)

    return hp
