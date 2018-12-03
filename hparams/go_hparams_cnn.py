import tensorflow as tf


def base_go_hparams_cnn():
    """Base CNN hyperparameters for Go Models."""
    return tf.contrib.training.HParams(
        data_dir="./data",
        tmp_dir="./data/tmp/",

        use_gogod_data=True,
        use_kgs_data=False,

        # During training, we drop sequences whose inputs and targets are shorter
        # than min_length
        min_length=50,

        split_to_min_length=False,

        # Keep the last positions as history
        # resulting in a history_length*2+1 x board_size x board_size input
        history_length=8,

        # model settings
        num_filters=256,
        num_res_blocks=9,

        # only used in cnn models with a dense layer
        # num filters to reshape the dense output to
        num_dense_filter=2,

        # trainer settings
        num_epochs=1,
        batch_size=32,
        eval_every=1000,
        save_summary_steps=100,

        # optimizer settings
        reg_strength=1e-4,
        value_loss_weight=0.01,
        sgd_momentum=0.9,

        lr_boundaries=[275000, 550000, 825000, 962500],
        lr_rates=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
    )


def go_hparams_19_cnn():
    hp = base_go_hparams_cnn()

    hp_dict = {
        'use_gogod_data': True,
        'use_kgs_data': True,
    }

    hp.override_from_dict(hp_dict)

    return hp


def go_hparams_19_cnn_split():
    hp = base_go_hparams_cnn()

    hp_dict = {
        'use_gogod_data': True,
        'use_kgs_data': True,
        'split_to_min_length': True,

        # min_length = 150:
        # 'lr_boundaries': [160000, 320000, 480000, 560000],

        # min_length = 50:
        'lr_boundaries': [67500, 135000, 202500, 236250],
    }

    hp.override_from_dict(hp_dict)

    return hp
