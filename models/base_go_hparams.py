import tensorflow as tf


def base_go_hparams():
    return tf.contrib.training.HParams(
        data_dir=".\\data",
        tmp_dir=".\\data\\tmp\\",

        # During training, we drop sequences whose inputs and targets are shorter
        # than min_length
        min_length=40,

        batch_size=2,

        num_filters=128,
        num_res_blocks=8,

        reg_strength=1e-4,
        value_loss_weight=0.01,

        sgd_momentum=0.9,

        lr_boundaries=[200000, 400000, 600000, 700000],
        lr_rates=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5],

        num_epochs=10,
        save_summary_steps=10,
    )
