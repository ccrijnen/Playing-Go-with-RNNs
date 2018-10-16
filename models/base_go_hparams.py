import tensorflow as tf


def base_go_hparams():
    return tf.contrib.training.HParams(
        data_dir=".\\data",
        tmp_dir=".\\data\\tmp\\",

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
        save_summary_steps=250,
    )
