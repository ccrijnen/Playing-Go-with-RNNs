import tensorflow as tf

from models.base_go_model import GoModelRNN


class ConvLSTMModel(GoModelRNN):
    def rnn_cell(self, board_size):
        cell = tf.contrib.rnn.Conv2DLSTMCell(input_shape=[board_size, board_size, self.hparams.num_filters],
                                             kernel_shape=[3, 3],
                                             output_channels=2)
        return cell

    def body(self, features):
        hp = self.hparams
        board_size = hp.board_size

        inputs = features["inputs"]
        inputs = tf.reshape(inputs, [-1, 3, board_size, board_size])

        with tf.variable_scope("conv_block"):
            out = self.conv_block(inputs)

        for i in range(hp.num_res_blocks):
            with tf.variable_scope("residual_block_{}".format(i+1)):
                out = self.residual_block(out)

        with tf.variable_scope("lstm"):
            cell = self.rnn_cell(board_size)
            rnn_in = tf.reshape(out, [-1, self.max_game_length, hp.num_filters, board_size, board_size])
            rnn_in = tf.transpose(rnn_in, [0, 1, 3, 4, 2])

            rnn_outputs, final_state = tf.nn.dynamic_rnn(cell,
                                                         rnn_in,
                                                         time_major=False,
                                                         dtype=tf.float32)
            rnn_outputs = tf.transpose(rnn_outputs, [0, 1, 4, 2, 3])
        return rnn_outputs
