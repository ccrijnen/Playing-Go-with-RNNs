import tensorflow as tf

from models.base_go_model import GoModel
from models import rnn_cells


class GoModelRNN(GoModel):
    def bottom(self, features):
        self.max_game_length = tf.reduce_max(features["game_length"])
        return features

    def top(self, body_output, features):
        hp = self._hparams

        board_size = hp.board_size
        num_moves = hp.num_moves
        is_training = hp.mode == tf.estimator.ModeKeys.TRAIN

        legal_moves = features["legal_moves"]

        body_output = tf.reshape(body_output, [-1, hp.num_dense_filter, board_size, board_size])

        # Policy Head
        with tf.variable_scope('policy_head'):
            p_conv = self.my_conv2d(body_output, filters=2, kernel_size=1)
            p_conv = self.my_batchnorm(p_conv, center=False, scale=False, training=is_training)
            p_conv = tf.nn.relu(p_conv)

            p_logits = tf.reshape(p_conv, [-1, self.max_game_length, 2 * board_size * board_size])
            p_logits = tf.layers.dense(p_logits, num_moves)
            p_logits = tf.multiply(p_logits, legal_moves, name='policy_logits')

        # Value Head
        with tf.variable_scope('value_head'):
            v_conv = self.my_conv2d(body_output, filters=1, kernel_size=1)
            v_conv = self.my_batchnorm(v_conv, center=False, scale=False, training=is_training)
            v_conv = tf.nn.relu(v_conv)

            v_fc = tf.reshape(v_conv, [-1, self.max_game_length, board_size * board_size])
            v_fc = tf.layers.dense(v_fc, 256)
            v_fc = tf.nn.relu(v_fc)

            v_output = tf.layers.dense(v_fc, 1)
            v_output = tf.reshape(v_output, [-1, self.max_game_length])
            v_output = tf.nn.tanh(v_output, name='value_output')

        return p_logits, v_output

    def loss(self, logits, features):
        game_lengths = features["game_length"]
        mask = tf.sequence_mask(game_lengths)

        p_logits, v_output = logits

        with tf.variable_scope('policy_loss'):
            p_targets = features["p_targets"]
            p_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=p_logits,
                                                                      labels=tf.stop_gradient(p_targets))
            p_losses_masked = tf.boolean_mask(p_losses, mask)
            p_loss = tf.reduce_mean(p_losses_masked)

        with tf.variable_scope('value_loss'):
            v_targets = features['v_targets']
            v_losses = tf.square(v_targets - v_output)
            v_losses_masked = tf.boolean_mask(v_losses, mask)
            v_loss = tf.reduce_mean(v_losses_masked)

        with tf.variable_scope('l2_loss'):
            reg_vars = [v for v in tf.trainable_variables()
                        if 'bias' not in v.name and 'beta' not in v.name]
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in reg_vars])

        return [p_loss, v_loss, l2_loss], [p_losses, v_losses]

    def policy_accuracy(self, features, predictions, mask=None):
        with tf.variable_scope('policy_accuracy'):
            p_targets = features["p_targets"]
            game_lengths = features["game_length"]

            p_correct = tf.equal(p_targets, predictions)

            if mask is None:
                mask = tf.sequence_mask(game_lengths)

            p_correct = tf.boolean_mask(p_correct, mask)

            p_acc = tf.reduce_mean(tf.cast(p_correct, tf.float32))
            return p_acc


class VanillaRNNModel(GoModelRNN):
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

        with tf.variable_scope("VanillaRNN"):
            rnn_in = tf.reshape(out, [-1, self.max_game_length, hp.num_filters * board_size * board_size])
            rnn_in = tf.transpose(rnn_in, [1, 0, 2])

            num_units = hp.num_dense_filter * board_size * board_size
            rnn = tf.contrib.cudnn_rnn.CudnnRNNTanh(num_layers=1, num_units=num_units)
            rnn_outputs, _ = rnn(rnn_in)

            rnn_outputs = tf.transpose(rnn_outputs, [1, 0, 2])
            rnn_outputs = tf.reshape(rnn_outputs,
                                     [-1, self.max_game_length, hp.num_dense_filter, board_size, board_size])

        return rnn_outputs


class LSTMModel(GoModelRNN):
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
            rnn_in = tf.reshape(out, [-1, self.max_game_length, hp.num_filters * board_size * board_size])
            rnn_in = tf.transpose(rnn_in, [1, 0, 2])

            num_units = hp.num_dense_filter * board_size * board_size
            lstm = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers=1, num_units=num_units)
            rnn_outputs, _ = lstm(rnn_in)

            rnn_outputs = tf.transpose(rnn_outputs, [1, 0, 2])
            rnn_outputs = tf.reshape(rnn_outputs,
                                     [-1, self.max_game_length, hp.num_dense_filter, board_size, board_size])

        return rnn_outputs


class GRUModel(GoModelRNN):
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

        with tf.variable_scope("gru"):
            rnn_in = tf.reshape(out, [-1, self.max_game_length, hp.num_filters * board_size * board_size])
            rnn_in = tf.transpose(rnn_in, [1, 0, 2])

            num_units = hp.num_dense_filter * board_size * board_size
            gru = tf.contrib.cudnn_rnn.CudnnGRU(num_layers=1, num_units=num_units)
            rnn_outputs, _ = gru(rnn_in)

            rnn_outputs = tf.transpose(rnn_outputs, [1, 0, 2])
            rnn_outputs = tf.reshape(rnn_outputs,
                                     [-1, self.max_game_length, hp.num_dense_filter, board_size, board_size])

        return rnn_outputs


class GoModelConvRNN(GoModel):
    def bottom(self, features):
        self.max_game_length = tf.reduce_max(features["game_length"])
        return features

    def top(self, body_output, features):
        hp = self._hparams

        board_size = hp.board_size
        num_moves = hp.num_moves
        is_training = hp.mode == tf.estimator.ModeKeys.TRAIN

        legal_moves = features["legal_moves"]

        body_output = tf.reshape(body_output, [-1, hp.num_filters, board_size, board_size])

        # Policy Head
        with tf.variable_scope('policy_head'):
            p_conv = self.my_conv2d(body_output, filters=2, kernel_size=1)
            p_conv = self.my_batchnorm(p_conv, center=False, scale=False, training=is_training)
            p_conv = tf.nn.relu(p_conv)

            p_logits = tf.reshape(p_conv, [-1, self.max_game_length, 2 * board_size * board_size])
            p_logits = tf.layers.dense(p_logits, num_moves)
            p_logits = tf.multiply(p_logits, legal_moves, name='policy_logits')

        # Value Head
        with tf.variable_scope('value_head'):
            v_conv = self.my_conv2d(body_output, filters=1, kernel_size=1)
            v_conv = self.my_batchnorm(v_conv, center=False, scale=False, training=is_training)
            v_conv = tf.nn.relu(v_conv)

            v_fc = tf.reshape(v_conv, [-1, self.max_game_length, board_size * board_size])
            v_fc = tf.layers.dense(v_fc, 256)
            v_fc = tf.nn.relu(v_fc)

            v_output = tf.layers.dense(v_fc, 1)
            v_output = tf.reshape(v_output, [-1, self.max_game_length])
            v_output = tf.nn.tanh(v_output, name='value_output')

        return p_logits, v_output

    def loss(self, logits, features):
        game_lengths = features["game_length"]
        mask = tf.sequence_mask(game_lengths)

        p_logits, v_output = logits

        with tf.variable_scope('policy_loss'):
            p_targets = features["p_targets"]
            p_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=p_logits,
                                                                      labels=tf.stop_gradient(p_targets))
            p_losses_masked = tf.boolean_mask(p_losses, mask)
            p_loss = tf.reduce_mean(p_losses_masked)

        with tf.variable_scope('value_loss'):
            v_targets = features['v_targets']
            v_losses = tf.square(v_targets - v_output)
            v_losses_masked = tf.boolean_mask(v_losses, mask)
            v_loss = tf.reduce_mean(v_losses_masked)

        with tf.variable_scope('l2_loss'):
            reg_vars = [v for v in tf.trainable_variables()
                        if 'bias' not in v.name and 'beta' not in v.name]
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in reg_vars])

        return [p_loss, v_loss, l2_loss], [p_losses, v_losses]

    def policy_accuracy(self, features, predictions, mask=None):
        with tf.variable_scope('policy_accuracy'):
            p_targets = features["p_targets"]
            game_lengths = features["game_length"]

            p_correct = tf.equal(p_targets, predictions)

            if mask is None:
                mask = tf.sequence_mask(game_lengths)

            p_correct = tf.boolean_mask(p_correct, mask)

            p_acc = tf.reduce_mean(tf.cast(p_correct, tf.float32))
            return p_acc


class ConvRNNModel(GoModelConvRNN):
    def body(self, features):
        hp = self.hparams
        board_size = hp.board_size

        game_length = features["game_length"]
        inputs = features["inputs"]
        inputs = tf.reshape(inputs, [-1, 3, board_size, board_size])

        with tf.variable_scope("conv_block"):
            out = self.conv_block(inputs)

        for i in range(hp.num_res_blocks):
            with tf.variable_scope("residual_block_{}".format(i+1)):
                out = self.residual_block(out)

        with tf.variable_scope("conv_rnn"):
            rnn_in = tf.reshape(out, [-1, self.max_game_length, hp.num_filters, board_size, board_size])

            cell = rnn_cells.ConvRNNCell(input_shape=[hp.num_filters, board_size, board_size],
                                         output_channels=hp.num_filters,
                                         kernel_shape=[3, 3],
                                         activation=tf.nn.relu)

            rnn_outputs, _ = tf.nn.dynamic_rnn(cell, rnn_in, sequence_length=game_length,
                                               time_major=False, dtype=tf.float32)
        return rnn_outputs


class ConvLSTMModel(GoModelConvRNN):
    def body(self, features):
        hp = self.hparams
        board_size = hp.board_size

        game_length = features["game_length"]
        inputs = features["inputs"]
        inputs = tf.reshape(inputs, [-1, 3, board_size, board_size])

        with tf.variable_scope("conv_block"):
            out = self.conv_block(inputs)

        for i in range(hp.num_res_blocks):
            with tf.variable_scope("residual_block_{}".format(i + 1)):
                out = self.residual_block(out)

        with tf.variable_scope("conv_lstm"):
            rnn_in = tf.reshape(out, [-1, self.max_game_length, hp.num_filters, board_size, board_size])
            rnn_in = tf.transpose(rnn_in, perm=[0, 1, 3, 4, 2])

            cell = tf.contrib.rnn.Conv2DLSTMCell(input_shape=[board_size, board_size, hp.num_filters],
                                                 kernel_shape=[3, 3],
                                                 output_channels=hp.num_filters,
                                                 use_bias=False,
                                                 skip_connection=False)

            rnn_outputs, _ = tf.nn.dynamic_rnn(cell, rnn_in, sequence_length=game_length,
                                               time_major=False, dtype=tf.float32)
            rnn_outputs = tf.transpose(rnn_outputs, perm=[0, 1, 4, 2, 3])

        return rnn_outputs


class ConvGRUModel(GoModelConvRNN):
    def body(self, features):
        hp = self.hparams
        board_size = hp.board_size

        game_length = features["game_length"]
        inputs = features["inputs"]
        inputs = tf.reshape(inputs, [-1, 3, board_size, board_size])

        with tf.variable_scope("conv_block"):
            out = self.conv_block(inputs)

        for i in range(hp.num_res_blocks):
            with tf.variable_scope("residual_block_{}".format(i+1)):
                out = self.residual_block(out)

        with tf.variable_scope("conv_gru"):
            rnn_in = tf.reshape(out, [-1, self.max_game_length, hp.num_filters, board_size, board_size])

            cell = rnn_cells.ConvGRUCell(input_shape=[board_size, board_size],
                                         kernel_shape=[3, 3],
                                         output_channels=hp.num_filters,
                                         normalize=True,
                                         data_format='channels_first')

            rnn_outputs, _ = tf.nn.dynamic_rnn(cell, rnn_in, sequence_length=game_length,
                                               time_major=False, dtype=tf.float32)
        return rnn_outputs


def static_rnn(cell, inputs, init_state, min_length, name):
    inputs = inputs[:, :min_length]
    inputs = tf.unstack(inputs, min_length, axis=1)

    rnn_outputs = []
    with tf.variable_scope(name) as scope:
        for i, rnn_in in enumerate(inputs):
            if i > 0:
                scope.reuse_variables()

            rnn_output, init_state = cell(rnn_in, init_state)
            rnn_outputs.append(rnn_output)

    rnn_outputs = tf.stack(rnn_outputs, axis=1)
    return rnn_outputs


class MyConvRNNModel(GoModelRNN):
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

        rnn_ins = tf.reshape(out, [-1, self.max_game_length, hp.num_filters, board_size, board_size])

        cell = rnn_cells.ConvRNNCell(input_shape=[hp.num_filters, board_size, board_size],
                                     output_channels=hp.num_dense_filter,
                                     kernel_shape=[3, 3],
                                     activation=tf.nn.relu)

        init_state = cell.zero_state(hp.batch_size, tf.float32)

        rnn_outputs = static_rnn(cell, rnn_ins, init_state, hp.min_length, "conv_rnn")

        self.max_game_length = hp.min_length
        features["game_length"] = tf.constant([hp.min_length] * hp.batch_size, tf.int64)
        features["p_targets"] = features["p_targets"][:, :hp.min_length]
        features["v_targets"] = features["v_targets"][:, :hp.min_length]
        features["legal_moves"] = features["legal_moves"][:, :hp.min_length]
        return rnn_outputs


class MyConvLSTMModel(GoModelRNN):
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

        rnn_in = tf.reshape(out, [-1, self.max_game_length, hp.num_filters, board_size, board_size])
        rnn_in = tf.transpose(rnn_in, perm=[0, 1, 3, 4, 2])

        cell = tf.contrib.rnn.Conv2DLSTMCell(input_shape=[board_size, board_size, hp.num_filters],
                                             kernel_shape=[3, 3],
                                             output_channels=hp.num_dense_filter,
                                             use_bias=False,
                                             skip_connection=False)

        init_state = cell.zero_state(hp.batch_size, tf.float32)

        rnn_outputs = static_rnn(cell, rnn_in, init_state, hp.min_length, "my_conv_lstm")
        rnn_outputs = tf.transpose(rnn_outputs, perm=[0, 1, 4, 2, 3])

        self.max_game_length = hp.min_length
        features["game_length"] = tf.constant([hp.min_length] * hp.batch_size, tf.int64)
        features["p_targets"] = features["p_targets"][:, :hp.min_length]
        features["v_targets"] = features["v_targets"][:, :hp.min_length]
        features["legal_moves"] = features["legal_moves"][:, :hp.min_length]
        return rnn_outputs


class MyConvGRUModel(GoModelRNN):
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

        rnn_ins = tf.reshape(out, [-1, self.max_game_length, hp.num_filters, board_size, board_size])

        cell = rnn_cells.ConvGRUCell(input_shape=[board_size, board_size],
                                     kernel_shape=[3, 3],
                                     output_channels=hp.num_dense_filter,
                                     normalize=True,
                                     data_format='channels_first')

        init_state = cell.zero_state(hp.batch_size, tf.float32)

        rnn_outputs = static_rnn(cell, rnn_ins, init_state, hp.min_length, "conv_gru")

        self.max_game_length = hp.min_length
        features["game_length"] = tf.constant([hp.min_length] * hp.batch_size, tf.int64)
        features["p_targets"] = features["p_targets"][:, :hp.min_length]
        features["v_targets"] = features["v_targets"][:, :hp.min_length]
        features["legal_moves"] = features["legal_moves"][:, :hp.min_length]
        return rnn_outputs
