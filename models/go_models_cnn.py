import tensorflow as tf

from models.base_go_model import GoModel


class GoModelCNN(GoModel):
    def bottom(self, features):
        return features

    def top(self, body_output, features):
        hp = self._hparams

        board_size = hp.board_size
        num_moves = hp.num_moves
        is_training = hp.mode == tf.estimator.ModeKeys.TRAIN

        legal_moves = features["legal_moves"]

        # Policy Head
        with tf.variable_scope('policy_head'):
            p_conv = self.my_conv2d(body_output, filters=2, kernel_size=1)
            p_conv = self.my_batchnorm(p_conv, center=False, scale=False, training=is_training)
            p_conv = tf.nn.relu(p_conv)

            p_logits = tf.reshape(p_conv, [-1, 2 * board_size * board_size])
            p_logits = tf.layers.dense(p_logits, num_moves)
            p_logits = tf.multiply(p_logits, legal_moves, name='policy_logits')

        # Value Head
        with tf.variable_scope('value_head'):
            v_conv = self.my_conv2d(body_output, filters=1, kernel_size=1)
            v_conv = self.my_batchnorm(v_conv, center=False, scale=False, training=is_training)
            v_conv = tf.nn.relu(v_conv)

            v_fc = tf.reshape(v_conv, [-1, board_size * board_size])
            v_fc = tf.layers.dense(v_fc, 256)
            v_fc = tf.nn.relu(v_fc)

            v_output = tf.layers.dense(v_fc, 1)
            v_output = tf.reshape(v_output, [-1])
            v_output = tf.nn.tanh(v_output, name='value_output')

        return p_logits, v_output

    def loss(self, logits, features):
        p_logits, v_output = logits

        with tf.variable_scope('policy_loss'):
            p_targets = features["p_targets"]
            p_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=p_logits,
                                                                      labels=tf.stop_gradient(p_targets))
            p_loss = tf.reduce_mean(p_losses)

        with tf.variable_scope('value_loss'):
            v_targets = features['v_targets']
            v_losses = tf.square(v_targets - v_output)
            v_loss = tf.reduce_mean(v_losses)

        with tf.variable_scope('l2_loss'):
            reg_vars = [v for v in tf.trainable_variables()
                        if 'bias' not in v.name and 'beta' not in v.name]
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in reg_vars])

        return [p_loss, v_loss, l2_loss], [p_losses, v_losses]

    def policy_accuracy(self, features, predictions, mask=None):
        with tf.variable_scope('policy_accuracy'):
            p_targets = features["p_targets"]
            p_correct = tf.equal(p_targets, predictions)

            if mask is not None:
                p_correct = tf.boolean_mask(p_correct, mask)

            p_acc = tf.reduce_mean(tf.cast(p_correct, tf.float32))
            return p_acc


class AlphaZeroModel(GoModelCNN):
    def body(self, features):
        hp = self.hparams
        inputs = features["inputs"]

        with tf.variable_scope("conv_block"):
            out = self.conv_block(inputs)

        for i in range(hp.num_res_blocks + 1):
            with tf.variable_scope("residual_block_{}".format(i + 1)):
                out = self.residual_block(out)

        return out
