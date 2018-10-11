import tensorflow as tf

import copy
import functools


class GoModel(object):
    def __init__(self,
                 hparams,
                 mode=tf.estimator.ModeKeys.TRAIN):
        hparams = copy.copy(hparams)
        self._original_hparams = hparams
        self.set_mode(mode)

        self.my_conv2d = functools.partial(
            tf.layers.conv2d,
            padding="SAME",
            data_format="channels_first",
            use_bias=False)

        self.my_batchnorm = functools.partial(
            tf.layers.batch_normalization,
            axis=1,
            momentum=.95,
            epsilon=1e-5,
            center=True,
            scale=True,
            fused=True)

    @property
    def hparams(self):
        return self._hparams

    @property
    def is_training(self):
        return self._hparams.mode == tf.estimator.ModeKeys.TRAIN

    @property
    def is_predicting(self):
        return self._hparams.mode == tf.estimator.ModeKeys.PREDICT

    def set_mode(self, mode):
        """Set hparams with the given mode."""
        log_info("Setting GoModel mode to '%s'", mode)
        hparams = copy.copy(self._original_hparams)
        hparams.add_hparam("mode", mode)
        # When not in training mode, set all forms of dropout to zero.
        if mode != tf.estimator.ModeKeys.TRAIN:
            for key in hparams.values():
                if key.endswith("dropout") or key == "label_smoothing":
                    log_info("Setting hparams.%s to 0.0", key)
                    setattr(hparams, key, 0.0)
        self._hparams = hparams

    def conv_block(self, inputs):
        hp = self._hparams

        is_training = hp.mode == tf.estimator.ModeKeys.TRAIN
        filters = hp.num_filters

        conv_output = self.my_conv2d(inputs, filters=filters, kernel_size=3)
        conv_output = self.my_batchnorm(conv_output, training=is_training)
        conv_output = tf.nn.relu(conv_output)

        return conv_output

    def residual_block(self, inputs):
        hp = self._hparams

        is_training = hp.mode == tf.estimator.ModeKeys.TRAIN
        filters = hp.num_filters

        add = tf.add

        conv1 = self.my_conv2d(inputs, filters=filters, kernel_size=3)
        conv1 = self.my_batchnorm(conv1, training=is_training)
        conv1 = tf.nn.relu(conv1)

        conv2 = self.my_conv2d(conv1, filters=filters, kernel_size=3)
        conv2 = self.my_batchnorm(conv2, training=is_training)

        add_residual = add(conv2, inputs)
        output = tf.nn.relu(add_residual)

        return output

    def bottom(self, features):
        self.max_game_length = tf.reduce_max(features["game_length"])
        return features

    def body(self, features):
        raise NotImplementedError("Abstract Method")

    def top(self, body_output, features):
        raise NotImplementedError("Abstract Method")

    def loss(self, logits, features):
        raise NotImplementedError("Abstract Method")

    def model_fn(self, features, mode):
        self.set_mode(mode)
        hp = self._hparams

        reuse = mode == tf.estimator.ModeKeys.EVAL
        is_training = mode == tf.estimator.ModeKeys.TRAIN

        with tf.variable_scope('model', reuse=reuse):
            transformed_features = self.bottom(features)
            body_out = self.body(transformed_features)
            logits = self.top(body_out, transformed_features)

            p_out, v_out = logits
            p_preds = tf.argmax(p_out, -1, output_type=tf.int32)
            v_preds = tf.round(v_out)

        p_loss, v_loss, l2_loss = self.loss(logits, transformed_features)
        loss = p_loss + hp.value_loss_weight * v_loss + l2_loss

        p_targets = transformed_features["p_targets"]
        p_acc = tf.reduce_mean(tf.cast(tf.equal(p_targets, p_preds), tf.float32))

        v_targets = transformed_features["v_targets"]
        v_acc = tf.reduce_mean(tf.cast(tf.equal(v_targets, v_preds), tf.float32))

        if is_training:
            global_step = tf.train.get_or_create_global_step()
            learning_rate = tf.train.piecewise_constant(global_step, hp.lr_boundaries, hp.lr_rates)

            optimizer = tf.train.MomentumOptimizer(learning_rate, hp.sgd_momentum)

            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                train_op = optimizer.minimize(loss, global_step=global_step)

        # -----------------------------------------------------------
        # METRICS AND SUMMARIES
        # Metrics for evaluation using tf.metrics (average over whole dataset)
        with tf.variable_scope("metrics"):
            metrics = {
                'policy_accuracy': tf.metrics.accuracy(labels=p_targets, predictions=p_preds),
                'value_accuracy': tf.metrics.accuracy(labels=v_targets, predictions=v_preds),
                'policy_loss': tf.metrics.mean(p_loss),
                'value_loss': tf.metrics.mean(v_loss),
                'l2_loss': tf.metrics.mean(l2_loss),
                'loss': tf.metrics.mean(loss)
            }

        # Group the update ops for the tf.metrics
        update_metrics_op = tf.group(*[op for _, op in metrics.values()])

        # Get the op to reset the local variables used in tf.metrics
        metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
        metrics_init_op = tf.variables_initializer(metric_variables)

        # Summaries for training
        tf.summary.scalar('policy_accuracy', p_acc)
        tf.summary.scalar('value_accuracy', v_acc)
        tf.summary.scalar('policy_loss', p_loss)
        tf.summary.scalar('value_loss', v_loss)
        tf.summary.scalar('l2_loss', l2_loss)
        tf.summary.scalar('loss', loss)

        # -----------------------------------------------------------
        # MODEL SPECIFICATION
        # Create the model specification and return it
        # It contains nodes or operations in the graph that will be used for training and evaluation
        model_spec = transformed_features
        variable_init_op = tf.group(*[tf.global_variables_initializer(), tf.tables_initializer()])
        model_spec['variable_init_op'] = variable_init_op

        model_spec['policy_predictions'] = p_preds
        model_spec['value_predictions'] = v_preds

        model_spec['value_loss'] = v_loss
        model_spec['loss'] = loss

        model_spec['policy_accuracy'] = p_acc
        model_spec['value_accuracy'] = v_acc

        model_spec['metrics_init_op'] = metrics_init_op
        model_spec['metrics'] = metrics
        model_spec['update_metrics'] = update_metrics_op
        model_spec['summary_op'] = tf.summary.merge_all()

        if is_training:
            model_spec['train_op'] = train_op

        return model_spec


class GoModelRNN(GoModel):
    def rnn_cell(self, board_size ):
        raise NotImplementedError("Abstract Method")

    def top(self, body_output, features):
        hp = self._hparams

        board_size = hp.board_size
        num_moves = hp.num_moves
        is_training = hp.mode == tf.estimator.ModeKeys.TRAIN

        legal_moves = features["legal_moves"]

        body_output = tf.reshape(body_output, [-1, 2, board_size, board_size])

        # Policy Head
        with tf.variable_scope('policy_head'):
            p_conv = self.my_conv2d(body_output, filters=2, kernel_size=1)
            p_conv = self.my_batchnorm(p_conv, center=False, scale=False, training=is_training)
            p_conv = tf.nn.relu(p_conv)

            logits = tf.reshape(p_conv, [-1, self.max_game_length, 2 * board_size * board_size])
            logits = tf.layers.dense(logits, num_moves)
            logits = tf.multiply(logits, legal_moves)

            p_output = tf.nn.softmax(logits, name='policy_output')

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

        return p_output, v_output

    def loss(self, logits, features):
        hp = self._hparams

        game_lengths = features["game_length"]
        mask = tf.sequence_mask(game_lengths)

        p_output, v_output = logits

        with tf.variable_scope('policy_loss'):
            p_targets = features["p_targets"]
            p_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=p_output, labels=p_targets)
            p_losses = tf.boolean_mask(p_losses, mask)
            p_loss = tf.reduce_mean(p_losses)

        with tf.variable_scope('value_loss'):
            v_targets = features['v_targets']
            v_losses = tf.square(v_output - v_targets)
            v_losses = tf.boolean_mask(v_losses, mask)
            v_loss = tf.reduce_mean(v_losses)

        with tf.variable_scope('l2_loss'):
            reg_vars = [v for v in tf.trainable_variables()
                        if 'bias' not in v.name and 'beta' not in v.name]
            l2_loss = hp.reg_strength * tf.add_n([tf.nn.l2_loss(v) for v in reg_vars])

        return p_loss, v_loss, l2_loss


class GoModelCNN(GoModel):
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

            logits = tf.reshape(p_conv, [-1, 2 * board_size * board_size])
            logits = tf.layers.dense(logits, num_moves)
            logits = tf.multiply(logits, legal_moves)

            p_output = tf.nn.softmax(logits, name='policy_output')

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

        return p_output, v_output

    def loss(self, logits, features):
        hp = self._hparams

        p_output, v_output = logits

        with tf.variable_scope('policy_loss'):
            p_targets = features["p_targets"]
            p_losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=p_output, labels=p_targets)
            p_loss = tf.reduce_mean(p_losses)

        with tf.variable_scope('value_loss'):
            v_targets = features['v_targets']
            v_losses = tf.square(v_output - v_targets)
            v_loss = tf.reduce_mean(v_losses)

        with tf.variable_scope('l2_loss'):
            reg_vars = [v for v in tf.trainable_variables()
                        if 'bias' not in v.name and 'beta' not in v.name]
            l2_loss = hp.reg_strength * tf.add_n([tf.nn.l2_loss(v) for v in reg_vars])

        return p_loss, v_loss, l2_loss


_already_logged = set()


def _eager_log(level, *args):
    if tf.contrib.eager.in_eager_mode() and args in _already_logged:
        return
    _already_logged.add(args)
    getattr(tf.logging, level)(*args)


def log_info(*args):
    _eager_log("info", *args)


def log_warn(*args):
    _eager_log("warn", *args)
