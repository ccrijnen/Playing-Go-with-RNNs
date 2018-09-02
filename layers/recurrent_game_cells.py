import tensorflow as tf
from tensorflow.python.ops.rnn_cell_impl import LayerRNNCell

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"


class LSTSCell(LayerRNNCell):
    """Long-Short-Term-Strategy Cell"""

    def __init__(self,
                 num_units,
                 forget_bias=1.0,
                 activation=None,
                 reuse=None,
                 name=None,
                 dtype=None):
        super(LSTSCell, self).__init__(_reuse=reuse, name=name, dtype=dtype)

        self._num_units = num_units
        self._forget_bias = forget_bias
        self._activation = activation or tf.tanh

    @property
    def state_size(self):
        return tf.nn.rnn_cell.LSTMStateTuple(self._num_units, self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def build(self, inputs_shape):
        if inputs_shape[1].value is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                             % inputs_shape)

        input_depth = inputs_shape[1].value
        h_depth = self._num_units
        self._kernel = self.add_variable(
            _WEIGHTS_VARIABLE_NAME,
            shape=[input_depth + h_depth, 5 * self._num_units])
        self._bias = self.add_variable(
            _BIAS_VARIABLE_NAME,
            shape=[5 * self._num_units],
            initializer=tf.zeros_initializer(dtype=self.dtype))

        self.built = True

    def call(self, inputs, state):
        sigmoid = tf.sigmoid
        one = tf.constant(1, dtype=tf.dtypes.int32)

        b, features = inputs
        s, h = state

        gate_inputs = tf.matmul(
            tf.concat([b, h], 1), self._kernel)
        gate_inputs = tf.bias_add(gate_inputs, self._bias)

        ic, fc, oh, fh, gc = tf.split(
            value=gate_inputs, num_or_size_splits=5, axis=one)

        forget_bias_tensor = tf.constant(self._forget_bias, dtype=fc.dtype)

        add = tf.add
        multiply = tf.multiply

        ic = sigmoid(ic)
        fc = sigmoid(add(fc, forget_bias_tensor))
        gc = self._activation(gc)

        oh = sigmoid(oh)
        fh = sigmoid(fh)

        new_s = add(multiply(s, fc),
                    multiply(gc, ic))

        new_h = self._activation(
            add(multiply(features, fh),
                multiply(self._activation(new_s), oh)))

        new_state = tf.nn.rnn_cell.LSTMStateTuple(new_s, new_h)

        return new_h, new_state
