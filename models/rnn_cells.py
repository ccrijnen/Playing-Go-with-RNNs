from tensorflow.python.ops import nn_ops
from tensorflow.python.keras import activations

import tensorflow as tf


class ConvRNNCell(tf.nn.rnn_cell.RNNCell):
    """A RNN cell with convolutions instead of multiplications."""
    def __init__(self, input_shape, output_channels, kernel_shape, activation=None, reuse=None, name="conv_rnn_cell"):
        """Construct ConvGRUCell.

        Args:
            input_shape: (int, int, int) Shape of the input as int tuple, excluding the batch size
            output_channels: (int) number of output channels of the conv LSTM
            kernel_shape: (int, int) Shape of kernel as in tuple of size 2
            activation: Activation function.
            reuse: (bool) whether to reuse the weights of a previous layer by the same name.
            name: Name of the module
        Raises:
            ValueError: If data_format is not 'channels_first' or 'channels_last'
        """
        super(ConvRNNCell, self).__init__(_reuse=reuse, name=name)
        self._input_shape = input_shape
        self._output_channels = output_channels
        self._kernel_shape = kernel_shape

        if activation:
            self._activation = activations.get(activation)
        else:
            self._activation = tf.tanh

        self._output_size = tf.TensorShape([self._output_channels] + self._input_shape[1:])

    @property
    def output_size(self):
        return self._output_size

    @property
    def state_size(self):
        return self._output_size

    def call(self, inputs, state, scope=None):
        args = [inputs, state]

        total_arg_size_depth = 0
        shapes = [a.get_shape().as_list() for a in args]
        shape_length = len(shapes[0])
        for shape in shapes:
            if len(shape) != 4:
                raise ValueError("Conv Linear expects 4D arguments: %s" % str(shapes))
            if len(shape) != len(shapes[0]):
                raise ValueError("Conv Linear expects all args "
                                 "to be of same Dimension: %s" % str(shapes))
            else:
                total_arg_size_depth += shape[1]
        dtype = [a.dtype for a in args][0]

        inputs = tf.concat(args, axis=1)

        strides = shape_length * [1]
        kernel = tf.get_variable('kernel', self._kernel_shape + [total_arg_size_depth, self._output_channels],
                                 dtype=dtype)

        new_hidden = tf.nn.conv2d(inputs, kernel, strides, padding='SAME', data_format='NCHW')

        output = self._activation(new_hidden)
        return output, output


class ConvGRUCell(tf.nn.rnn_cell.RNNCell):
    """A GRU cell with convolutions instead of multiplications."""
    def __init__(self, input_shape, output_channels, kernel_shape,
                 activation=tf.tanh,
                 normalize=True,
                 data_format='channels_last',
                 reuse=None):
        """Construct ConvGRUCell.

        Args:
            input_shape: (int, int, int) Shape of the input as int tuple, excluding the batch size
            output_channels: (int) number of output channels of the conv LSTM
            kernel_shape: (int, int) Shape of kernel as in tuple of size 2
            activation: Activation function.
            normalize: (bool) whether to layer normalize the conv output
            data_format: A string, one of 'channels_last' (default) or 'channels_first'. The ordering of the dimensions in
                the inputs. channels_last corresponds to inputs with shape (batch, height, width, channels) while
                channels_first corresponds to inputs with shape (batch, channels, height, width)
            reuse: (bool) whether to reuse the weights of a previous layer by the same name.
        Raises:
            ValueError: If data_format is not 'channels_first' or 'channels_last'
        """
        super(ConvGRUCell, self).__init__(_reuse=reuse)
        self._filters = output_channels
        self._kernel = kernel_shape
        self._activation = activation
        self._normalize = normalize

        self._channel_first_dict = {1: 'NCW', 2: 'NCHW', 3: 'NCDHW'}

        if data_format == 'channels_last':
            self._size = tf.TensorShape(input_shape + [self._filters])
            self._feature_axis = self._size.ndims
            self._data_format = None
        elif data_format == 'channels_first':
            self._size = tf.TensorShape([self._filters] + input_shape)
            self._feature_axis = 1
            self._data_format = self._channel_first_dict[len(self._kernel)]
        else:
            raise ValueError('Unknown data_format')

    @property
    def state_size(self):
        return self._size

    @property
    def output_size(self):
        return self._size

    def call(self, x, h):
        channels = x.shape[self._feature_axis].value

        with tf.variable_scope('gates'):
            inputs = tf.concat([x, h], axis=self._feature_axis)
            n = channels + self._filters
            m = 2 * self._filters if self._filters > 1 else 2
            W = tf.get_variable('kernel', self._kernel + [n, m])
            y = tf.nn.convolution(inputs, W, 'SAME', data_format=self._data_format)
            if self._normalize:
                r, u = tf.split(y, 2, axis=self._feature_axis)
                r = tf.contrib.layers.layer_norm(r)
                u = tf.contrib.layers.layer_norm(u)
            else:
                y += tf.get_variable('bias', [m], initializer=tf.ones_initializer())
                r, u = tf.split(y, 2, axis=self._feature_axis)
            r, u = tf.sigmoid(r), tf.sigmoid(u)

        with tf.variable_scope('candidate'):
            inputs = tf.concat([x, r * h], axis=self._feature_axis)
            n = channels + self._filters
            m = self._filters
            W = tf.get_variable('kernel', self._kernel + [n, m])
            y = tf.nn.convolution(inputs, W, 'SAME', data_format=self._data_format)
            if self._normalize:
                y = tf.contrib.layers.layer_norm(y)
            else:
                y += tf.get_variable('bias', [m], initializer=tf.zeros_initializer())
            h = u * h + (1 - u) * self._activation(y)

        return h, h


class MyConv2DLSTMCell(tf.nn.rnn_cell.RNNCell):
    """A LSTM cell with convolutions instead of multiplications.
    Reference:
        Xingjian, S. H. I., et al. "Convolutional LSTM network: A machine learning approach for precipitation
            nowcasting." Advances in Neural Information Processing Systems. 2015.
    """
    def __init__(self,
                 input_shape,
                 output_channels,
                 kernel_shape,
                 use_bias=True,
                 forget_bias=1.0,
                 activation=tf.tanh,
                 data_format='channels_last',
                 reuse=None,
                 name="conv_2d_lstm_cell"):
        """Construct Conv2DLSTMCell.

        Args:
            input_shape: (int, int, int) Shape of the input as int tuple, excluding the batch size
            output_channels: (int) number of output channels of the conv LSTM
            kernel_shape: (int, int) Shape of kernel as in tuple of size 2
            use_bias: (bool) whether the convolutions use biases
            forget_bias: (float) Forget bias
            activation: Activation function.
            data_format: A string, one of 'channels_last' (default) or 'channels_first'. The ordering of the dimensions in
                the inputs. channels_last corresponds to inputs with shape (batch, height, width, channels) while
                channels_first corresponds to inputs with shape (batch, channels, height, width)
            reuse: (bool) whether to reuse the weights of a previous layer by the same name.
            name: Name of the module
        Raises:
            ValueError: If data_format is not 'channels_first' or 'channels_last'
        """
        super(MyConv2DLSTMCell, self).__init__(_reuse=reuse, name=name)

        self._input_shape = input_shape
        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._use_bias = use_bias
        self._forget_bias = forget_bias
        self._activation = activation

        if data_format == 'channels_last':
            state_size = tf.TensorShape(self._input_shape[:-1] + [self._output_channels])
            self._state_size = tf.nn.rnn_cell.LSTMStateTuple(state_size, state_size)
            self._output_size = state_size

            self._feature_axis = self.state_size.ndims
            self._data_format = None
        elif data_format == 'channels_first':
            state_size = tf.TensorShape([self._output_channels] + self._input_shape[1:])
            self._state_size = tf.nn.rnn_cell.LSTMStateTuple(state_size, state_size)
            self._output_size = state_size

            self._feature_axis = 1
            self._data_format = 'NCHW'
        else:
            raise ValueError('Unknown data_format')

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def call(self, inputs, state, scope=None):
        cell, hidden = state
        args = [inputs, hidden]

        total_arg_size_depth = 0
        shapes = [a.get_shape().as_list() for a in args]
        shape_length = len(shapes[0])
        for shape in shapes:
            if len(shape) != 4:
                raise ValueError("Conv Linear expects 4D arguments: %s" % str(shapes))
            if len(shape) != len(shapes[0]):
                raise ValueError("Conv Linear expects all args "
                                 "to be of same Dimension: %s" % str(shapes))
            else:
                total_arg_size_depth += shape[self._feature_axis]
        dtype = [a.dtype for a in args][0]

        inputs = tf.concat(args, axis=self._feature_axis)

        num_features = 4 * self._output_channels if self._output_channels > 1 else 4
        strides = shape_length * [1]

        kernel = tf.get_variable('kernel', self._kernel_shape + [total_arg_size_depth, num_features], dtype=dtype)

        new_hidden = nn_ops.conv2d(inputs, kernel, strides, padding='SAME', data_format=self._data_format)
        if self._use_bias:
            new_hidden += tf.get_variable('bias', [num_features], initializer=tf.zeros_initializer(), dtype=dtype)

        gates = tf.split(new_hidden, 4, axis=self._feature_axis)

        input_gate, new_input, forget_gate, output_gate = gates
        new_cell = tf.sigmoid(forget_gate + self._forget_bias) * cell
        new_cell += tf.sigmoid(input_gate) * self._activation(new_input)
        output = self._activation(new_cell) * tf.sigmoid(output_gate)

        new_state = tf.nn.rnn_cell.LSTMStateTuple(new_cell, output)
        return output, new_state
