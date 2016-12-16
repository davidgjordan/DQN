import tensorflow as tf
import numpy as np


class ConvNet(object):
    """Creates the tensorflow computation graph for a feed-forward convolutional net with rectifier nonlinearities"""

    def __init__(self, height, width, num_channels, num_class, num_filters, filter_shapes, filter_strides,
                 fc_hidden_units, pooling_size=None, pooling_stride=2):

        self.width = width
        self.height = height
        self.num_class = num_class
        self.num_filters = [num_channels] + num_filters
        self.filter_shapes = filter_shapes
        self.filter_strides = filter_strides
        self.pooling_size = pooling_size
        self.pooling_stride = pooling_stride
        self.fc_units = fc_hidden_units
        with tf.name_scope("inputs"):
            self.x = tf.placeholder(tf.float32, [None, self.height, self.width, self.num_filters[0]], name='x-input')
        self.biases_conv = None
        self.weights_conv = None
        self.biases_fc = None
        self.weights_fc = None
        self.create_conv_variables()
        self.hypothesis = self.create_hypothesis()
        self.weights = self.biases_conv + self.weights_conv + self.biases_fc + self.weights_fc

    def create_hypothesis(self):
        return self.fc_feedforward(self.convolutional_feedforward(self.x))

    def convolutional_feedforward(self, h):
        for (w, b, stride) in zip(self.weights_conv, self.biases_conv, self.filter_strides):
            h = self.create_conv_layer(h, w, b, stride)
        return self.input_to_fc_layers(h)

    def fc_feedforward(self, h):
        for (w, b) in zip(self.weights_fc, self.biases_fc)[:-1]:
            h = self.create_fc_layer(h, w, b)
        # treat final layer separately
        h = self.create_fc_layer(h, self.weights_fc[-1], self.biases_fc[-1], apply_relu=False)
        return h

    def input_to_fc_layers(self, h):
        h, input_size = self.flatten_conv_layer(h)
        self.create_fc_variables(input_size)
        return h

    def create_conv_layer(self, input, w, b, stride):
        with tf.name_scope('convolution_layer'):
            output = tf.nn.conv2d(input=input, filter=w,
                                  strides=[1, stride, stride, 1],
                                  padding='SAME')
            output = tf.nn.relu(output + b)

        if self.pooling_size is not None:
            tf.nn.max_pool(value=output,
                           ksize=[1, self.pooling_size, self.pooling_size, 1],
                           strides=[1, self.pooling_stride, self.pooling_stride, 1],
                           padding='SAME')
        return output

    @staticmethod
    def create_fc_layer(input, w, b, apply_relu=True):
        with tf.name_scope('fc_layer'):
            output = tf.matmul(input, w) + b
            if apply_relu:
                output = tf.nn.relu(output)
        return output

    @staticmethod
    def flatten_conv_layer(input):
        size = np.prod(input.get_shape()[1:].as_list())
        return tf.reshape(input, [-1, size]), size

    @staticmethod
    def rand_variable(shape, name, sigma=0.1):
        initial = tf.truncated_normal(shape, stddev=sigma, name=name)
        return tf.Variable(initial)

    @staticmethod
    def const_variable(shape, name, c=0.1):
        initial = tf.constant(c, shape=shape, name=name)
        return tf.Variable(initial)

    # TODO: These methods are quite terse. Think of a way to simplify them.

    def create_conv_variables(self):
        with tf.name_scope('conv_variables'):
            self.biases_conv = [self.const_variable([i], name='biases_conv{}'.format(layer+1))
                                for layer, i in enumerate(self.num_filters[1:])]
            self.weights_conv = [self.rand_variable([k, k, i, j], name='weights_conv{}'.format(layer+1))
                                 for layer, (i, j, k) in
                                 enumerate(zip(self.num_filters[:-1], self.num_filters[1:], self.filter_shapes))]

    def create_fc_variables(self, input_size):
        with tf.name_scope('fc_variables'):
            sizes = [input_size] + self.fc_units + [self.num_class]
            self.biases_fc = [self.const_variable([i], name='biases_fc{}'.format(layer+1))
                              for layer, i in enumerate(sizes[1:])]
            self.weights_fc = [self.rand_variable([i, j], name='weights_fc{}'.format(layer+1))
                               for layer, (i, j) in enumerate(zip(sizes[:-1], sizes[1:]))]

    # update weights and biases of self to those of another network
    def update_to(self, net):
        for w_target, b_target, w, b in zip(self.weights_conv, self.biases_conv, net.weights_conv, net.biases_conv):
            tf.assign(w_target, w)
            tf.assign(b_target, b)
        for w_target, b_target, w, b in zip(self.weights_fc, self.biases_fc, net.weights_fc, net.biases_fc):
            tf.assign(w_target, w)
            tf.assign(b_target, b)

