import numpy as np
import tensorflow as tf


class Vggnet:
    def __init__(self, inp, dropout_rate=0.5, width_fc6=512, width_fc7=512, num_features=100):
        self.x = inp
        self.drop_rate = dropout_rate
        self.w_fc6 = width_fc6
        self.w_fc7 = width_fc7
        self.w_fc8 = num_features
        with tf.name_scope('vggnet'):
            self.build()
        self.out = self.fc8

    def build(self):
        self.conv1_1 = self._conv_layer(self.x, "conv1_1", 64)
        self.conv1_2 = self._conv_layer(self.conv1_1, "conv1_2", 64)
        self.pool1 = self._max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self._conv_layer(self.pool1, "conv2_1", 128)
        self.conv2_2 = self._conv_layer(self.conv2_1, "conv2_2", 128)
        self.pool2 = self._max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self._conv_layer(self.pool2, "conv3_1", 256)
        self.conv3_2 = self._conv_layer(self.conv3_1, "conv3_2", 256)
        self.pool3 = self._max_pool(self.conv3_2, 'pool3')

        self.conv4_1 = self._conv_layer(self.pool3, "conv4_1", 512)
        self.conv4_2 = self._conv_layer(self.conv4_1, "conv4_2", 512)
        self.pool4 = self._max_pool(self.conv4_2, 'pool4')

        self.conv5_1 = self._conv_layer(self.pool4, "conv5_1", 512)
        self.conv5_2 = self._conv_layer(self.conv5_1, "conv5_2", 512)
        self.pool5 = self._avg_pool(self.conv5_2, 'pool5')
        self.flat_pool5 = tf.reshape(self.pool5, [-1, np.prod(self.pool5.get_shape().as_list()[1:])], name="flat_pool5")

        self.fc6 = self._fc_layer(self.flat_pool5, self.w_fc6, 'fc6')
        self.relu6 = tf.nn.relu(self.fc6)
        self.dropout6 = tf.nn.dropout(self.relu6, rate=1-self.drop_rate)

        self.fc7 = self._fc_layer(self.dropout6, self.w_fc7, 'fc7')
        self.relu7 = tf.nn.relu(self.fc7)
        self.dropout7 = tf.nn.dropout(self.relu7, rate=1-self.drop_rate)

        self.fc8 = self._fc_layer(self.dropout7, self.w_fc8, 'fc8')

    def _max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def _avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def _get_conv_filter(self, name, depth_in, depth_out):
        initializer = tf.contrib.layers.xavier_initializer()
        return tf.get_variable(name + '_filter', shape=[3, 3, depth_in, depth_out], initializer=initializer)

    def _get_conv_bias(self, name, depth):
        initializer = tf.constant_initializer(0.1)
        return tf.get_variable(name + '_biases', shape=[depth], initializer=initializer)

    def _conv_layer(self, bottom, name, depth):
        with tf.variable_scope(name):
            depth_in = bottom.get_shape().as_list()[-1]
            filt = self._get_conv_filter(name, depth_in, depth)
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            conv_biases = self._get_conv_bias(name, depth)
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)
            return relu

    def _get_fc_weights(self, bottom, dim_out, name):
        shape_bottom = bottom.get_shape().as_list()[1:]
        dim_in = np.prod(shape_bottom)
        initializer = tf.random_normal([dim_in, dim_out], stddev=0.1)
        return tf.get_variable(name + '_weights', initializer=initializer)

    def _get_fc_bias(self, dim_out, name):
        initializer = tf.random_normal([dim_out], stddev=0.1)
        return tf.get_variable(name + '_bias', initializer=initializer)

    def _fc_layer(self, bottom, dim_out, name):
        with tf.variable_scope(name):
            weights = self._get_fc_weights(bottom, dim_out, name)
            biases = self._get_fc_bias(dim_out, name)
            fc = tf.nn.bias_add(tf.matmul(bottom, weights), biases)
            return fc
