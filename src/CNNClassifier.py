# Feature classifier based on Convolution Neural Network
# author: Ben Quick
# email: benquickdenn@foxmail.com
# date: 2018-11-1

import tensorflow as tf

# global variable
CC_CHANNEL = 1  # the number of convolution core
CC_WIDTH = 5  # the width of convolution core window
CC_HEIGHT = 5  # the height of convolution core window
CC_STRIDE_X = 1  # the stride length of convolution core in X dimension
CC_STRIDE_Y = 1  # the stride length of convolution core in Y dimension
CC_PADDING = 'VALID'  # whether we need to consider the boundary pixel in convolution core

MP_WIDTH = 2  # the width of max pool window
MP_HEIGHT = 2  # the height of max pool window
MP_STRIDE_X = 2  # the stride length of max pool in X dimension
MP_STRIDE_Y = 2  # the stride length of max pool in Y dimension

# load data set


# define weight and bias function
# @param shape the number of dimension of tensor


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)  # generate normal distribution, stddev is the standard deviation
    return tf.Variable(initial)


# define convolution function


def convolution_2d(c_input, c_filter):
    # convolution on every stencil pixel of an image
    return tf.nn.conv2d(c_input, c_filter, strides=[1, CC_STRIDE_X, CC_STRIDE_Y, 1], padding=CC_PADDING)

# define max value pooling function


def pooling_2_2(p_input):
    return tf.nn.max_pool(p_input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # the size of each sample is
