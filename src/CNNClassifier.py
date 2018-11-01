# Feature classifier based on Convolution Neural Network
# author: Ben Quick
# email: benquickdenn@foxmail.com
# date: 2018-11-1

import tensorflow as tf

# global variable
IMAGE_SIZE = 256

C_1_WIDTH = 40
C_1_HEIGHT = 40
C_1_CORE = 32

C_2_WIDTH = 40
C_2_HEIGHT = 40
C_2_CORE = 64

CC_STRIDE_X = 1  # the stride length of convolution core in X dimension
CC_STRIDE_Y = 1  # the stride length of convolution core in Y dimension
CC_PADDING = 'SAME'  # whether we need to consider the boundary pixel in convolution core

MP_WIDTH = 40  # the width of max pool window
MP_HEIGHT = 40  # the height of max pool window
MP_STRIDE_X = 4  # the stride length of max pool in X dimension
MP_STRIDE_Y = 4  # the stride length of max pool in Y dimension
MP_PADDING = 'SAME'  # whether we need to consider the boundary pixel in max pool

NUM_HIDDEN_LAYER = 1024  # the number of hidden layer
NUM_NEURON = 10  # the number of neuron

# load data set


# define weight and bias function
# @param shape the number of dimension of tensor


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)  # generate normal distribution, stddev is the standard deviation
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# define convolution function


def convolution_2d(c_input, weight):
    # convolution on every stencil pixel of an image
    return tf.nn.conv2d(c_input, weight, strides=[1, CC_STRIDE_X, CC_STRIDE_Y, 1], padding=CC_PADDING)

# define max value pooling function


def max_pooling_2x2(p_input):
    return tf.nn.max_pool(p_input, ksize=[1, MP_WIDTH, MP_HEIGHT, 1],
                          strides=[1, MP_STRIDE_X, MP_STRIDE_Y, 1], padding=MP_PADDING)


def deep_learning(x):
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, IMAGE_SIZE, IMAGE_SIZE, 1])
    # first convolutional layer
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([40, 40, 1, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(convolution_2d(x_image, W_conv1) + b_conv1)

    with tf.name_scope('pool1'):
        h_pool1 = max_pooling_2x2(h_conv1)

    # Second convolutional layer -- maps 32 feature maps to 64.
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(convolution_2d(h_pool1, W_conv2) + b_conv2)

    with tf.name_scope('pool2'):
        h_pool2 = max_pooling_2x2(h_conv2)

    # connected layer
    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # output
    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    return y_conv, keep_prob


def construct_flow_graph(x):
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, IMAGE_SIZE, IMAGE_SIZE, 1])
    # the first convolution layer
    with tf.name_scope('convolution_1'):
        weight_convolution_1 = weight_variable([C_1_WIDTH, C_2_HEIGHT, 1, C_1_CORE])
        bias_convolution_1 = bias_variable([C_1_CORE])
        # ReLu function
        h_convolution_1 = tf.nn.relu(convolution_2d(x_image, weight_convolution_1) + bias_convolution_1)
    # the first pooling layer
    with tf.name_scope('pool_1'):
        h_pool_1 = max_pooling_2x2(h_convolution_1)
    # the second convolution layer
    with tf.name_scope('convolution_2'):
        weight_convolution_2 = weight_variable([C_2_WIDTH, C_2_HEIGHT, 32, C_2_CORE])
        bias_convolution_2 = bias_variable([C_2_CORE])
        h_convolution_2 = tf.nn.relu(convolution_2d(h_pool_1, weight_convolution_2) + bias_convolution_2)
        h_pool_2 = max_pooling_2x2(h_convolution_2)
    # the first fully connected layer
    # combine all pooled tensor into a vector
    with tf.name_scope('fully_connect_1'):
        vector_size = int((IMAGE_SIZE / (MP_STRIDE_X * MP_STRIDE_X)) * (IMAGE_SIZE / (MP_STRIDE_Y * MP_STRIDE_Y)))
        weight_fully_connect_1 = weight_variable([vector_size, NUM_HIDDEN_LAYER])
        bias_fuuly_connect_1 = bias_variable([NUM_HIDDEN_LAYER])
        h_pool_2_flat = tf.reshape(h_pool_2, [-1, vector_size])
        h_fully_connect_1 = tf.nn.relu(tf.matmul(h_pool_2_flat, weight_fully_connect_1) + bias_fuuly_connect_1)
    # dropout,to avoid overfitting
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)  # A scalar Tensor with the same type as x. The probability that each element is kept.
        h_fullly_drop = tf.nn.dropout(h_fully_connect_1, keep_prob)
    # output layer
    with tf.name_scope('output'):
        weight_output = weight_variable([NUM_HIDDEN_LAYER, NUM_NEURON])
        bias_output = weight_variable([NUM_NEURON])
        y_convolution = tf.nn.softmax(tf.matmul(h_fullly_drop, weight_output) + bias_output)
    return y_convolution, keep_prob

