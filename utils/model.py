import tensorflow as tf
from tensorflow.contrib.layers import flatten


def LeNet(x, n_channels=3, n_classes=43):

    conv1 = conv2d_relu(x, (5, 5, n_channels, 6))
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    conv2 = conv2d_relu(conv1, (5, 5, 6, 16))
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Flatten. Input = 5x5x16. Output = 400.
    fc0 = flatten(conv2)
    fc1 = fc_layer(fc0, 120)
    fc2 = fc_layer(fc1, 84)
    logits = tf.contrib.layers.fully_connected(fc2, n_classes)

    return logits


def conv2d_relu(x, w_shape, strides=1, padding='VALID'):
    W = tf.Variable(tf.truncated_normal(shape=w_shape, mean=mu, stddev=sigma))
    b = tf.Variable(tf.zeros(w_shape[-1]))
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1],
                     padding=padding)
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def fc_layer(x, num_outputs):
    fc1 = tf.contrib.layers.fully_connected(x, num_outputs)
    fc1 = tf.nn.dropout(fc1, keep_prob)
    return fc1