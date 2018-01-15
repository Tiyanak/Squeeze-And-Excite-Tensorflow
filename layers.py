import tensorflow as tf
import numpy as np

def selu(x):

    alpha = 1.6733
    scale = 1.0507

    return scale * tf.where(x > 0.0, x, alpha * tf.exp(x) - alpha, 'selu')

def relu(x):

    return tf.nn.relu(x)

def sigmoid(x):

    return tf.nn.sigmoid(x)

def conv2d(input, filters=16, kernel_size=[5, 5], padding="same", stride=1, activation_fn=relu, name="conv2d"):

   return tf.layers.conv2d(input, filters=filters, kernel_size=kernel_size, padding=padding,
                           strides=stride, name=name, activation=activation_fn)

def max_pool2d(input, pool_size=None, strides=2, name="max_pool2d"):

    if pool_size is None:
        pool_size = [3, 3]

    return tf.layers.max_pooling2d(input, pool_size=pool_size, strides=strides, name=name)

def global_pool(input, pool_size=None, strides=1, name="global_pool2d", padding="valid"):

    if pool_size is None:
        pool_size = [input.shape[1], input.shape[2]]

    return tf.layers.average_pooling2d(input, pool_size=pool_size, strides=strides, padding=padding, name=name)

def flatten(input, name="flatten"):

    return tf.reshape(input, shape=[tf.shape(input)[0], np.prod(input.shape[1:]).value], name=name)

def fc(input, output_shape, activation_fn, name="fc"):

    return tf.layers.dense(input, output_shape, activation=activation_fn, name=name)

def batchNormalization(input, axis=-1):

    return tf.layers.batch_normalization(inputs=input, axis=axis)