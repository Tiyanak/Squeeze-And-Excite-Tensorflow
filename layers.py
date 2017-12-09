import tensorflow as tf
import tensorflow.contrib.layers as tf_layers

def selu(x):

    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946

    return scale * tf.where(x > 0.0, x, alpha * tf.exp(x) - alpha, 'selu')

def relu(x):

    return tf.nn.relu(x)

def sigmoid(x):

    return tf.nn.sigmoid(x)

def conv2d(input, filters=16, kernel_size=[5, 5], padding="same", activation_fn=relu, name="conv2d"):
           # output_shape, is_train, activation_fn, k_h=5, k_w=5, stddev=0.02, name="conv2d"):

   return tf.layers.conv2d(input, filters=filters, kernel_size=kernel_size, padding=padding, name=name,
                           activation=activation_fn)

    # with tf.variable_scope(name, reuse=None):

        # w = tf.get_variable('w', [k_h, k_w, input.get_shape()[-1], output_shape],
        #                     initializer=tf.truncated_normal_initializer(stddev=stddev))
        #
        # conv = tf.nn.conv2d(input, w, strides=[1, 2, 2, 1], padding='SAME', name=name)
        # biases = tf.get_variable('biases', [output_shape], initializer=tf.constant_initializer(0.0))
        # _ = activation_fn(conv + biases)
        #
        # if activation_fn is not selu:
        #     _ = tf.contrib.layers.batch_norm(_, center=True, scale=True, decay=0.9, is_training=is_train, updates_collections=None)

    # return _

def max_pool2d(input, pool_size=None, strides=2, name="max_pool2d"):

    if pool_size is None:
        pool_size = [3, 3]

    return tf.layers.max_pooling2d(input, pool_size=pool_size, strides=strides, name=name)

def global_pool(input, pool_size=None, strides=1, name="global_pool2d", padding="valid"):

    if pool_size is None:
        pool_size = [input.shape[1], input.shape[2]]

    return tf.layers.average_pooling2d(input, pool_size=pool_size, strides=strides, padding=padding, name=name)

def flatten(input, name="flatten"):

    return tf_layers.flatten(input, scope=name)


def fc(input, output_shape, is_train, activation_fn, name="fc"):

    output = tf_layers.fully_connected(input, output_shape, activation_fn=activation_fn, scope=name)

    return output