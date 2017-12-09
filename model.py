import tensorflow as tf
import layers as layers
from util import util, constant

class Model:

    def __init__(self, is_training=True):

        self.output_shape = constant.config['output_shape']
        self.pool_size = constant.config['pool_size']
        self.strides = constant.config['strides']
        self.learning_rate = constant.config['learning_rate']

        self.fc_1_output = constant.config['fc_1_output']
        self.fc_2_output = constant.config['fc_2_output']
        self.num_class = constant.config['num_class']

        self.batch_size = constant.config['batch_size']
        self.input_w = constant.config['img_width']
        self.input_h = constant.config['img_height']
        self.input_c = constant.config['img_channel']

        self.activation_fn = constant.config['activation_functions'][constant.config['activation_fn']]

        self.X = tf.placeholder(name='image', dtype=tf.float32, shape=[None, self.input_w, self.input_h, self.input_c])
        self.Yoh = tf.placeholder(name='label', dtype=tf.float32, shape=[None, self.num_class])

        self.is_training = tf.placeholder_with_default(bool(is_training), [], name='is_training')

        self.build()

    def loss_fn(self, logits, labels):

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(self.Yoh, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

        return (train_op, loss, accuracy)

    def model_net(self, image, activation_fn, is_training=True, scope="Classifier"):

        net = []

        with tf.variable_scope(scope) as scope:

            net += [layers.conv2d(image, filters=16, kernel_size=[5, 5], padding="same", activation_fn=self.activation_fn, name="conv_1")]
            net += [layers.max_pool2d(net[-1], pool_size=self.pool_size, strides=self.strides, name="max_pool_1")]

            net += [layers.conv2d(image, filters=32, kernel_size=[5, 5], padding="same", activation_fn=self.activation_fn, name="conv_2")]
            net += [layers.max_pool2d(net[-1], pool_size=self.pool_size, strides=self.strides, name="max_pool_2")]

            net += [layers.conv2d(image, filters=16, kernel_size=[5, 5], padding="same", activation_fn=self.activation_fn, name="conv_3")]
            net += [layers.max_pool2d(net[-1], pool_size=self.pool_size, strides=self.strides, name="max_pool_3")]

            if constant.config['use_se'] == True:
                se = self.squeeze_excite(net[-1], filters=16)
                net += [tf.multiply(net[-1], se[-1])]

            net += [layers.flatten(net[-1], name="flatten")]

            net += [layers.fc(net[-1], self.fc_1_output, is_train=is_training, activation_fn=activation_fn, name="fc_1")]
            net += [layers.fc(net[-1], self.fc_2_output, is_train=is_training, activation_fn=activation_fn, name="fc_2")]
            net += [layers.fc(net[-1], self.num_class, is_train=is_training, activation_fn=None, name="fc_3")]

        return net

    def predict(self, logits):

        return tf.nn.softmax(logits)

    def squeeze_excite(self, input, filters=16, is_training=True, scope="SE"):

        se = []

        with tf.variable_scope(scope) as scope:

            se += [layers.global_pool(input, name="se_global_pool")]
            se += [layers.fc(se[-1], filters, is_train=is_training, activation_fn=layers.relu, name="se_fc_1")]
            se += [layers.fc(se[-1], filters, is_train=is_training, activation_fn=layers.sigmoid, name="se_fc_2")]

        return se

    def build(self):

        self.net = self.model_net(self.X, self.activation_fn)
        self.logits = self.net[-1]
        self.prediction = self.predict(self.logits)
        self.train_op, self.loss, self.accuracy = self.loss_fn(self.logits, self.Yoh)

        # tf.summary missing here
