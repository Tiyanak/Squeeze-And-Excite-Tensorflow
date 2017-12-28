import tensorflow as tf
import layers as layers
from util import util, constant

class Model:

    def __init__(self, is_training=True):

        self.output_shape = constant.config['output_shape']
        self.pool_size = constant.config['pool_size']
        self.strides = constant.config['strides']
        self.starter_learning_rate = constant.config['learning_rate']

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

        self.global_step = tf.Variable(0, trainable=False)

        self.is_training = tf.placeholder_with_default(bool(is_training), [], name='is_training')

        self.build()

    def build(self):

        if (constant.config['model'] == 'resnet50'):
            self.net = self.resnet_50(self.X, self.activation_fn)
        else:
            self.net = self.model_net(self.X, self.activation_fn)

        self.logits = self.net[-1]
        self.prediction = self.predict(self.logits)
        self.train_op, self.loss, self.accuracy, self.learning_rate = self.loss_fn(self.logits, self.Yoh)

        # tf.summary missing here

    def loss_fn(self, logits, labels):

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(self.Yoh, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        learning_rate = tf.train.exponential_decay(self.starter_learning_rate,
                                                        self.global_step, constant.config['decay_steps'],
                                                        constant.config['decay_base'], staircase=False)

        train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=self.global_step)

        return (train_op, loss, accuracy, learning_rate)

    def model_net(self, image, activation_fn, scope="Classifier"):

        net = []

        with tf.variable_scope(scope) as scope:

            net += [layers.conv2d(image, filters=16, kernel_size=[5, 5], padding="same", activation_fn=self.activation_fn, name="conv_1")]
            net += [layers.max_pool2d(net[-1], pool_size=self.pool_size, strides=self.strides, name="max_pool_1")]

            net += [layers.conv2d(image, filters=32, kernel_size=[5, 5], padding="same", activation_fn=self.activation_fn, name="conv_2")]
            net += [layers.max_pool2d(net[-1], pool_size=self.pool_size, strides=self.strides, name="max_pool_2")]

            net += [layers.conv2d(image, filters=64, kernel_size=[5, 5], padding="same", activation_fn=self.activation_fn, name="conv_3")]
            net += [layers.max_pool2d(net[-1], pool_size=self.pool_size, strides=self.strides, name="max_pool_3")]

            # SE block
            if constant.config['use_se'] == True:
                se = self.squeeze_excite(net[-1], filters=64)
                net += [tf.multiply(net[-1], se)]

            net += [layers.flatten(net[-1], name="flatten")]

            net += [layers.fc(net[-1], self.fc_1_output, activation_fn=activation_fn, name="fc_1")]
            net += [layers.fc(net[-1], self.fc_2_output, activation_fn=activation_fn, name="fc_2")]
            net += [layers.fc(net[-1], self.num_class, activation_fn=None, name="fc_3")]

        return net

    def predict(self, logits):

        return tf.nn.softmax(logits)

    def squeeze_excite(self, input, filters=16, scope="SE", se_conv_block_name="conv1"):

        filters1 = (int) (filters / constant.config['se_r'])

        with tf.variable_scope(scope) as scope:

            se = layers.global_pool(input, name="se_global_pool")
            se = layers.fc(se, filters1, activation_fn=layers.relu, name="se_fc_" + se_conv_block_name + "_1")
            se = layers.fc(se, filters, activation_fn=layers.sigmoid, name="se_fc_" + se_conv_block_name + "_2")

        return se

    def resnet_50(self, image, activation_fn, scope="Classifier"):

        net = []

        with tf.variable_scope(scope) as scope:

            # conv1 block
            net += [layers.conv2d(image, filters=64, kernel_size=[7, 7], padding="same", stride=2, activation_fn=None, name="conv1")]
            net[-1] = layers.batchNormalization(net[-1])
            net[-1] = self.activation_fn(net[-1])
            net[-1] = layers.max_pool2d(net[-1], pool_size=[3, 3], strides=2, name="max_pool_1")

            # conv2 block
            net += [self.conv_block(net[-1], [64, 64, 256], 3, "conv2", strides=1)]

            # conv3 block
            net += [self.conv_block(net[-1], [128, 128, 512], 4, "conv3")]

            # conv4 block
            net += [self.conv_block(net[-1], [256, 256, 1024], 6, "conv4")]

            # conv5 block
            net += [self.conv_block(net[-1], [512, 512, 2048], 3, "conv5")]

            # Ending block
            global_avg_pool = [7, 7]
            if (net[-1].shape[1] < 7 or net[-1].shape[2] < 7):
                global_avg_pool = [1, 1]

            net += [layers.global_pool(net[-1], pool_size=global_avg_pool, strides=1, name="global_pool_avg")]

            net += [layers.flatten(net[-1], name="flatten")]

            net += [layers.fc(net[-1], 1000, activation_fn=activation_fn, name="fc_1000")]

            if constant.config['dataset_name'] != 'imagenet':
                net += [layers.fc(net[-1], 512, activation_fn=activation_fn, name="fc_1")]
                net += [layers.fc(net[-1], 256, activation_fn=activation_fn, name="fc_2")]
                net += [layers.fc(net[-1], self.num_class, activation_fn=None, name="fc_3")]

        return net

    def conv_block(self, input, filters, repeat_num, conv_name="conv", strides=2):

        net = input
        block_input = input
        filters1, filters2, filters3 = filters

        for i in range(repeat_num):

            repeat_index = str(i)
            se_name = conv_name + "_" + repeat_index
            conv_name1, conv_name2, conv_name3 = [conv_name + "_" + repeat_index + "_1",
                                                  conv_name + "_" + repeat_index + "_2",
                                                  conv_name + "_" + repeat_index + "_3"]
            block_strides = strides
            if (i != 0):
                block_strides = 1

            net = layers.conv2d(block_input, filters=filters1, kernel_size=[1, 1], padding="valid", stride=block_strides, activation_fn=None, name=conv_name1)
            net = layers.batchNormalization(net)
            net = self.activation_fn(net)

            net = layers.conv2d(net, filters=filters2, kernel_size=[3, 3], padding="same", stride=1, activation_fn=None, name=conv_name2)
            net = layers.batchNormalization(net)
            net = self.activation_fn(net)

            net = layers.conv2d(net, filters=filters3, kernel_size=[1, 1], padding="valid", stride=1, activation_fn=None, name=conv_name3)
            net = layers.batchNormalization(net)

            # SE block
            if constant.config['use_se'] == True:
                se = self.squeeze_excite(net, filters=filters3, se_conv_block_name=se_name)
                net = tf.multiply(net, se)

            if (i == 0):
                # calculating shortcut for next block iteration - then put in block_input for programming easyness purposes
                shortcut = layers.conv2d(block_input, filters3, kernel_size=[1, 1], padding="valid", stride=block_strides, activation_fn=None, name="shortcut_" + conv_name + "_" + repeat_index)
                shortcut = layers.batchNormalization(shortcut)
                block_input = shortcut

            net = tf.add(net, block_input)
            net = self.activation_fn(net)

            # always need to save the first inputs in the block to sum with result of block
            block_input = net

        return net