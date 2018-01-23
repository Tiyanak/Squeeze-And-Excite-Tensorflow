from utils import constant, util
import numpy as np
import os
import tensorflow as tf

class ImageResizer():

    def __init__(self):

        self.__init_datasets__()

    def __init_datasets__(self):

        width = 32
        height = 32
        channel = 3

        train_x = np.ndarray((0, width * height * channel), dtype=np.float32)
        train_y = []
        for i in range(1, 6):
            subset = util.unpickle(os.path.join(constant.CIFAR_DATA_DIR, 'data_batch_%d' % i))
            train_x = np.vstack((train_x, subset['data']))
            train_y += subset['labels']
        self.train_x = train_x.reshape((-1, channel, height, width)).transpose(0, 2, 3, 1)
        self.train_y = np.array(train_y, dtype=np.uint8)

        subset = util.unpickle(os.path.join(constant.CIFAR_DATA_DIR, 'test_batch'))
        self.test_x = subset['data'].reshape((-1, channel, height, width)).transpose(0, 2, 3, 1).astype(np.float32)
        self.test_y = np.array(subset['labels'], dtype=np.uint8)

        data_mean = self.train_x.mean((0, 1, 2))
        data_std = self.train_x.std((0, 1, 2))

        self.train_x = (self.train_x - data_mean) / data_std
        self.test_x = (self.test_x - data_mean) / data_std

    def resize(self):

        batch_size = 10000
        w = 128; h = 128; c = 3

        with tf.variable_scope('image_resizer') as scope:

            tf_batch = tf.placeholder(dtype=tf.float32, shape=[batch_size, 32, 32, 3])
            resized_images = tf.image.resize_images(tf_batch, [w, h], method=tf.image.ResizeMethod.BICUBIC)
            tf_data = tf.reshape(tf.transpose(resized_images, [0, 3, 1, 2]), [batch_size, w * h * c])
            tf_data = tf.cast(tf_data, tf.uint8)

        session = tf.Session()
        session.run(tf.global_variables_initializer())

        for i in range(0, (int)(len(self.train_x) / batch_size)):
            dict = {}
            batch = self.train_x[(i * batch_size):((i + 1) * batch_size)]

            dict['data'] = session.run([tf_data], feed_dict={tf_batch: batch})
            dict['labels'] = self.train_y[(i * batch_size):((i + 1) * batch_size)]

            util.pickle_save(dict, os.path.join(constant.CIFAR_DATA_DIR_128, 'data_batch_%d' % (i+1)))
            print('Saved resized batch number {}'.format(i+1))

        for i in range(0, (int)(len(self.test_x) / batch_size)):
            dict = {}
            batch = self.test_x[(i * batch_size):((i + 1) * batch_size)]

            dict['data'] = session.run([tf_data], feed_dict={tf_batch: batch})
            dict['labels'] = self.test_y[(i * batch_size):((i + 1) * batch_size)]

            util.pickle_save(dict, os.path.join(constant.CIFAR_DATA_DIR_128, 'test_batch_%d' % (i+1)))
            print('Saved resized test batch number {}'.format(i+1))

imageResizer = ImageResizer()
imageResizer.resize()


