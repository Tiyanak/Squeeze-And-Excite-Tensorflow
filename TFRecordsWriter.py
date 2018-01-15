import tensorflow as tf
from utils import util, constant
import numpy as np
import os
import matplotlib.pyplot as plt

class TFRecordsWriter():

    def __init__(self):

        print("TFRecordsWriter created")

    def start_writer(self):

        trainFiles = [os.path.join(util.constant.CIFAR_DATA_DIR_128, 'data_batch_%d' % x) for x in range(1, 6)]
        testFile = os.path.join(util.constant.CIFAR_DATA_DIR_128, 'test_batch')

        valid_x = np.ndarray((0, 128, 128, 3), dtype=np.float32)
        valid_y = np.ndarray((0,), dtype=np.int64)

        for i in range(0, len(trainFiles)):

            print("Processing train file: {}".format(trainFiles[i]))

            train_x, train_y = self.read_numpy_batch(trainFiles[i])
            train_x, train_y, valid_x_temp, valid_y_temp = self.get_valid(train_x, train_y)

            valid_x = np.vstack((valid_x, valid_x_temp))
            valid_y = np.concatenate((valid_y, valid_y_temp), axis=0)

            # write train batch
            self.write(os.path.join(util.constant.CIFAR_TF_RECORDS_DIR, 'train_batch_%d.tfrecords' % (i+1)), train_x, train_y, 'train')
            train_x = None; train_y = None

        # write valid batch
        print("Processing valid batch")
        self.write(os.path.join(util.constant.CIFAR_TF_RECORDS_DIR, 'valid_batch.tfrecords'), valid_x, valid_y, 'valid')
        valid_x = None; valid_y = None

        # write test batch
        print("Processing test file: {}".format(testFile))
        test_x, test_y = self.read_numpy_batch(testFile)
        self.write(os.path.join(util.constant.CIFAR_TF_RECORDS_DIR, 'test_batch.tfrecords'), test_x, test_y, 'test')
        test_x = None; test_y = None

    def get_valid(self, train_x, train_y, valid_size=1000):

        valid_x = train_x[:valid_size, ...]
        valid_y = train_y[:valid_size, ...]

        train_x = train_x[valid_size:, ...]
        train_y = train_y[valid_size:, ...]

        return (train_x, train_y, valid_x, valid_y)

    def read_numpy_batch(self, filename):

        width = 128; height = 128; channel = 3

        batch_x = np.ndarray((0, width * height * channel), dtype=np.float32)
        batch_y = []

        subset = util.unpickle(filename)
        batch_x = np.vstack((batch_x, subset['data'][0]))
        batch_y = np.concatenate((batch_y, subset['labels']), axis=0)

        subset.clear()

        batch_x = batch_x.reshape((-1, channel, height, width)).transpose(0, 2, 3, 1)
        batch_y = np.array(batch_y, dtype=int)

        batch_x, batch_y = util.shuffle_data(batch_x, batch_y)

        return (batch_x, batch_y)

    def write(self, filename, images, labels, dataset_type):

        writer = tf.python_io.TFRecordWriter(filename)

        for i in range(0, len(images)):

            if not i % 1000:
                print("Images: {}/{}".format(i, len(images)))

            image = images[i]
            label = labels[i]

            feature = {dataset_type + '/label' : util._int64_feature(np.asscalar(label)),
                       dataset_type + '/image' : util._bytes_feature(tf.compat.as_bytes(image.tostring()))}

            example = tf.train.Example(features=tf.train.Features(feature=feature))

            writer.write(example.SerializeToString())

        writer.close()

tfRecordWriter = TFRecordsWriter()
tfRecordWriter.start_writer()