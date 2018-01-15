import tensorflow as tf
from utils import util

class TFRecordsReader():

    def __init__(self):

        self.fileDict = util.constant.CIFAR_FILE_DICT

    def create_iterator(self, dataset_type, num_epochs, batch_size, capacity):

        self.feature = {dataset_type + '/image': tf.FixedLenFeature([], tf.string),
                   dataset_type + '/label': tf.FixedLenFeature([], tf.int64)}

        self.filename_queue = tf.train.string_input_producer(self.fileDict[dataset_type], num_epochs=num_epochs)

        self.reader = tf.TFRecordReader()
        _, ser_example = self.reader.read(self.filename_queue)

        self.features = tf.parse_single_example(ser_example, features=self.feature)

        self.image = tf.decode_raw(self.features[dataset_type + '/image'], tf.float32)
        self.label = tf.cast(self.features[dataset_type + '/label'], tf.int32)

        self.image = tf.reshape(self.image, [128, 128, 3])

        self.images, self.labels = tf.train.shuffle_batch([self.image, self.label], batch_size=batch_size,
                                                              capacity=capacity, num_threads=1, min_after_dequeue=50)



