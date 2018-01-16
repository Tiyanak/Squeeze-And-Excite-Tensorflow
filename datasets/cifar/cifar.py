from datasets.AbstractDataset import AbstractDataset
from utils import constant, util
import numpy as np
import os

class CifarDataset(AbstractDataset):

    def __init__(self):

        self.name = 'mnist'
        self.__init_datasets__()

    def __init_datasets__(self):

        width = constant.config['cifar_32_img_width']
        height = constant.config['cifar_32_img_height']
        channel = constant.config['cifar_32_img_channel']

        self.train_x = np.ndarray((0, width * height * channel), dtype=np.float32)
        self.train_y = np.ndarray((0, ))
        for i in range(1, 6):
            subset = util.unpickle(os.path.join(constant.CIFAR_DATA_DIR, 'data_batch_%d' % i))
            self.train_x = np.vstack((self.train_x, subset['data']))
            self.train_y = np.concatenate((self.train_y, subset['labels']), axis=0)
        self.train_x = self.train_x.reshape((-1, channel, height, width)).transpose(0, 2, 3, 1)
        self.train_y = np.array(self.train_y, dtype=np.uint8)

        subset = util.unpickle(os.path.join(constant.CIFAR_DATA_DIR, 'test_batch'))
        self.test_x = subset['data'].reshape((-1, channel, height, width)).transpose(0, 2, 3, 1).astype(np.uint8)
        self.test_y = np.array(subset['labels'], dtype=np.uint8)
        self.test_y = util.class_to_onehot(self.test_y, 10)

        valid_size = 5000
        self.train_x, self.train_y = util.shuffle_data(self.train_x, self.train_y)
        self.train_y = util.class_to_onehot(self.train_y, 10)

        self.valid_x = self.train_x[:valid_size, ...]
        self.valid_y = self.train_y[:valid_size, ...]

        self.train_x = self.train_x[valid_size:, ...]
        self.train_y = self.train_y[valid_size:, ...]

        data_mean = self.train_x.mean((0, 1, 2))
        data_std = self.train_x.std((0, 1, 2))

        self.train_x = (self.train_x - data_mean) / data_std
        self.valid_x = (self.valid_x - data_mean) / data_std
        self.test_x = (self.test_x - data_mean) / data_std

    def train_set(self):

        return (self.train_x, self.train_y)

    def validate_set(self):

        return (self.valid_x, self.valid_y)

    def test_set(self):

        return (self.test_x, self.test_y)

