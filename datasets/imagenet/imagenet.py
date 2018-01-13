from datasets.AbstractDataset import AbstractDataset
from utils import constant, util
import numpy as np
import os

class ImagenetDataset(AbstractDataset):

    def __init__(self):

        self.name = 'mnist'
        self.__init_datasets__()

    def __init_datasets__(self):

        width = constant.config['imagenet_img_width']
        height = constant.config['imagenet_img_height']
        channel = constant.config['imagenet_img_channel']

        train_x = np.ndarray((0, width * height* channel), dtype=np.float32)
        train_y = []
        for i in range(1, 6):
            subset = util.unpickle(os.path.join(constant.DATA_DIR, 'data_batch_%d' % i))
            train_x = np.vstack((train_x, subset['data']))
            train_y += subset['labels']
        train_x = train_x.reshape((-1, channel, height, width)).transpose(0, 2, 3, 1)
        train_y = np.array(train_y, dtype=np.int32)

        subset = util.unpickle(os.path.join(constant.DATA_DIR, 'test_batch'))
        test_x = subset['data'].reshape((-1, channel, height, width)).transpose(0, 2, 3, 1).astype(np.float32)
        test_y = np.array(subset['labels'], dtype=np.int32)
        self.test_y = util.class_to_onehot(test_y)

        valid_size = 5000
        train_x, train_y = util.shuffle_data(train_x, train_y)
        train_y = util.class_to_onehot(train_y)

        valid_x = train_x[:valid_size, ...]
        self.valid_y = train_y[:valid_size, ...]

        train_x = train_x[valid_size:, ...]
        self.train_y = train_y[valid_size:, ...]

        data_mean = train_x.mean((0, 1, 2))
        data_std = train_x.std((0, 1, 2))

        self.train_x = (train_x - data_mean) / data_std
        self.valid_x = (valid_x - data_mean) / data_std
        self.test_x = (test_x - data_mean) / data_std

    def train_set(self):

        return (self.train_x, self.train_y)

    def validate_set(self):

        return (self.valid_x, self.valid_y)

    def test_set(self):

        return (self.test_x, self.test_y)

