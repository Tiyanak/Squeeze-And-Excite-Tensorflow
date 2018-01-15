# from tensorflow.examples.tutorials.mnist import input_data
from datasets.AbstractDataset import AbstractDataset
from utils import constant


class MnistDataset(AbstractDataset):

    def __init__(self):

        self.name = 'mnist'
        # self.dataset = input_data.read_data_sets(constant.DATA_DIR, one_hot=True)
        self.__init_datasets__()

    def __init_datasets__(self):

        width = constant.config['mnist_img_width']
        height = constant.config['mnist_img_height']
        channel = constant.config['mnist_img_channel']

        # self.train_x = self.dataset.train.images.reshape([-1, width, height, channel])
        # self.train_y = self.dataset.train.labels
        # self.validate_x = self.dataset.validation.images.reshape([-1, width, height, channel])
        # self.validate_y = self.dataset.validation.labels
        # self.test_x = self.dataset.test.images.reshape([-1, width, height, channel])
        # self.test_y = self.dataset.test.labels

    # def train_set(self):
    #
    #     return (self.train_x, self.train_y)
    #
    # def validate_set(self):
    #
    #     return (self.validate_x, self.validate_y)
    #
    # def test_set(self):
    #
    #     return (self.test_x, self.test_y)

