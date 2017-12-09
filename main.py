from util import constant
from datasets.AbstractDataset import AbstractDataset
from datasets.cifar.cifar import CifarDataset
from datasets.imagenet.imagenet import ImagenetDataset
from datasets.mnist.mnist import MnistDataset
import cnn

def main():

    dataset_name = constant.config['dataset_name']
    dataset = AbstractDataset

    if dataset_name == 'mnist':
        dataset = MnistDataset()
    elif dataset_name == 'imagenet':
        dataset = ImagenetDataset()
    elif dataset_name == 'cifar':
        dataset = CifarDataset()
    else:
        raise ValueError(dataset_name)

    train_x, train_y = dataset.train_set()
    validate_x, validate_y = dataset.validate_set()
    test_x, test_y = dataset.test_set()

    trainer = cnn.CNN()

    trainer.train(train_x, train_y, validate_x, validate_y)
    test_yp = trainer.predict(test_x)

#   data.calculate_accuracy, precision, show images, and other stuff

if __name__ == '__main__':
    main()