from utils import constant
from datasets.cifar.cifar import CifarDataset
from datasets.mnist.mnist import MnistDataset
from datasets.imagenet.imagenet import ImagenetDataset
from cnn import CNN
from cnn_records import CNN_Records

def main():

    dataset_name = constant.config['dataset_name']

    if dataset_name == 'mnist':

        dataset = MnistDataset()
        trainer = CNN()
        trainer.train(dataset)

    elif dataset_name == 'cifar_32':

        dataset = CifarDataset()
        trainer = CNN()
        trainer.train(dataset)

    elif dataset_name == 'cifar_128':

        trainer = CNN_Records()
        trainer.train()

    elif dataset_name == 'imagenet':

        dataset = ImagenetDataset()
        trainer = CNN()
        trainer.train(dataset)

    else:
        raise ValueError(dataset_name)

if __name__ == '__main__':
    main()