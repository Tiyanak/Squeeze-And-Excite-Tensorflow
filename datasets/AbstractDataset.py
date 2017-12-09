from abc import abstractmethod


class AbstractDataset():

    @abstractmethod
    def train_set(self): pass

    @abstractmethod
    def validate_set(self): pass

    @abstractmethod
    def test_set(self): pass


