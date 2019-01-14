from abc import ABC, abstractmethod


class AbsModel(ABC):
    """
    Abstract class for object oriented machine learning models.
    """

    def __init__(self, value):
        super.__init__()

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict(self):
        pass
