from .imports import np, tf
from numpy.random import permutation
from abc import ABC
from typing import Tuple

class Dataset(ABC):
    n_classes: int = None

    @staticmethod
    def get_X_y() -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError()

    def load_data(
        self,
        frac: float = 1.0,
        shuffle: bool = True,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray]:
        X, y = self.__class__.get_X_y()
        if shuffle:
            np.random.seed(random_state)
            shuffled = permutation(list(zip(X, y)))
            X, y = tuple(zip(*shuffled))
        if frac < 1.0:
            n = int(frac*len(X))
            X, y = X[:n], y[:n]
        return np.array(X), np.array(y)


# MNIST
class MNIST(Dataset):
    n_classes = 10

    @staticmethod
    def get_X_y():
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
        X = np.concatenate((X_train, X_test)).astype('float32') / 255
        X = np.reshape(X, (len(X), 28, 28, 1))
        y = np.concatenate((y_train, y_test))
        return X, y
mnist = MNIST()


# CIFAR-10
class CIFAR10(Dataset):
    n_classes = 10

    @staticmethod
    def get_X_y():
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
        X = np.concatenate((X_train, X_test)).astype('float32') / 255
        y = np.concatenate((y_train, y_test))
        return X, y
cifar10 = CIFAR10()


# CIFAR-100
class CIFAR100(Dataset):
    n_classes = 100
    
    @staticmethod
    def get_X_y():
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar100.load_data()
        X = np.concatenate((X_train, X_test)).astype('float32') / 255
        y = np.concatenate((y_train, y_test))
        return X, y
cifar100 = CIFAR100()