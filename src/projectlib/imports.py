import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Activation,
    Conv2D,
    Conv2DTranspose,
    MaxPooling2D,
    ZeroPadding2D,
    LeakyReLU,
    ReLU,
    BatchNormalization,
    Flatten,
    AveragePooling2D,
    add,
    UpSampling2D,
    Dense,
    Flatten,
    Layer,
    Reshape,
    Dropout,
    Activation,
    Cropping2D,
)
from tensorflow.keras.regularizers import l1
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
from munch import Munch