import numpy as np
from tensorflow import keras


class DataGenerator(keras.utils.Sequence):
    def __init__(self, gen, ncl):
        self.gen = gen
        self.ncl = ncl

    def __getitem__(self, _):
        ims, lbs = next(iter(self.gen))
        ims = np.swapaxes(np.swapaxes(ims.numpy(), 1, 3), 1, 2)
        lbs = np.eye(self.ncl)[lbs]
        return ims, lbs

    def __len__(self):
        return len(self.gen)
