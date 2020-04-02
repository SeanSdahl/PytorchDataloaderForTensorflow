import numpy as np
import tensorflow.keras as k


class DataGenerator(k.utils.Sequence):
    
    def __init__(self, gen, ncl):
        self.gen = gen
        self.iter = iter(gen)
        self.ncl = ncl

    def __getitem__(self, _):
        try:
            ims, lbs = next(self.iter)
        except StopIteration:
            self.iter = iter(self.gen)
            ims, lbs = next(self.iter)
        ims = np.swapaxes(np.swapaxes(ims.numpy(), 1, 3), 1, 2)
        lbs = np.eye(self.ncl)[lbs].reshape(self.gen.batch_size, self.ncl)
        return ims, lbs

    def __len__(self):
        return len(self.gen)
