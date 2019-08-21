import numpy as np
import cv2
from PIL import Image

class DataSource:
    """"
    Returns lists of minibatches for image data
    """

    def __init__(self, data, batch_size=1, ntime=1):
        """

        :param data: tensor of size (timepoints,channels,x,y)
        :param batch_size: batch size
        :param ntime: number of consecutive time points to use
        """

        self.ntime = ntime

        if self.ntime == 1:

            self.data = data

        elif self.ntime == data.shape[0]:

            self.data = np.swapaxes(np.expand_dims(data, axis=0), 1, 2)

        else:  # reshape input data to time segments

            # number of examples of length ntime
            nexamples = (data.shape[0] // self.ntime)

            # constrain and reshape so we have (nexamples, nchannels, ntime, ...)
            self.data = np.swapaxes(np.reshape(data[0:(nexamples * self.ntime)], newshape=[nexamples, ntime] + list(data.shape[1:])), 1, 2)

        self.n = self.data.shape[0]

        self.idx = None

        self.batch_size = batch_size
        self.nbatch = self.n // self.batch_size
        self.batch_idx = None


    def __iter__(self):

        self.idx = np.arange(self.n)

        self.batch_idx = 0

        return self

    def __next__(self):

        if self.batch_idx < self.nbatch:

            idx = self.idx[self.batch_idx * self.batch_size:(self.batch_idx + 1) * self.batch_size]

            self.batch_idx += 1

            return self.data[idx]

        else:

            raise StopIteration

    def is_final(self):
        """Flags if final iteration is reached

        :return: boolean if final batch is reached
        """

        return (self.batch_idx==self.nbatch)