import numpy as np
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
        self.process = False
        self.nx, self.ny = 128, 128
        if self.ntime == 1:

            self.data = data

        elif self.ntime == data.shape[0]:

            self.data = np.swapaxes(np.expand_dims(data, axis=0), 1, 2)

        else:  # reshape input data to time segments
            self.process = True
            # number of examples of length ntime
            self.nexamples = (data.shape[0] // self.ntime)
            # constrain and reshape so we have (nexamples, nchannels, ntime, ...)
            self.data = np.swapaxes(np.reshape(data[0:(self.nexamples * self.ntime)], newshape=[self.nexamples, ntime] + list(data.shape[1:])), 1, 2)

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
            if self.process:
                # preprocess data in batches
                data_batch = self.preprocess(idx)
            return data_batch

        else:

            raise StopIteration
            
    def preprocess(self, idx):
        # preprocess data
        nexamples, nchannels, ntime, nx, ny = self.data.shape
        data_batch = np.zeros((len(idx), 1, ntime,  128, 128))
        j = 0 # index for data_batch
        for i in idx:
            for t in range(ntime):
                img = self.data[i,:,t,:,:]
                nchannels, nx, ny = img.shape
                img = np.reshape(img, (nx, ny, nchannels))
                img = Image.fromarray(np.uint8(img))
                img = img.resize((128, 128), Image.ANTIALIAS)
                img = np.expand_dims(np.array(img.convert('L')), axis=2)
                data_batch[j,:,t,:,:] = np.reshape(img/255.0, (1, 128, 128))
            j+=1
        return data_batch

    def is_final(self):
        """Flags if final iteration is reached

        :return: boolean if final batch is reached
        """

        return (self.batch_idx==self.nbatch)