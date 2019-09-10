import sys
sys.path.append('..')
from data import DataSource
from PIL import Image
import numpy as np
from apc import APCModel
# import cv2
import scipy.io as sio
# import time

# denk aan batch size, cutoff, ntime en hun interacties.
# vervang videosource e.d.

# device to run model on
device = -1

# number of saccades per example
ntime = 10

## prepare data; data should be multiple unique examples / frames of a movie (across time)

data = np.repeat(np.expand_dims(sio.loadmat('../data/Stimuli.mat')['stimTrn'], axis=1)[:5], repeats=ntime, axis=0)
data -= np.min(data.flatten())
data /= np.max(data.flatten())

source = DataSource(data, ntime=ntime, batch_size=5)

## Train model

# number of image channels
nout = source.data.shape[1]

model = APCModel(nhidden=10, nout=nout)

model.train(source, nepochs=1000)