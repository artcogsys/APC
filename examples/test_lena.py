from data import DataSource
from PIL import Image
import numpy as np
from apc import APCModel

# device to run model on
device = -1

# number of saccades per example
ntime = 10

## prepare data

# load and resize image
img = Image.open('../data/lena.png')  # image extension *.png, *.jpg
height = 100
hpercent = (height / float(img.size[1]))
width = int((float(img.size[0]) * float(hpercent)))
img = img.resize((width, height), Image.ANTIALIAS)

if False:  # color
    img = np.array(img)
else:  # grayscale
    img = np.expand_dims(np.array(img.convert('L')), axis=2)

# normalize image to [0, 1]
img = img / 255.0

# data is prepared as [examples, channels. width, height]
data = np.repeat(np.expand_dims(np.moveaxis(img, -1, 0).astype(np.float32), axis=0), repeats=ntime, axis=0)

source = DataSource(data, ntime=ntime, batch_size=1)

## Train model

model = APCModel(nhidden=10, nout=source.data.shape[1])

model.train(source, nepochs=1000)