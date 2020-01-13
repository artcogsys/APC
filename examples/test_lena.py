import sys
sys.path.append('..')
from data import DataSource
from PIL import Image
import numpy as np
from apc import APCModel
import matplotlib.pyplot as plt
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.labelweight'] = 'bold'
#from chainer import serializers

# device to run model on set to -1 if you want to run it on the cpu
device = -1
# number of saccades per example
ntime = 10
# number of epochs
nepochs = 1000

## prepare data
RGB = False # TRUE: color image, FALSE: grayscale image
# load and resize image
img = Image.open('../data/lena.png')  # image extension *.png, *.jpg

img = img.resize((64, 64), Image.ANTIALIAS)

if RGB:  # color
    img = np.array(img)
else:  # grayscale
    img = np.expand_dims(np.array(img.convert('L')), axis=2)

# normalize image to [0, 1]
img = img / 255.0

# data is prepared as [examples, channels. width, height]
data = np.repeat(np.expand_dims(np.moveaxis(img, -1, 0).astype(np.float32), axis=0), repeats=ntime, axis=0)

source = DataSource(data, ntime=ntime, batch_size=1)

## Train model

model = APCModel(nout=32,  nlayers=5, device=device)


L= model.train(source, nepochs=nepochs)

#plt.figure(2, figsize=(7,7))
#plt.title('MSE of patched vs model')
#plt.plot(np.arange(nepochs),MSE_f)
#plt.plot(np.arange(nepochs), MSE_m)
#plt.xlabel('number of epochs')
#plt.ylabel('MSE')
#plt.legend(['patched', 'model'])
#plt.show()

#serializers.save_npz('3l_lena_100u_model', model)