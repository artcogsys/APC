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
device = 0
# number of saccades per example
ntime = 10
# number of epochs
nepochs = 100

## prepare data
RGB = False # TRUE: color image, FALSE: grayscale image
# load and resize image
img = Image.open('../data/lena.png')  # image extension *.png, *.jpg

height = 128 # image height must be power of 2
hpercent = (height / float(img.size[1]))
width = int((float(img.size[0]) * float(hpercent)))
img = img.resize((width, height), Image.ANTIALIAS)
# hidden dimensionality in lstm (power of 2)
nhidden = int(height/4)
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

model = APCModel(nhidden=nhidden, nout=source.data.shape[1], device=device)


L, L_E, MSE_f, MSE_m, L_w_rep = model.train(source, nepochs=nepochs)

#plt.figure(2, figsize=(7,7))
#plt.title('MSE of patched vs ' + str(nlayers)+ '-layer model')
#plt.plot(np.arange(nepochs),MSE_f)
#plt.plot(np.arange(nepochs), MSE_m)
#plt.xlabel('number of epochs')
#plt.ylabel('MSE')
#plt.legend(['patched', 'model'])
#plt.show()
#plt.figure(3, figsize=(7,7))
#plt.title('layer-wise rep of ' + str(nlayers)+ '-layer model')
#for l in range(nlayers):
#    plt.plot(np.arange(nepochs), L_E[:,l], label='layer '+ str(l+1))
#
#plt.xlabel('number of epochs')
#plt.ylabel('layer-wise error')
#plt.legend()
#plt.show()
## plot layer wise representations
#plt.subplots(1,nlayers, figsize=(12,5))
#plt.suptitle('Layer-wise Representations',)
#for l in range(nlayers):
#    plt.subplot(1,nlayers, l+1)
#    plt.title('layer: ' +str((l+1)))
#    if RGB:
#        plt.imshow(np.reshape(L_w_rep[l], (width,height,3)), interpolation='nearest')
#    else:
#        plt.imshow(np.reshape(L_w_rep[l], (width,height)),cmap='gray')
#serializers.save_npz('3l_lena_100u_model', model)