import sys
sys.path.append('..')
from data import DataSource
from PIL import Image
import numpy as np
from apc import APCModel
import cv2
from chainer import serializers
# import scipy.io as sio
# import time
import tqdm
import matplotlib.pyplot as plt

# device to run model on set to -1 to run on cpu
device = 0
# number of layers
nlayers= 3
# number of epochs
nepochs = 1000
# video source
fwidth = 1280
fheight = 720
height=72
hpercent = height / float(fheight)
width = int((float(fwidth) * float(hpercent)))

#cap = cv2.VideoCapture('../data/Samsung UHD Sample (Nature) [2160p 4k].mp4')
cap = cv2.VideoCapture('../data/samsung-uhd-iceland-(www.uhdsample.com).mkv')

nexamples = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
nexamples = 300

data = np.zeros([nexamples, 1, height, width])
for i in tqdm.trange(nexamples):

    ret, frame = cap.read()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = Image.fromarray(np.uint8(img))
    img = img.resize((width, height), Image.ANTIALIAS)
    img = np.array(img)
    img = img / 255.0
    img = img.astype(np.float32)
    data[i] = np.expand_dims(img, axis=0)

source = DataSource(data, ntime=nexamples, batch_size=1)

## Train model
## Train model

model = APCModel(nhidden=100, nout=1, nlayers=nlayers, device=device) # operating on grayscale


L, L_E, MSE_f, MSE_m, L_w_rep = model.train(source, nepochs=nepochs, cutoff=25)
plt.figure(2, figsize=(7,7))
plt.title('MSE of patched vs ' + str(nlayers)+ '-layer model')
plt.plot(np.arange(nepochs),MSE_f)
plt.plot(np.arange(nepochs), MSE_m)
plt.xlabel('number of epochs')
plt.ylabel('MSE')
plt.legend(['patched', 'model'])
plt.show()
plt.figure(3, figsize=(7,7))
plt.title('layer-wise error of ' + str(nlayers)+ '-layer model')
for l in range(nlayers):
    plt.plot(np.arange(nepochs), L_E[:,l], label='layer '+ str(l+1))

plt.xlabel('number of epochs')
plt.ylabel('layer-wise error')
plt.legend()
plt.show()

# plot layer wise representations
plt.subplots(1,nlayers, figsize=(12,5))
plt.suptitle('Layer-wise Representations',)
for l in range(nlayers):
    plt.subplot(1,nlayers, l+1)
    plt.title('layer: ' +str((l+1)))
    plt.imshow(np.reshape(L_w_rep[l], (100,100)),cmap='gray')
serializers.save_npz('models/3l_100u2_model', model)





