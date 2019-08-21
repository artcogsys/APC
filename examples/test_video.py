from data import DataSource
from PIL import Image
import numpy as np
from apc import APCModel
import cv2
# import scipy.io as sio
# import time
import tqdm

# device to run model on
device = -1

# video source
fwidth = 1280
fheight = 720
height=72
hpercent = height / float(fheight)
width = int((float(fwidth) * float(hpercent)))

cap = cv2.VideoCapture('../data/Samsung UHD Sample (Nature) [2160p 4k].mp4')
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

model = APCModel(nhidden=10, nout=1) # operating on grayscale

model.train(source, nepochs=1000, cutoff=25)

