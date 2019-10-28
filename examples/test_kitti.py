# -*- coding: utf-8 -*-
import sys
sys.path.append('..')
from data import DataSource
from PIL import Image
import numpy as np
from apc import APCModel

from chainer import serializers
import os
# import scipy.io as sio
# import time

import matplotlib.pyplot as plt
import hickle as hkl
import time
from datetime import datetime
# device to run model on set to -1 to run on cpu
device = -1
nlayers=5
# create folder for current session
session = 'session_'+str(time.time())[-5:-1]
fname = session+'_'+  datetime.now().strftime('%Y%m%d_%H%M')+'/'
#os.makedirs('../models/' + fname)
# training parameters
nepochs=50
batch_size = 1
samples_per_epoch = 50
ntime = 10
DATA_DIR = '../data/kitti_hkl/'

train_file = os.path.join(DATA_DIR, 'X_test.hkl')
val_file = os.path.join(DATA_DIR, 'X_val.hkl')

train_data = hkl.load(train_file)
val_data = hkl.load(val_file)
# reshape data for iterator into (nexamples,nchannels,nx,ny)
nexamples,nx,ny,nchannels = train_data.shape
train_data = np.reshape(train_data, (nexamples,nchannels,nx,ny))
nexamples,nx,ny,nchannels = val_data.shape
val_data = np.reshape(val_data, (nexamples, nchannels,nx,ny))
train_source = DataSource(train_data, ntime=10, batch_size=batch_size)
val_source = DataSource(val_data, ntime=10, batch_size=1)

# set up width height 
width, height = 128,128
nhidden = height // (2*nlayers)
model = APCModel(nhidden=nhidden, nout=32, nlayers=nlayers, device=device) # operating on grayscale
serializers.load_npz('../models/kitti_model_95', model)

L, L_val = model.train(train_source, nepochs=nepochs, ds_val = val_source, cutoff=ntime, fname=fname)
 

