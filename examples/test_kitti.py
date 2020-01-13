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
from matplotlib import gridspec
import hickle as hkl
import time
from datetime import datetime
# device to run model on set to -1 to run on cpu
device = -1
nlayers=2

#os.makedirs('../models/' + fname)
# training parameters
nepochs=10
batch_size = 1
samples_per_epoch = 50
ntime = 10
DATA_DIR = '../data/kitti_hkl/'

test_file = os.path.join(DATA_DIR, 'X_test.hkl')
#val_file = os.path.join(DATA_DIR, 'X_val.hkl')

test_data = hkl.load(test_file)
#val_data = hkl.load(val_file)
# reshape data for iterator into (nexamples,nchannels,nx,ny)
nexamples,nx,ny,nchannels = test_data.shape
test_data = np.reshape(test_data, (nexamples,nchannels,nx,ny))
#nexamples,nx,ny,nchannels = val_data.shape
#val_data = np.reshape(val_data, (nexamples, nchannels,nx,ny))
test_source = DataSource(test_data, ntime=10, batch_size=batch_size)
#val_source = DataSource(val_data, ntime=10, batch_size=1)

# set up width height 
width, height = 128,128
nhidden = height // (2*nlayers)
model = APCModel(nhidden=nhidden, nout=16, nlayers=nlayers, device=device) # operating on grayscale
serializers.load_npz('../models/kitti_model_110', model)
L, L_val = model.train(test_source, nepochs=nepochs, ds_val = [], cutoff=ntime, fname='')

#loss, predicted = model.test(test_source)
# 
#
#print('average sequence test loss = ' + str(loss))
#
#gt_data = test_source.preprocess([0])
#ncol = 10
#nrow = 2
#fig, ax = plt.subplots(nrow, ncol)
#fig.set_size_inches((25,8))
#ax = ax.flatten()
#total = ncol*nrow
#for i in range(total):
#        if i < ncol: 
#            gt_img = gt_data[0,0, i, ...]
#            fig.add_subplot(nrow, ncol, i+1)
#            plt.imshow(gt_img, interpolation="nearest", cmap='gray')
#        else:
#            pred_img = predicted[0,:, i % ncol, ...] 
#            fig.add_subplot(nrow, ncol, i+1)
#            plt.imshow(pred_img[0,...], interpolation="nearest", cmap='gray')
#        ax[i].axis('off')
#
#plt.show()