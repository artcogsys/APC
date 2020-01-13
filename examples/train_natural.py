# -*- coding: utf-8 -*-

import sys
sys.path.append('..')
import numpy as np
import os
from apc import APCModel
from data import DataSource
from datetime import datetime
from chainer import serializers
import matplotlib.pyplot as plt
import glob
from PIL import Image
DATA_DIR = '../data'

def preprocess(img, ntime):
    # preprocess  and normalize image
    # return (ntime,nchannels nx, ny)
    img = img.resize((64,64), Image.ANTIALIAS)
    img = np.expand_dims(np.array(img.convert('L')), axis=2).astype(np.float32)
    img /= img.max()
    return np.repeat(np.expand_dims(np.reshape(img, (1, 64, 64)), axis=0), repeats=ntime, axis=0)

session = 'input_cond_scenes'
fname = session+'_'+  datetime.now().strftime('%Y%m%d_%H%M')+'/'
path = '../models/' + fname
if not os.path.exists(path):
    os.makedirs(path)

device = 0
batch_size = 4
ntime = 20
nlayers=5
nepochs = 100
train_path = os.path.join(DATA_DIR, 'natural_scenes/')
filelist = glob.glob(os.path.join(train_path, '*.jpg'))
train = np.swapaxes(np.array([preprocess(Image.open(fname), ntime) for fname in filelist]).astype(np.float32),0, 1)

train_source = DataSource(train, ntime=ntime, batch_size=batch_size)  

val_path = os.path.join(DATA_DIR, 'natural_scenes_val/')
filelist = glob.glob(os.path.join(val_path, '*.jpg'))
val = np.swapaxes(np.array([preprocess(Image.open(fname), ntime) for fname in filelist]).astype(np.float32),0, 1)

val_source = DataSource(val, ntime=ntime, batch_size=batch_size)
# set up different training conditions (input)
args = ['fovea', 'fandp', 'periphery','full']
train_losses, val_losses = [], []
final_train_loss, final_val_loss = [], []
for arg in args:
    model = APCModel(nout=32, nlayers=nlayers, fsize=(7, 7), device=device) 
    L, L_val = model.train(train_source, nepochs=nepochs, ds_val = val_source, cutoff=ntime, fname=fname, inp_config=arg)
    serializers.save_npz('../models/'+fname+'scene_model_'+str(arg)+'_'+'final', model)
    train_losses.append(L)
    val_losses.append(L_val)
    final_train_loss.append(L[-1])
    final_val_loss.append(L_val[-1])
    
# create and save figure of train loss for each condition
plt.figure(1, figsize=(7,7))
plt.title('Train loss different input conditions')
for i, train_loss in enumerate(train_losses):
    plt.plot(np.arange(nepochs), train_loss, label=args[i])
    plt.xlabel('number of epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('../figures/train_loss_'+str(session)+ '.png')
    
# create and save figure of train loss for each condition
plt.figure(2, figsize=(7,7))
plt.title('Validation loss different input conditions')
for i, val_loss in enumerate(val_losses):
    plt.plot(np.arange(nepochs), val_loss, label=args[i])
    plt.xlabel('number of epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('../figures/val_loss_'+str(session)+ '.png')

# save final train and validation losses to file:
loss_file =open('../figures/tv_losses_'+fname[:-1],'w')

for i, arg in enumerate(args):
    loss_file.write(arg+ ' train_loss: '+str(final_train_loss[i]) + ' ' + 'validation loss: ' + str(final_val_loss[i])+'\n')
loss_file.close()






