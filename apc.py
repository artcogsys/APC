
from chainer import Chain, ChainList
from CGRU import CGRU2D
import chainer.links as L
import matplotlib.pyplot as plt
import numpy as np
import chainer
from chainer import serializers
from chainer.backends import cuda
import tqdm
import chainer.functions as F
from chainer import Variable

DEBUG = False
class APCModel(Chain):

    def __init__(self, nout, nlayers=1, fsize=(15, 15), device=-1):
        """
        :param nout: number of output (image) channels
        :param nlayers: number of layers in the network
        :param fsize: size of the fovea (default (rx=15, ry=15))
        :param device: device id to push to the model on
        """
        super().__init__()
        # check availability of gpu
        if device > -1:
            device =self. _check_gpu(device)
        self.nlayers = nlayers
        
        # identifies device id
        self.device_id = device
       
        # maintain input representations
        self.fovea = None
        self.periphery = None

        # radius of pixels for foveal representation
        self.rx = fsize[0]
        self.ry = fsize[1]

        # blurring using average pooling
        self.blur = 11
       
        with self.init_scope(): 
            self.conv, self.deconv = ChainList(), ChainList()
            # initialize batch norm layers for conv and deconv layers
            self.bn_conv, self.bn_deconv = ChainList(), ChainList()
            for i in range(self.nlayers):
                # specify outsize for deconv layer
                outsize = (64 // (2**(self.nlayers - i - 1)), 64 // (2**(self.nlayers - i - 1)))
                if i == 0: # deconv layer after gru
                    # upsample from 1x1 gru output activations to 4x4 feature maps
                    self.deconv.append(L.Deconvolution2D(in_channels=None,  \
                                                         out_channels=((2**(self.nlayers - 1))*nout),  \
                                                         ksize=4, outsize=outsize))
                    self.bn_deconv.append(L.BatchNormalization((2**(self.nlayers - 1))*nout))
                    self.conv.append(L.Convolution2D(in_channels=None, \
                                                     out_channels=(2**i)*nout, ksize=4, stride=2, pad=int(1)))
                    self.bn_conv.append(L.BatchNormalization((2**i)*nout))
                    
                elif i == self.nlayers - 1: # conv layer before gru & final deconv layer 
                    # downsample to 1x1 feature maps 
                    self.conv.append(L.Convolution2D(in_channels=None, out_channels=(2**i)*nout, ksize=4)) 
                    self.bn_conv.append(L.BatchNormalization((2**i)*nout))
                    # final layer should have 1 output channel, no batch normalization
                    self.deconv.append(L.Deconvolution2D(in_channels=None, out_channels=1, ksize=4, \
                                                          stride=2,  pad=int(1), outsize=outsize)) 
                  
                else:
                    # normal conv & deconv layers: 
                    self.conv.append(L.Convolution2D(in_channels=None, out_channels=(2**i)*nout, ksize=4, stride=2, pad=int(1)))
                    self.bn_conv.append(L.BatchNormalization((2**i)*nout))
                    
                    self.deconv.append(L.Deconvolution2D(in_channels=None,  \
                                                         out_channels=((2**(self.nlayers - 1 - i))*nout),  \
                                                         ksize=4, stride=2, pad=int(1), outsize=outsize))
                    self.bn_deconv.append(L.BatchNormalization(((2**(self.nlayers - 1 - i))*nout)))
            
            # GRU layer, output size will be number of feature maps of the final conv layer
            self.gru = L.GRU(in_size=None, out_size=((self.nlayers - 1)**2)*nout)
               
            
        self.optimizer = chainer.optimizers.Adam()  # other learning rate?
        self.optimizer.setup(self)
        self.optimizer.add_hook(chainer.optimizer_hooks.GradientClipping(5))
        if self.device_id > -1:
            self.to_gpu()
            
        
    def _check_gpu(self, device):
        """
        adapted from: https://www.programcreek.com/python/example/96830/chainer.cuda.check_cuda_available
        

        Returns:
            gpu device id, otherwise -1 (default to cpu)
        """

        try:
            import cupy as cp
            cuda.check_cuda_available()
            cp.cuda.Device(device).use()
            return device
        # if gpu is not available, RuntimeError arises
        except RuntimeError:
            print('Cuda backend and/or cupy not available, defaulting to cpu')
            return -1 
        
    def config_inp(self, x, pos, inp_config):
        """
        Determine what input the network gets
        """
        [nbatch, nchannels, nx, ny] = x.shape
        if inp_config == 'fovea':
            return self.fovea
        elif inp_config =='fandp':
            # add periphery to foveal input exp. cond. 2
            for i in range(nbatch):
                cx = pos[i,0]
                cy = pos[i,1]
                self.fovea[i,:,0:(cx - self.rx), :] = self.periphery[i,:,0:(cx - self.rx), :]
                self.fovea[i,:,(cx + self.rx + 1):-1, :] = self.periphery[i,:,(cx + self.rx + 1):-1, :]
                self.fovea[i,:,:, (cy+self.ry+1):-1] = self.periphery[i,:,:,(cy+self.ry+1):-1]
                self.fovea[i,:,:, 0:(cy - self.ry)] = self.periphery[i,:,:,0:(cy - self.ry)]
            return self.fovea 
        
        elif inp_config =='periphery':
            # exp. cond. 3: set fovea equal to periphery
            self.fovea = self.periphery
            return self.fovea
        elif inp_config == 'full':
            # exp cond. 4: set fovea to full image
            self.fovea = x
            return self.fovea
        
    def forward(self, x, pos, inp_config='fovea'):
        [nbatch, nchannels, nx, ny] = x.shape
        # build up foveal input and foveal mask
        self.fovea = np.zeros([nbatch, nchannels, nx, ny]).astype(np.float32)
        self.mask = np.zeros([nbatch, nchannels, nx, ny]).astype(np.bool)
        for i in range(nbatch):
            cx = pos[i,0]
            cy = pos[i,1]
            self.fovea[i,:,(cx-self.rx):(cx+self.rx+1), (cy-self.ry):(cy+self.ry+1)] = x[i,:,(cx-self.rx):(cx+self.rx+1), (cy-self.ry):(cy+self.ry+1)]
            self.mask[i, :, (cx - self.rx):(cx + self.rx + 1), (cy - self.ry):(cy + self.ry + 1)] = 1
        # build up peripheral input
        self.periphery = F.average_pooling_2d(x.astype(np.float32), ksize=self.blur, pad=int((self.blur-1)/2), stride=1).data
       
        self.fovea = self.config_inp(x, pos, inp_config)
        if self.device_id > -1: # put on gpu if enabled
            self.fovea = cuda.to_gpu(self.fovea)
            self.periphery = cuda.to_gpu(self.periphery)    
        # forward pass over conv layers
        conv = self.fovea
        for i in range(self.nlayers):
            conv = self.bn_conv[i](F.relu(self.conv[i](conv)))
            
        # remove singletons from final conv layer and pass vector to GRU
        # restore singleton dimensions after pass
        gru_out = F.expand_dims(F.expand_dims(self.gru(F.squeeze(conv)), -1),-1)
        
        # pass over deconv layers
        deconv = gru_out
        for i in range(self.nlayers):
            if i == self.nlayers - 1:
                # final deconv layer no batchnorm and sigmoid instead of relu
                deconv = F.sigmoid(self.deconv[i](deconv))
            else:
                deconv = self.bn_deconv[i](F.relu(self.deconv[i](deconv)))
                
        # return reconstruction
        return deconv

    
        
    


    def reset_state(self):
        """
        reset state of latent representation
        """
        self.gru.reset_state()
        

    def sample_position(self, mode='random', nx=None, ny=None, batch_size = None, error=None):
        """
        Deterimine next position to foveate
        """
        if mode == 'random':  # sample at random location
            pos = np.concatenate(
                [np.random.random_integers(self.rx, nx - self.rx, [batch_size, 1]),
                 np.random.random_integers(self.ry, ny - self.ry, [batch_size, 1])],
                axis=1).astype(np.int)

        elif mode == 'center':  # sample at fovea
            pos = np.concatenate([np.round(nx / 2) * np.ones([batch_size, 1]),
                                  np.round(ny / 2) * np.ones([batch_size, 1])],
                                  axis=1).astype(np.int)

        elif mode == 'error':  # error-based sampling
            batch_size = error.shape[0]
            pos = np.zeros([batch_size, 2]).astype(np.int)
            for i in range(batch_size):
                E = error.data[i, self.rx:-self.rx, self.ry:-self.ry]  # take foveal RF size into account
                pos[i] = np.unravel_index(E.argmax(), E.shape)
                pos[i, 0] += self.rx
                pos[i, 1] += self.ry

        return pos
    

    
    def evaluate_patched_model(self, data, predictions, pos_history):
        """
        Evaluates the error-based model against a baseline in
        which foveations are replaced with high resolution patches
        on a mean_image background
        """
        
        # get dimensions of pos history
        nsamples, ntime, nbatch, _ = pos_history.shape
        # get dimensions of frames
        nx, ny = data.shape[3], data.shape[4]
        mse_F, mse_M = 0,0
        for i in range(nsamples):
            for b in range(nbatch):
                # construct mean pixel image
                ground_truth = data[i, b, 0, ...]
                predicted = predictions[i, b, :, :]
                m_image = np.ones((nx,ny)) * np.mean(ground_truth)
                # insert high res foveal patches based on positions
                mse_f, mse_m = 0,0
                for t in range(ntime):
                    pos = pos_history[i,t,b,...]
                    cx = pos[0]
                    cy = pos[1]
                    # set patch to high res
                    m_image[(cx-self.rx):(cx+self.rx+1), (cy-self.ry):(cy+self.ry+1)] = ground_truth[(cx-self.rx):(cx+self.rx+1), (cy-self.ry):(cy+self.ry+1)]
                # compute MSE between ground truth & foveatic patch image & prediction respectively
                mse_f += (np.square(ground_truth - m_image)).mean()
                mse_m += (np.square(ground_truth - predicted)).mean()
            mse_F, mse_M = mse_F + (mse_f/nbatch), mse_M + (mse_m/nbatch)
        return mse_F/nsamples, mse_M/nsamples
    
    def init_graphics(self, shape):
        fig, axes = plt.subplots(2, 3)
        a = axes.flatten()
        for i in range(a.size - 1):
            a[i].axis('off')
        a[0].set_title('ground truth')
        a[1].set_title('fovea')
        a[2].set_title('periphery')
        a[3].set_title('error')
        a[4].set_title('representation')
        a[5].set_title('loss')
       
        x = np.zeros(shape)
        imgs = []
        for i in range(5):
            if x.ndim == 2:
                imgs.append(a[i].imshow(x, cmap='gray', vmin=0.0, vmax=1.0))
            else:
                imgs.append(a[i].imshow(x, cmap ='gray', vmin=0.0, vmax=1.0))
        hl, = a[5].plot([],[]) # loss
        self.a = a
        self.hl = hl
        self.fig = fig
        self.imgs = imgs

    def plot_graphics(self, x, error, z):
        if self.device_id > -1:
            self.fovea = cuda.to_cpu(self.fovea)
            self.periphery = cuda.to_cpu(self.periphery)
            z_plot = cuda.to_cpu(z[0].data)
        else:
            z_plot = z[0].data
        d = []
        d.append(np.squeeze(np.moveaxis(x, 0, -1)))
        d.append(np.squeeze(np.moveaxis(self.fovea[0], 0, -1)))
        d.append(np.squeeze(np.moveaxis(self.periphery[0], 0, -1)))
        d.append(error)
        d.append(np.squeeze(np.moveaxis(z_plot, 0, -1)))
        for i in range(5):
            self.imgs[i].set_data(d[i])
        plt.draw()
        plt.pause(0.001)
        if self.device_id > -1:
            self.fovea = cuda.to_gpu(self.fovea)
            self.periphery = cuda.to_gpu(self.periphery)

    def update_graphics(self, x, y):
        self.hl.set_xdata(x)
        self.hl.set_ydata(y)
        self.a[5].relim()
        self.a[5].autoscale_view()

    def train(self, ds_train, nepochs, ds_val=[], cutoff=None, fname=None, inp_config='fovea'):
        """
        Trial-based training regime. Here we iterate for a fixed number of steps

        :param ds:
        :param nepochs:
        :param cutoff:
        :return:
        """
        # get width & height
        nx, ny = ds_train.data.shape[-2], ds_train.data.shape[-1]
        if cutoff is None:
            cutoff = ds_train.ntime

        #L, MSE_f, MSE_m = np.zeros(nepochs), np.zeros(nepochs), np.zeros(nepochs)
        L_train, L_val = np.zeros(nepochs), np.zeros(nepochs)
        if DEBUG:
            self.init_graphics(shape=np.squeeze(np.moveaxis(ds_train.data[0, :, 0, ...], 0, -1)).shape)
   
        with chainer.using_config('train', True):
            for e in tqdm.trange(nepochs):
                for i, x in enumerate(ds_train):
                    if self.device_id > -1:
                        loss = Variable(cuda.to_gpu(np.array([0.0]).astype(np.float32)))
                    else:
                        loss = Variable(np.array([0.0]).astype(np.float32))

                    # reset model at start of each sequence
                    self.reset_state()

                    # sample initial positions
                    pos = self.sample_position(mode='random', nx=nx, ny=ny, batch_size=ds_train.batch_size)
                    z = [None] * cutoff
                    idx = 0
            
            
                    for t in range(ds_train.ntime):
                        # record position of fovea
                        #pos_history[i, t, :, :] = pos
                        inputs = x[:,:,t,...]
                        [nbatch,nchannels,nx,ny] = inputs.shape
                     
                        # compute predicted high-resolution internal representation from foveal and/or peripheral representations
                        z[idx]= self(inputs, pos, inp_config)
                        z_data = z[idx][0].data
                        z_data = cuda.to_cpu(z_data)
                        
                        # minimize error between past and current represenation
                        if idx > 0:
                            foveal_error =  F.mean(F.absolute_error(z[idx - 1][self.mask], self.fovea[self.mask]))
                            loss += F.mean(foveal_error)


                        # get peripheral error
                        zb = F.average_pooling_2d(z[idx], ksize=self.blur, pad=int((self.blur - 1) / 2), stride=1)
              
                        peripheral_error = F.mean(F.absolute_error(zb, self.periphery), axis=1)
                        #loss += F.mean(peripheral_error)
                        peripheral_error.to_cpu()
                        if self.device_id != -1:
                            loss.to_cpu()
                            L_train[e] += loss.data
                            loss.to_gpu()
                        else:
                            L_train[e] += loss.data

                        if DEBUG:
                            z_idx = z[idx]
                            self.plot_graphics(x[0, :, t, ...], peripheral_error[0].data, z_idx)
                            self.fig.suptitle('epoch ' + str(e) + '/' + str(nepochs) + '; example ' + str(ds_train.batch_idx) +
                                         '/' + str(ds_train.nbatch) + '; timestep ' + str(t + 1) + '/' + str(ds_train.ntime))

                        idx += 1
                       
                        if idx == cutoff - 1:

                            self.cleargrads()
                            loss.backward()
                            loss.unchain_backward()
                            self.optimizer.update()
                           
                            idx = 0
                            z = [None] * cutoff
                            if self.device_id > -1:
                                loss = Variable(cuda.to_gpu(np.array([0.0]).astype(np.float32)))
                            else:
                                loss = Variable(np.array([0.0]).astype(np.float32))
                     
                        # determine new position based on peripheral vision
                        pos = self.sample_position(mode='error', error=peripheral_error)

                # normalize train loss
                L_train[e] /= ds_train.batch_size * ds_train.nbatch
                # compare final representation against fovea baseline
                #MSE_f[e], MSE_m[e] = self.evaluate_patched_model(ds.data, predictions, pos_history)
                if ds_val != []:
                    self.validate(ds_val, L_val, e)


                if DEBUG:
                    self.update_graphics(np.arange(e + 1), L_train[:(e + 1)])
                if e % 5 == 0 and fname != None: # save model every 5 epochs
                    serializers.save_npz('../models/'+fname+'model_' + inp_config+'_'+ str(e), self)

        
        return L_train, L_val 

    def validate(self, ds_val, L_val, e):
        """
        Method that passes over de validation set once
        to determine if the model learns to generalize
        """
        # compute validation error
        nx, ny = ds_val.data.shape[-2], ds_val.data.shape[-1]

        with chainer.using_config('train', False):
            for i, x in enumerate(ds_val):
                if self.device_id > -1:
                    loss = Variable(cuda.to_gpu(np.array([0.0]).astype(np.float32)))
                else:
                    loss = Variable(np.array([0.0]).astype(np.float32))
    
                # reset model at start of each sequence
                self.reset_state()
    
                # sample initial positions
                pos = self.sample_position(mode='random', nx=nx, ny=ny, batch_size=ds_val.batch_size)
                z = [None]*ds_val.ntime
                idx = 0
                
                for t in range(ds_val.ntime):
                    inputs = x[:,:,t,...]
                    [nbatch,nchannels,nx,ny] = inputs.shape
                         
                    # compute predicted high-resolution internal representation from foveal and/or peripheral representations
                    z[idx]= self(inputs, pos)
                    z_data = z[idx][0].data
                    z_data = cuda.to_cpu(z_data)
                     

                    for u in range(idx + 1):
                        foveal_error = F.mean(F.absolute_error(z[u][self.mask], self.fovea[self.mask]))
                        loss += F.mean(foveal_error)
                        
                    # get peripheral error
                    zb = F.average_pooling_2d(z[idx], ksize=self.blur, pad=int((self.blur - 1) / 2), stride=1)
                    peripheral_error = F.mean(F.absolute_error(zb, self.periphery), axis=1)
                    peripheral_error.to_cpu()
                    if self.device_id > -1:
                        loss.to_cpu()
                        L_val[e] += loss.data
                        loss.to_gpu()
                    else:
                        L_val[e] += loss.data
                    idx += 1 
                # determine new position based on peripheral vision
                pos = self.sample_position(mode='error', error=peripheral_error)
        # normalize validation loss
        L_val[e] /= ds_val.batch_size * ds_val.nbatch
        
    def test(self, ds_test, category=None):
        """
        Method that passes over test data once
        """
        # compute test error
        total_loss = 0
        nexamples, nchannels, ntime, nx, ny = ds_test.data.shape
        pred = np.zeros((nexamples, 1, ntime, nx, ny))
        # compute test error
        tradj_list = [] # list of trajectories (category, filenum, [(x,y)])        
        with chainer.using_config('train', False):
            for i, x in enumerate(ds_test):
                pos_vec = [] # trajectory for a single examples
                if self.device_id > -1:
                    loss = Variable(cuda.to_gpu(np.array([0.0]).astype(np.float32)))
                else:
                    loss = Variable(np.array([0.0]).astype(np.float32))
    
                # reset model at start of each sequence
                self.reset_state()
    
                # sample initial positions
                pos = self.sample_position(mode='random', nx=nx, ny=ny, batch_size=ds_test.batch_size)
                pos_vec.append(pos)
                z = [None]*ds_test.ntime
                idx = 0
                
                for t in range(ds_test.ntime):
                    inputs = x[:,:,t,...]
                    [nbatch,nchannels,nx,ny] = inputs.shape
                         
                    # compute predicted high-resolution internal representation from foveal and/or peripheral representations
                    z[idx]= self(inputs, pos)
                    z_data = z[idx][0].data
                    z_data = cuda.to_cpu(z_data)
                    pred[i,:,t,...] = z_data
                    # minimize error between past and current represenation
                    if idx > 0:
                        foveal_error =  F.mean(F.absolute_error(z[idx - 1][self.mask], self.fovea[self.mask]))
                        loss += F.mean(foveal_error)
                    # get peripheral error
                    zb = F.average_pooling_2d(z[idx], ksize=self.blur, pad=int((self.blur - 1) / 2), stride=1)
                    peripheral_error = F.mean(F.absolute_error(zb, self.periphery), axis=1)
                    peripheral_error.to_cpu()
                
                    idx += 1 
                    # determine new position based on peripheral vision
                    pos = self.sample_position(mode='error', error=peripheral_error)
                    pos_vec.append(pos)
                # post process trajectories to account for batches
                pos1, pos2, pos3, pos4 = [], [], [], []
                for p in pos_vec:
                    pos1.append(list(p[0]))
                    pos2.append(list(p[1]))
                    pos3.append(list(p[2]))
                    pos4.append(list(p[3]))
                tradj_list += [(category, pos1), (category, pos2), (category, pos3), (category, pos4)]
            total_loss += loss.data
        return total_loss / (ds_test.nbatch * ds_test.batch_size), tradj_list