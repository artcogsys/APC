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

DEBUG = True

class APCModel(Chain):

    def __init__(self, nhidden, nout, nlayers=1, device=-1):
        """
        :param nhidden: number of hidden channels
        :param nout: number of output (image) channels
        :param nlayers: legacy variable
        """
        super().__init__()
        # check availability of gpu
        if device > -1:
            device =self. _check_gpu(device)
        
        self.nlayers = nlayers
        self.nhidden = nhidden # hidden dimensionality
        self.lstm_units = nhidden*nhidden
        self.l_units = (2*nhidden)*(2*nhidden)
        self.channels = nout
        
        # identifies device id
        self.gpu = device
       
        # maintain input representations
        self.fovea = None
        self.periphery = None

        # radius of pixels for foveal representation
        self.rx = 15
        self.ry = 15

        # blurring using average pooling
        self.blur = 11

        with self.init_scope():
            self.conv = L.Convolution2D(in_channels=None, out_channels=nout, ksize=3, pad=int((3-1)/2))
            self.l1 = L.Linear(in_size=None, out_size=self.l_units)
            self.lstm = L.LSTM(in_size=None, out_size=self.lstm_units)
            self.l2 = L.Linear(in_size=None, out_size=self.l_units)
            self.deconv = L.Deconvolution2D(in_channels=None, out_channels=nout, ksize=3, pad=int((3-1)/2))

        self.optimizer = chainer.optimizers.Adam()  # other learning rate?
        self.optimizer.setup(self)
        self.optimizer.add_hook(chainer.optimizer_hooks.GradientClipping(5))
        if self.gpu > -1:
            self.to_gpu()
            
        
    def _check_gpu(self, device):
        """
        Check
        adapted from: https://www.programcreek.com/python/example/96830/chainer.cuda.check_cuda_available
        

        Returns:
            gpu device id, otherwise -1 (default to cpu)
        """

        try:
            cuda.check_cuda_available()
            return device
        # if gpu is not available, RuntimeError arises
        except RuntimeError:
            print('Cuda backend not available, defaulting to cpu')
            return -1 
        
    def forward(self, x, pos, E, R):
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
        
        if self.gpu > -1: # put on gpu if enabled
            self.fovea = cuda.to_gpu(self.fovea)
            self.periphery = cuda.to_gpu(self.periphery)
        # forward pass
        conv = F.relu(self.conv(self.fovea))
        conv = F.max_pooling_2d(conv, ksize=2, stride=2)
        l1 = F.relu(self.l1(conv))
        lstm = self.lstm(l1)
        l2 = F.relu(self.l2(lstm))
        # reshape back into 2d shape
        l2 = F.reshape(l2, (1, self.channels,int(np.sqrt(self.l_units)), int(np.sqrt(self.l_units))))
        l2 = F.unpooling_2d(l2, ksize=2, stride=2, cover_all=False)
        return [self.deconv(l2)], E, R
    


    def reset_state(self):
        """
        reset state of latent representation
        """
        self.lstm.reset_state()

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
    

    
    def evaluate_fovea(self, data, predictions, pos_history):
        """
        TBD
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
                imgs.append(a[i].imshow(x, vmin=0.0, vmax=1.0))
        hl, = a[5].plot([],[]) # loss
        self.a = a
        self.hl = hl
        self.fig = fig
        self.imgs = imgs

    def plot_graphics(self, x, error, z):
        if self.gpu > -1:
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
        if self.gpu > -1:
            self.fovea = cuda.to_gpu(self.fovea)
            self.periphery = cuda.to_gpu(self.periphery)

    def update_graphics(self, x, y,f,m):
        self.hl.set_xdata(x)
        self.hl.set_ydata(y)
        self.a[5].relim()
        self.a[5].autoscale_view()

    def train(self, ds, nepochs, cutoff=None):
        """
        Trial-based training regime. Here we iterate for a fixed number of steps

        :param ds:
        :param nepochs:
        :param cutoff:
        :return:
        """
            
        if cutoff is None:
            cutoff = ds.ntime

        L, L_E, MSE_f, MSE_m = np.zeros(nepochs), np.zeros((nepochs,self.nlayers)), np.zeros(nepochs), np.zeros(nepochs)
        l_w_rep = [None] * self.nlayers # record final layer representations
        if DEBUG:
            self.init_graphics(shape=np.squeeze(np.moveaxis(ds.data[0, :, 0, ...], 0, -1)).shape)

        with chainer.using_config('train', True):
            for e in tqdm.trange(nepochs):
                # initialize & reset pos history & record predictions
                pos_history = np.zeros([len(ds.data),ds.ntime, ds.nbatch, 2]).astype(np.int)
                predictions = np.zeros([len(ds.data),ds.nbatch, ds.data.shape[1], ds.data.shape[3], ds.data.shape[4]])
                for i, x in enumerate(ds):
                    if self.gpu > -1:
                        loss = Variable(cuda.to_gpu(np.array([0.0]).astype(np.float32)))
                    else:
                        loss = Variable(np.array([0.0]).astype(np.float32))

                    # reset model at start of each sequence
                    self.reset_state()

                    # sample initial position
                    pos = self.sample_position(mode='random', nx=ds.data.shape[3], ny=ds.data.shape[4], batch_size=ds.batch_size)
                    z = [None] * cutoff
                    idx = 0
                     # initialize erros and representations
                    E, R = [None] * self.nlayers, [None]*self.nlayers
                    errors = np.zeros((self.nlayers,ds.ntime)) # save errors from layers
                    for t in range(ds.ntime):
                        # record position of fovea
                        pos_history[i, t, :, :] = pos
                        inputs = x[:,:,t,...]
                        [nbatch,nchannels,nx,ny] = inputs.shape
                        # initialize error with apropriate dimensions per layer
                        for l in range(self.nlayers):
                            if self.gpu != -1:
                                E[l] = cuda.to_gpu(np.zeros([nbatch,nchannels,int(nx/(2**l)), int(ny/(2**l))]).astype(np.float32))
                            else:
                                E[l] = np.zeros([nbatch,nchannels,int(nx/(2**l)), int(ny/(2**l))]).astype(np.float32)
                        # compute predicted high-resolution internal representation from foveal and/or peripheral representations
                        z[idx], E, R = self(inputs, pos, E, R)
                        z_data = z[idx][0].data
                        z_data = cuda.to_cpu(z_data)
                        #if e == nepochs - 1:
                            #for l in range(self.nlayers):
                                #l_w_rep[l] = cuda.to_cpu(z[idx][l].data)
                        # record predictions, result will hold prediction after final saccade of the trial
                        predictions[i,:,:,:,:] = z_data
                        # record layer-wise error
                        for l in range(self.nlayers):
                            errors[l,t] = F.mean(E[l]).data
                            
                        # minimize error between past and current represenation
                        if idx > 0:
                            foveal_error =  F.mean(F.absolute_error(z[idx - 1][0][self.mask], self.fovea[self.mask]))
                            loss += F.mean(foveal_error)
                        else:
                            foveal_error =  F.mean(F.absolute_error(z[idx][0][self.mask], self.fovea[self.mask]))
                            loss += F.mean(foveal_error)
                    

                        # get peripheral error
                        zb = F.average_pooling_2d(z[idx][0], ksize=self.blur, pad=int((self.blur - 1) / 2), stride=1)
                        #zb.to_cpu()
                        peripheral_error = F.mean(F.absolute_error(zb, self.periphery), axis=1)
                        #loss += F.mean(peripheral_error)
                        peripheral_error.to_cpu()
                        if self.gpu != -1:
                            loss.to_cpu()
                            L[e] += loss.data
                            loss.to_gpu()
                        else:
                            L[e] += loss.data

                        if DEBUG:
                            z_idx = z[idx][0]
                            self.plot_graphics(x[0, :, t, ...], peripheral_error[0].data, z_idx)
                            self.fig.suptitle('epoch ' + str(e) + '/' + str(nepochs) + '; example ' + str(ds.batch_idx) +
                                         '/' + str(ds.nbatch) + '; timestep ' + str(t + 1) + '/' + str(ds.ntime))

                        idx += 1
                       
                        if idx == cutoff:

                            self.cleargrads()
                            loss.backward()
                            loss.unchain_backward()
                            self.optimizer.update()
                           
                            idx = 0
                            z = [None] * cutoff
                            if self.gpu != -1:
                                loss = Variable(cuda.to_gpu(np.array([0.0]).astype(np.float32)))
                            else:
                                loss = Variable(np.array([0.0]).astype(np.float32))
                     
                        # determine new position based on peripheral vision
                        pos = self.sample_position(mode='error', error=peripheral_error)
                        

                L[e] /= ds.batch_size * ds.nbatch
                # compare final representation against fovea baseline
                MSE_f[e], MSE_m[e] = self.evaluate_fovea(ds.data, predictions, pos_history)
                for l in range(self.nlayers):
                    L_E[e,l] = np.sum(errors[l,:])
                if DEBUG:
                    self.update_graphics(np.arange(e + 1), L[:(e + 1)], MSE_f[:(e + 1)], MSE_m[:(e+1)])
                    #serializers.save_npz('../models/ed_car_model', self)


        return L, L_E, MSE_f, MSE_m, l_w_rep