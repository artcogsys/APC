from chainer import Chain, ChainList
from CGRU import CGRU2D
import chainer.links as L
import matplotlib.pyplot as plt
import numpy as np
import chainer
from chainer.backends import cuda
import cupy as cp
import tqdm
import chainer.functions as F
from chainer import Variable

DEBUG = True

class APCModel(Chain):

    def __init__(self, nhidden, nout, nlayers=1, device=-1):
        """
        :param nhidden: number of hidden channels
        :param nout: number of output (image) channels
        :param nlayers: number of layers for predictive coding model
        """
        super().__init__()

        self.nlayers = nlayers
        # check if model should be run on gpu
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

            self.R = ChainList()
            self.Ahat = ChainList()
            self.E = ChainList()
            for l in range(self.nlayers):
                self.R.append(CGRU2D(out_channels=nhidden, ksize=1, device=self.gpu))
                self.Ahat.append(L.Deconvolution2D(in_channels=None, out_channels=nout, ksize=3, pad=int((3 - 1) / 2)))
                self.E.append(L.Convolution2D(in_channels=None, out_channels=nout, ksize=3, pad=int((3 - 1) / 2)))
               
                  

        self.optimizer = chainer.optimizers.Adam()  # other learning rate?
        self.optimizer.setup(self)
        self.optimizer.add_hook(chainer.optimizer_hooks.GradientClipping(5))
        if self.gpu != -1:
            self.to_gpu()

    def forward(self, x, pos, E, R):
        """

        :param x:
        :param pos: (nbatch, 2) with (x,y) positions ranging from 0 to 1
        :return:
        """

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
        
        # Intialize Error populations, representations and predictions
        Ahat = [None] * self.nlayers
        A = [None] * self.nlayers
        
        # Set first layer input to actual input

       
        for l in range(self.nlayers):
            if self.gpu != -1:
                A[l] = cuda.to_gpu(np.zeros([nbatch,nchannels,int(nx/(2**l)), int(ny/(2**l))]).astype(np.float32))
                Ahat[l] = cuda.to_gpu(np.zeros([nbatch,nchannels,int(nx/(2**l)), int(ny/(2**l))]).astype(np.float32))
            else:
                A[l] = np.zeros([nbatch,nchannels,int(nx/(2**l)), int(ny/(2**l))]).astype(np.float32)
                Ahat[l] = np.zeros([nbatch,nchannels,int(nx/(2**l)), int(ny/(2**l))]).astype(np.float32)
                
        #A[0] = np.concatenate([self.fovea, self.periphery], axis=1)
        A[0] = self.fovea
        [batch,channels,x,y] = A[0].shape 
    
        if self.gpu != -1: # put on gpu if enabled
            self.fovea = cuda.to_gpu(self.fovea)
            A[0] = cuda.to_gpu(A[0])

        # only use fovea as sensory input
        if self.nlayers == 1: #directly return prediction
            res = F.clipped_relu(self.Ahat[0](F.relu(self.R[0](self.fovea))), 1.0)
            return res, E, R

        # top-down pass
        for l in reversed(range(self.nlayers)):
            if l == self.nlayers - 1:
                R[l] = self.R[l](E[l])
                #R[l] = self.R[l](A[l])
            elif l == 0:
                upR = F.unpooling_2d(R[l+1],ksize=2,stride=2, cover_all=False)
                R[l] = self.R[l](F.concat((self.fovea,upR),axis=1))
            else:
                upR = F.unpooling_2d(R[l+1],ksize=2,stride=2, cover_all=False)
                R[l] = self.R[l](F.concat((E[l], upR),axis=1))
                #R[l] = self.R[l](F.concat(A[l], upR))
        
        # bottom-up pass 
        for l in range(self.nlayers):
            if l == 0:
                Ahat[l] = F.clipped_relu(self.Ahat[l](F.relu(R[l])), 1.0)
            else:
                Ahat[l] = self.Ahat[l](R[l])
            E[l] = F.concat((F.relu(Ahat[l] - A[l]), F.relu(A[l] - Ahat[l])),axis=1)
            #E[l] = F.absolute_error(Ahat[l], A[l])
            if l < self.nlayers - 1:
                A[l+1] = F.max_pooling_2d(self.E[l](E[l]), ksize=2, stride=2)
                #A[l+1] = F.max_pooling_2d(Ahat[l], ksize=2, stride=2)        
                
        # return prediction at lowest level
        return Ahat[0], E, R
        

    def reset_state(self):
        [x.reset_state() for x in self.R]

    def sample_position(self, mode='random', nx=None, ny=None, batch_size = None, error=None):

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
        if self.gpu != None:
            self.fovea = cuda.to_cpu(self.fovea)
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
        self.fovea = cuda.to_gpu(self.fovea)

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

        if DEBUG:
            self.init_graphics(shape=np.squeeze(np.moveaxis(ds.data[0, :, 0, ...], 0, -1)).shape)

        with chainer.using_config('train', True):
            for e in tqdm.trange(nepochs):
                # initialize & reset pos history & record predictions
                pos_history = np.zeros([len(ds.data),ds.ntime, ds.nbatch, 2]).astype(np.int)
                predictions = np.zeros([len(ds.data),ds.nbatch, 1, ds.data.shape[3], ds.data.shape[4]])
                for i, x in enumerate(ds):
                    if self.gpu != -1:
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
                        z_data = z[idx].data
                        z_data = cuda.to_cpu(z_data)
                        # record predictions, result will hold prediction after final saccade of the trial
                        predictions[i,:,:,:,:] = z_data
                        # record layer-wise error
                        for l in range(self.nlayers):
                            errors[l,t] = F.mean(E[l]).data
                            
                        # minimize error between all past (and current) representations and the current foveated patch
                        for u in range(idx + 1):
                            foveal_error = F.mean(F.absolute_error(z[u][self.mask], self.fovea[self.mask]))
                            # only take error from lowest layer (Cox 2016)
                            #foveal_error = F.mean(errors[u][0])
                            loss += F.mean(foveal_error)
                            # integrate layer wise loss 
                            #loss += np.sum(errors[:,u])

                        # get peripheral error
                        zb = F.average_pooling_2d(z[idx], ksize=self.blur, pad=int((self.blur - 1) / 2), stride=1)
                        zb.to_cpu()
                        peripheral_error = F.mean(F.absolute_error(zb, self.periphery), axis=1)
                        # loss += F.mean(peripheral_error)
                        
                        if self.gpu != -1:
                            loss.to_cpu()
                            L[e] += loss.data
                            loss.to_gpu()
                        else:
                            L[e] += loss.data

                        if DEBUG:
                            z_idx = z[idx]
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

        return L, L_E, MSE_f, MSE_m