from chainer import Chain
from CGRU import CGRU2D
import chainer.links as L
import matplotlib.pyplot as plt
import numpy as np
import chainer
import tqdm
import chainer.functions as F
from chainer import Variable

DEBUG = True

class APCModel(Chain):

    def __init__(self, nhidden, nout):
        """
        :param nhidden: number of hidden channels
        :param nout: number of output (image) channels
        """
        super().__init__()

        # maintain input representations
        self.fovea = None
        self.periphery = None

        # radius of pixels for foveal representation
        self.rx = 15
        self.ry = 15

        # blurring using average pooling
        self.blur = 11

        with self.init_scope():

            self.R = CGRU2D(out_channels=nhidden, ksize=1)
            self.decoder = L.Deconvolution2D(in_channels=None, out_channels=nout, ksize=3, pad=int((3 - 1) / 2))

        self.optimizer = chainer.optimizers.Adam()  # other learning rate?
        self.optimizer.setup(self)
        self.optimizer.add_hook(chainer.optimizer_hooks.GradientClipping(5))

    def forward(self, x, pos):
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

        # use both fovea and periphery
        # return F.clipped_relu(self.decoder(F.relu(self.R(np.concatenate([self.fovea, self.periphery], axis=1)))), 1.0)

        # only use fovea
        return F.clipped_relu(self.decoder(F.relu(self.R(self.fovea))), 1.0)

    def reset_state(self):
        self.R.reset_state()

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
        hl, = a[5].plot([], [])

        self.a = a
        self.hl = hl
        self.fig = fig
        self.imgs = imgs

    def plot_graphics(self, x, error, z):

        d = []
        d.append(np.squeeze(np.moveaxis(x, 0, -1)))
        d.append(np.squeeze(np.moveaxis(self.fovea[0], 0, -1)))
        d.append(np.squeeze(np.moveaxis(self.periphery[0], 0, -1)))
        d.append(error)
        d.append(np.squeeze(np.moveaxis(z[0].data, 0, -1)))
        for i in range(5):
            self.imgs[i].set_data(d[i])
        plt.draw()
        plt.pause(0.001)

    def update_graphics(self, x, y):
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

        L = np.zeros(nepochs)

        if DEBUG:
            self.init_graphics(shape=np.squeeze(np.moveaxis(ds.data[0, :, 0, ...], 0, -1)).shape)

        with chainer.using_config('train', True):

            for e in tqdm.trange(nepochs):

                for x in ds:

                    loss = Variable(np.array([0.0]).astype(np.float32))

                    # reset model at start of each sequence
                    self.reset_state()

                    # sample initial position
                    pos = self.sample_position(mode='random', nx=ds.data.shape[3], ny=ds.data.shape[4], batch_size=ds.batch_size)

                    z = [None] * cutoff

                    idx = 0

                    for t in range(ds.ntime):

                        # compute predicted high-resolution internal representation from foveal and/or peripheral representations
                        z[idx] = self(x[:,:,t,...], pos)

                        # minimize error between all past (and current) representations and the current foveated patch
                        for u in range(idx + 1):
                            foveal_error = F.mean(F.absolute_error(z[u][self.mask], self.fovea[self.mask]))
                            loss += F.mean(foveal_error)

                        # get peripheral error
                        zb = F.average_pooling_2d(z[idx], ksize=self.blur, pad=int((self.blur - 1) / 2), stride=1)
                        peripheral_error = F.mean(F.absolute_error(zb, self.periphery), axis=1)
                        # loss += F.mean(peripheral_error)

                        L[e] += loss.data

                        if DEBUG:
                            self.plot_graphics(x[0, :, t, ...], peripheral_error[0].data, z[idx])
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
                            loss = Variable(np.array([0.0]).astype(np.float32))

                        # determine new position based on peripheral vision
                        pos = self.sample_position(mode='error', error=peripheral_error)

                L[e] /= ds.batch_size * ds.nbatch

                if DEBUG:
                    self.update_graphics(np.arange(e + 1), L[:(e + 1)])

        return L