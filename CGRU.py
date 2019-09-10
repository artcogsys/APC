from chainer import Chain
import chainer.functions as F
import chainer.links as L
import chainer.initializers as I
from chainer import Variable
import numpy as np

class CGRU2D(Chain):
    def __init__(self, out_channels, ksize, device=-1):
        super(CGRU2D, self).__init__()
        pad = int((ksize - 1) / 2)

        with self.init_scope():
            self.U_h = L.Convolution2D(out_channels, out_channels, ksize, pad=pad)
            self.U_r = L.Convolution2D(out_channels, out_channels, ksize, pad=pad)
            self.U_z = L.Convolution2D(out_channels, out_channels, ksize, pad=pad)
            self.W_h = L.Convolution2D(None, out_channels, ksize, pad=pad)
            self.W_r = L.Convolution2D(None, out_channels, ksize, pad=pad)
            self.W_z = L.Convolution2D(None, out_channels, ksize, pad=pad)

        self.reset_state()
        if device != -1:
            self.to_gpu(device)

    def forward(self, x):

        h_bar = self.W_h(x)
        z = self.W_z(x)

        if self.h is not None:
            r = F.sigmoid(self.U_r(self.h) + self.W_r(x))
            h_bar += self.U_h(self.h * r)
            z += self.U_z(self.h)

        h_bar = F.tanh(h_bar)
        z = F.sigmoid(z)
        h = h_bar * z

        if self.h is not None:
            h += self.h * (1 - z)

        self.h = h

        return self.h

    def reset_state(self):
        self.h = None

    def to_cpu(self):
        super(CGRU2D, self).to_cpu()
        if self.h is not None:
            self.h.to_cpu()

    def to_gpu(self, device=None):
        super(CGRU2D, self).to_gpu(device)
        if self.h is not None:
            self.h.to_gpu(device)