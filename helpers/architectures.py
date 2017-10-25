#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from custom_layers_with_mult_tracking import *
from tensorpack.models.nonlin import *
from tensorpack.tfutils.symbolic_functions import *

def fcn_net(inp):
    # https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/44681.pdf
    # sized so that total number of weights is 400000 is 230
    l = FullyConnectedWithTrackedMults('linear0', inp, out_dim=232, nl=BNReLU)
    l = FullyConnectedWithTrackedMults('linear1', l, out_dim=232, nl=BNReLU)
    l = FullyConnectedWithTrackedMults('linear2', l, out_dim=232, nl=BNReLU)
    l = FullyConnectedWithTrackedMults('linear3', l, out_dim=232, nl=BNReLU)
    return l

def cnn_net(inp):
    # https://pdfs.semanticscholar.org/ef8d/6c4c65a9a227f63f857fcb789db4202f2180.pdf
    l = tf.reshape(inp, (-1, 50, 20, 1))
    l = Conv2DWithTrackedMults('conv0', l, 16, kernel_shape=12, nl=BNReLU)
    l = FullyConnectedWithTrackedMults('linear0', inp, out_dim=256, nl=BNReLU)
    l = FullyConnectedWithTrackedMults('linear1', l, out_dim=256, nl=BNReLU)
    l = FullyConnectedWithTrackedMults('linear2', l, out_dim=256, nl=BNReLU)
    return l

def maxout_net(inp):
    # https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/41939.pdf
    l = FullyConnectedWithTrackedMults('linear0', inp, out_dim=290)
    l = BatchNorm('bn0', l)
    l = Maxout(l, 2)
    l = FullyConnectedWithTrackedMults('linear1', l, out_dim=290)
    l = BatchNorm('bn1', l)
    l = Maxout(l, 2)
    l = FullyConnectedWithTrackedMults('linear2', l, out_dim=290)
    l = BatchNorm('bn2', l)
    l = Maxout(l, 2)
    l = Dropout(l, keep_prob=0.5)
    l = FullyConnectedWithTrackedMults('linear3', l, out_dim=290)
    l = BatchNorm('bn3', l)
    l = Maxout(l, 2)
    l = Dropout(l, keep_prob=0.5)
    return l

def lcn_net(inp):
    # https://pdfs.semanticscholar.org/ef8d/6c4c65a9a227f63f857fcb789db4202f2180.pdf
    inp = tf.reshape(inp, (-1, 50, 20))
    print('here')
    print(inp[:,0:10,0:10])
    with argscope(FullyConnectedWithTrackedMults, out_dim=62, nl=BNReLU):
        # batch flatten within FCWithTrackedMults
        l0a = FullyConnectedWithTrackedMults('linear0a', inp[:,0:10,0:10])
        l0b = FullyConnectedWithTrackedMults('linear0b', inp[:,0:10,0:20])
        l0c = FullyConnectedWithTrackedMults('linear0c', inp[:,10:20,0:10])
        l0d = FullyConnectedWithTrackedMults('linear0d', inp[:,10:20,0:20])
        l0e = FullyConnectedWithTrackedMults('linear0e', inp[:,20:30,0:10])
        l0f = FullyConnectedWithTrackedMults('linear0f', inp[:,20:30,0:20])
        l0g = FullyConnectedWithTrackedMults('linear0g', inp[:,30:40,0:10])
        l0h = FullyConnectedWithTrackedMults('linear0h', inp[:,30:40,0:20])
        l0i = FullyConnectedWithTrackedMults('linear0i', inp[:,40:50,0:10])
        l0j = FullyConnectedWithTrackedMults('linear0j', inp[:,40:50,0:20])
    l0 = tf.concat([l0a, l0b, l0c, l0d, l0e, l0f, l0g, l0h, l0i, l0j], 1)
    l1 = FullyConnectedWithTrackedMults('linear1', l0, out_dim=256, nl=BNReLU)
    l2 = FullyConnectedWithTrackedMults('linear2', l1, out_dim=256, nl=BNReLU)
    l3 = FullyConnectedWithTrackedMults('linear3', l2, out_dim=256, nl=BNReLU)
    return l3

def dsc_net(inp):
    l = tf.reshape(inp, (-1, 50, 20, 1))
    l = DepthwiseSeparableConvWithTrackedMults('conv0', l, 16, nl=BNReLU)
    l = FullyConnectedWithTrackedMults('linear0', inp, out_dim=256, nl=BNReLU)
    l = FullyConnectedWithTrackedMults('linear1', l, out_dim=256, nl=BNReLU)
    l = FullyConnectedWithTrackedMults('linear2', l, out_dim=256, nl=BNReLU)
    return l

def dsc2_net(inp):
    l = tf.reshape(inp, (-1, 50, 20, 1))
    l = DepthwiseSeparableConvWithTrackedMults('conv0', l, 4, nl=BNReLU)
    l = DepthwiseSeparableConvWithTrackedMults('conv1', l, 16, nl=BNReLU, downsample=True)
    l = DepthwiseSeparableConvWithTrackedMults('conv2', l, 32, nl=BNReLU, downsample=True)
    l = FullyConnectedWithTrackedMults('linear0', inp, out_dim=256, nl=BNReLU)
    l = FullyConnectedWithTrackedMults('linear1', l, out_dim=256, nl=BNReLU)
    l = FullyConnectedWithTrackedMults('linear2', l, out_dim=256, nl=BNReLU)
    return l
