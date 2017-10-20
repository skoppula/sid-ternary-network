#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import argparse
import os

from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *

from tensorpack.models.common import layer_register, VariableHolder, rename_get_variable
from tensorpack.utils.argtools import shape2d, shape4d
from tensorpack.utils.develop import log_deprecated
from tensorpack.tfutils import symbolic_functions as symbf
from tensorflow.contrib.framework import add_model_variable
from tensorpack.tfutils import get_current_tower_context

import tensorflow as tf

@layer_register(log_shape=True)
def FullyConnectedWithTrackedMults(x, out_dim, network_complexity=None,
                   W_init=None, b_init=None,
                   nl=tf.identity, use_bias=True):
    """
    Fully-Connected layer, takes a N>1D tensor and returns a 2D tensor.
    It is an equivalent of `tf.layers.dense` except for naming conventions.

    Args:
        x (tf.Tensor): a tensor to be flattened except for the first dimension.
        out_dim (int): output dimension
        W_init: initializer for W. Defaults to `variance_scaling_initializer`.
        b_init: initializer for b. Defaults to zero.
        nl: a nonlinearity function
        use_bias (bool): whether to use bias.

    Returns:
        tf.Tensor: a NC tensor named ``output`` with attribute `variables`.

    Variable Names:

    * ``W``: weights of shape [in_dim, out_dim]
    * ``b``: bias
    """
    x = symbf.batch_flatten(x)

    if W_init is None:
        W_init = tf.contrib.layers.variance_scaling_initializer()
    if b_init is None:
        b_init = tf.constant_initializer()

    if get_current_tower_context().is_main_training_tower:
        network_complexity['weights'] += out_dim*x.get_shape().as_list()[1]
        network_complexity['mults'] += out_dim*x.get_shape().as_list()[1]
        if use_bias:
            network_complexity['weights'] += out_dim

    W = tf.get_variable('W', (x.get_shape().as_list()[1], out_dim), initializer=W_init)
    if use_bias:
        b = tf.get_variable('b', out_dim, initializer=W_init)

    product = tf.matmul(x, W)

    ret = nl(tf.nn.bias_add(product, b) if use_bias else product, name='output')
    ret.variables = VariableHolder(W=W)
    if use_bias:
        ret.variables.b = b

    return ret

def log2(x):
  numerator = tf.log(x)
  denominator = tf.log(tf.constant(2, dtype=numerator.dtype))
  return numerator / denominator

@layer_register(use_scope=None)
def BNReLUNoAffine(l, name=None):
    l = BatchNorm('bn', l, use_scale=False, use_bias=False)
    l = tf.nn.relu(l, name=name)
    return l


@layer_register(log_shape=True)
def DepthwiseSeparableConvWithTrackedMults(inputs, out_channels, depth_multiplier=0.25, network_complexity=None, downsample=False, nl=tf.identity):
    out_channels = round(out_channels * depth_multiplier)
    filter_shape = [3,3]
    _stride = 2 if downsample else 1

    # tf.nn.relu is default activation; applies activation after batchnorm, but setting normalizer=None
    depthwise_conv = tf.contrib.layers.separable_conv2d(inputs,
                                                  num_outputs=None,
                                                  depth_multiplier=1,
                                                  stride=_stride,
                                                  kernel_size=filter_shape,
                                                  biases_initializer=None,
                                                  normalizer_fn=None,
                                                  activation_fn=nl)

    pointwise_conv = tf.identity(depthwise_conv, name='output')

    if get_current_tower_context().is_main_training_tower:
        in_shape = inputs.get_shape().as_list()
        network_complexity['weights'] += filter_shape[0]*filter_shape[1]*in_shape[-1] # assuming 'NHWC'
        network_complexity['mults'] += in_shape[1]*in_shape[2]*filter_shape[0]*filter_shape[1]*in_shape[-1]

    # network complexity handled in Conv2DWithTrackedMults
    pointwise_conv = Conv2DWithTrackedMults('PointwiseConv2D', depthwise_conv, out_channels, kernel_shape=1, use_bias=False, nl=nl, network_complexity=network_complexity)

    return pointwise_conv


@layer_register(log_shape=True)
def Conv2DWithTrackedMults(x, out_channel, kernel_shape, network_complexity=None, padding='SAME', stride=1, W_init=None, b_init=None, nl=tf.identity, split=1, use_bias=True, data_format='NHWC'):
    """
    2D convolution on 4D inputs.
    Args:
        x (tf.Tensor): a 4D tensor.
            Must have known number of channels, but can have other unknown dimensions.
        out_channel (int): number of output channel.
        kernel_shape: (h, w) tuple or a int.
        stride: (h, w) tuple or a int.
        padding (str): 'valid' or 'same'. Case insensitive.
        split (int): Split channels as used in Alexnet. Defaults to 1 (no split).
        W_init: initializer for W. Defaults to `variance_scaling_initializer`.
        b_init: initializer for b. Defaults to zero.
        nl: a nonlinearity function.
        use_bias (bool): whether to use bias.
    Returns:
        tf.Tensor named ``output`` with attribute `variables`.
    Variable Names:
    * ``W``: weights
    * ``b``: bias
    """
    in_shape = x.get_shape().as_list()
    channel_axis = 3 if data_format == 'NHWC' else 1
    in_channel = in_shape[channel_axis]
    assert in_channel is not None, "[Conv2D] Input cannot have unknown channel!"
    assert in_channel % split == 0
    assert out_channel % split == 0

    kernel_shape = shape2d(kernel_shape)
    padding = padding.upper()
    filter_shape = kernel_shape + [in_channel / split, out_channel]
    stride = shape4d(stride, data_format=data_format)

    if W_init is None:
        W_init = tf.contrib.layers.variance_scaling_initializer()
    if b_init is None:
        b_init = tf.constant_initializer()

    W = tf.get_variable('W', filter_shape, initializer=W_init)
    if get_current_tower_context().is_main_training_tower:
        network_complexity['weights'] += filter_shape[0]*filter_shape[1]*filter_shape[2]*filter_shape[3]

    if use_bias:
        b = tf.get_variable('b', [out_channel], initializer=b_init)
        if get_current_tower_context().is_main_training_tower:
            network_complexity['weights'] += out_channel

    assert split == 1
    xsh = x.get_shape().as_list()
    if get_current_tower_context().is_main_training_tower:
        network_complexity['mults'] += xsh[1]*xsh[2]*filter_shape[0]*filter_shape[1]*filter_shape[2]*filter_shape[3]
    conv = tf.nn.conv2d(x, W, stride, padding, data_format=data_format)

    ret = nl(tf.nn.bias_add(conv, b, data_format=data_format) if use_bias else conv, name='output')
    ret.variables = VariableHolder(W=W)
    if use_bias:
        ret.variables.b = b

    return ret

