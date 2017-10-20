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
from custom_layers_with_mult_tracking import *

import tensorflow as tf

def update_ema(xn, moving_max, moving_min, decay):
    batch_max = tf.reduce_max(xn, axis=[0,1,2])
    batch_min = tf.reduce_min(xn, axis=[0,1,2])
    update_op1 = moving_averages.assign_moving_average(
        moving_max, batch_max, decay, zero_debias=False,
        name='max_ema_op')
    update_op2 = moving_averages.assign_moving_average(
        moving_min, batch_min, decay, zero_debias=False,
        name='min_ema_op')
    # Only add to model var when we update them
    add_model_variable(moving_min)
    add_model_variable(moving_max)

    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_op1)
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_op2)

    return xn

def log2(x):
  numerator = tf.log(x)
  denominator = tf.log(tf.constant(2, dtype=numerator.dtype))
  return numerator / denominator

@layer_register(log_shape=True)
def RescaleActivationLayer(inputs, decay=0.9, bit_a=8):
    in_shape = inputs.get_shape().as_list()
    moving_max = tf.get_variable('activation_max/EMA', [in_shape[-1]], initializer=tf.constant_initializer(), trainable=False)
    moving_min = tf.get_variable('activation_min/EMA', [in_shape[-1]], initializer=tf.constant_initializer(), trainable=False)

    named_inputs = tf.identity(inputs, name='rescaling_input_activation')
    # xn = (named_inputs - moving_min) / tf.pow(tf.constant(2.0), log2(moving_max) - tf.constant(float(bit_a)))
    xn = (named_inputs - (moving_min + moving_max)/2.0) / (moving_max - moving_min)
    named_xn = tf.identity(xn, name='rescaled_activation')
    named_xn = tf.Print(named_xn, [named_xn])

    ctx = get_current_tower_context()
    if ctx.is_main_training_tower:
        ret = update_ema(xn, moving_max, moving_min, decay)
    else:
        ret = tf.identity(xn, name='output')
    vh = ret.variables = VariableHolder(mean=moving_max, variance=moving_min)
    return ret

@layer_register(log_shape=True)
def PositiveIntegerActivationReLU(inputs, bit_a=8):
    xn = tf.maximum(inputs, tf.pow(tf.constant(2.0), tf.constant(bit_a/2.0)))
    relu_xn = tf.identity(xn, name='positive_relu_activation')
    return relu_xn

@layer_register(log_shape=True)
def RescaleAndPositiveIntegerReLULayer(l, bit_a=8):
    l = RescaleActivationLayer('activation_rescaling', l, decay=0.9, bit_a=8)
    l = PositiveIntegerActivationReLU('relu0', l, bit_a=bit_a)
    return l
