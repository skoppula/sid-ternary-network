#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import argparse
import os

from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *
from tensorpack.utils.gpu import get_nr_gpu
from tensorpack.dataflow import dataset

from tensorpack.models.common import layer_register, VariableHolder, rename_get_variable
from tensorpack.utils.argtools import shape2d, shape4d
from tensorpack.utils.develop import log_deprecated

from helpers.custom_layers_with_mult_tracking import *
from tensorpack.tfutils.varreplace import remap_variables
from helpers.ternary import tw_ternarize

import tensorflow as tf
from tensorflow.contrib.layers import variance_scaling_initializer

class ResnetModel(ModelDesc):

    def __init__(self, n):
        super(ResnetModel, self).__init__()
        self.n = n
        self.network_complexity = {'mults':0, 'weights':0}

    def _get_inputs(self):
        return [InputDesc(tf.float32, [None, 32, 32, 3], 'input'),
                InputDesc(tf.int32, [None], 'label')]

    def _build_graph(self, inputs):
        image, label = inputs
        image = image / 128.0 # this is actually just a bit-shift


        old_get_variable = tf.get_variable
        def new_get_variable(name, shape=None, **kwargs):
            v = old_get_variable(name, shape, **kwargs)
            if name is not 'W':
                return v
            else:
                logger.info("Ternarizing weight {}".format(v.op.name))
                return tw_ternarize(v, args.t)
        tf.get_variable = new_get_variable

        def residual(name, l, increase_dim=False, first=False):
            shape = l.get_shape().as_list()
            in_channel = shape[3] # because NHWC

            if increase_dim:
                out_channel = in_channel * 2
                stride1 = 2
            else:
                out_channel = in_channel
                stride1 = 1

            with tf.variable_scope(name) as scope:
                if first:
                    b1 = l
                else:
                    b1 = BNReLU(l)
                c1 = Conv2DWithTrackedMults('conv1', b1, out_channel, stride=stride1, nl=BNReLU)
                c2 = Conv2DWithTrackedMults('conv2', c1, out_channel)
                if increase_dim:
                    l = AvgPooling('pool', l, 2)
                    l = tf.pad(l, [[0, 0], [0, 0], [0, 0], [in_channel // 2, in_channel // 2]])

                l = c2 + l
                return l

        with argscope([Conv2DWithTrackedMults, Conv2D, AvgPooling, BatchNorm, GlobalAvgPooling], data_format='NHWC'), \
                argscope([Conv2DWithTrackedMults, Conv2D], nl=tf.identity, use_bias=False, kernel_shape=3,
                         W_init=variance_scaling_initializer(mode='FAN_OUT')), \
                argscope(Conv2DWithTrackedMults, network_complexity=self.network_complexity):

            l = Conv2DWithTrackedMults('conv0', image, 16, nl=BNReLU)
            l = residual('res1.0', l, first=True)
            for k in range(1, self.n):
                l = residual('res1.{}'.format(k), l)
            # 32,c=16

            l = residual('res2.0', l, increase_dim=True)
            for k in range(1, self.n):
                l = residual('res2.{}'.format(k), l)
            # 16,c=32

            l = residual('res3.0', l, increase_dim=True)
            for k in range(1, self.n):
                l = residual('res3.' + str(k), l)
            l = BNReLU('bnlast', l)
            # 8,c=64
            l = GlobalAvgPooling('gap', l)

            logits = FullyConnectedWithTrackedMults('linear', l, out_dim=10, nl=tf.identity, network_complexity=self.network_complexity)

        tf.get_variable = old_get_variable
        prob = tf.nn.softmax(logits, name='output')

        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')

        wrong = prediction_incorrect(logits, label)
        add_moving_summary(tf.reduce_mean(wrong, name='train_error'))

        # weight decay on all W of fc layers
        wd_w = tf.train.exponential_decay(0.0002, get_global_step_var(), 480000, 0.2, True)
        wd_cost = tf.multiply(wd_w, regularize_cost('.*/W', tf.nn.l2_loss), name='wd_cost')
        add_moving_summary(cost, wd_cost)

        add_param_summary(('.*/W', ['rms', 'histogram']))   # monitor W
        self.cost = tf.add_n([cost, wd_cost], name='cost')
        tf.constant([self.network_complexity['mults']], name='TotalMults')
        tf.constant([self.network_complexity['weights']], name='TotalWeights')
        logger.info("Parameter count: {}".format(self.network_complexity))

    def _get_optimizer(self):
        lr = get_scalar_var('learning_rate', 0.01, summary=True)
        opt = tf.train.MomentumOptimizer(lr, 0.9)
        return opt

def create_dataflow(partition='train'):
    ds = dataset.Cifar10(partition)
    pp_mean = ds.get_per_pixel_mean()
    if partition == 'train':
        augmentors = [
            imgaug.CenterPaste((40, 40)),
            imgaug.RandomCrop((32, 32)),
            imgaug.Flip(horiz=True),
            imgaug.MapImage(lambda x: x - pp_mean),
        ]
    else:
        augmentors = [
            imgaug.MapImage(lambda x: x - pp_mean)
        ]
    ds = AugmentImageComponent(ds, augmentors)
    ds = BatchData(ds, batch_size, remainder=(partition != 'train'))
    if partition == 'train':
        ds = PrefetchData(ds, 3, 2)
    return ds

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='list of GPU(s)', default='2')
    parser.add_argument('--t', help='ternary threshold constant', default=0.05, type=float)
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    logger.auto_set_dir(action='d')
    batch_size = 128; load_ckpt = 'train_log/cifar10_resnet_with_tracking/checkpoint'

    train_dataflow = create_dataflow('train')
    val_dataflow = create_dataflow('test')

    resnet_model = ResnetModel(n=3)
    config = TrainConfig(
        model=resnet_model,
        dataflow=train_dataflow,
        callbacks=[
            ModelSaver(),
            MinSaver('validation_error'),
            InferenceRunner(val_dataflow, [ScalarStats('cost'), ClassificationError()]),
            ScheduledHyperParamSetter('learning_rate', [(1, 0.1), (75, 0.01), (100, 0.001), (150, 0.0002)])
        ],
        max_epoch=400,
        nr_tower=max(get_nr_gpu(), 1),
        session_init=SaverRestore(load_ckpt) if load_ckpt else None
    )
    SyncMultiGPUTrainerParameterServer(config).train()
