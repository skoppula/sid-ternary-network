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

from tensorpack.utils.gpu import get_nr_gpu
from tensorpack.models.common import layer_register, VariableHolder, rename_get_variable
from tensorpack.utils.argtools import shape2d, shape4d
from tensorpack.utils.develop import log_deprecated

from helpers import read_labels
from custom_layers_with_mult_tracking import *

import tensorflow as tf
from tensorflow.contrib.layers import variance_scaling_initializer

flags = tf.app.flags
flags.DEFINE_float('depth_multiplier', 1.0, 'depth multiplier in mobilenet')
flags.DEFINE_string('gpu', None, 'IDs of GPUs, comma seperated: e.g. 2,3')

FLAGS = flags.FLAGS

class MobilenetModel(ModelDesc):

    def __init__(self, depth_multiplier):
        super(MobilenetModel, self).__init__()
        self.network_complexity = {'mults':0, 'weights':0}
        self.updated_complexity = False
        self.depth_multiplier=depth_multiplier

    def _get_inputs(self):
        return [InputDesc(tf.float32, [None, 32, 32, 3], 'input'),
                InputDesc(tf.int32, [None], 'label')]

    def _build_graph(self, inputs):
        image, label = inputs
        image = image / 128.0 # this is actually just a bit-shift

        if not self.updated_complexity:
            complexity = self.network_complexity
            self.updated_complexity = True
        else:
            complexity = None

        with argscope([Conv2DWithTrackedMults, AvgPooling, BatchNorm, GlobalAvgPooling], data_format='NHWC'), \
                argscope([Conv2DWithTrackedMults], nl=tf.identity, use_bias=False, 
                         W_init=variance_scaling_initializer(mode='FAN_OUT')), \
                argscope([DepthwiseSeparableConvWithTrackedMults, Conv2DWithTrackedMults], network_complexity=complexity):
            # BN relu batchnorms then relu
            l = Conv2DWithTrackedMults('conv0', image, 16, kernel_shape=3, nl=BNReLU) 
            l = DepthwiseSeparableConvWithTrackedMults('dsconv0', l, 16, self.depth_multiplier, downsample=False)
            l = DepthwiseSeparableConvWithTrackedMults('dsconv1', l, 64, self.depth_multiplier, downsample=False)
            l = DepthwiseSeparableConvWithTrackedMults('dsconv3', l, 128, self.depth_multiplier, downsample=False)
            l = DepthwiseSeparableConvWithTrackedMults('dsconv4', l, 256, self.depth_multiplier, downsample=True)
            l = DepthwiseSeparableConvWithTrackedMults('dsconv5', l, 256, self.depth_multiplier, downsample=False)
            l = DepthwiseSeparableConvWithTrackedMults('dsconv6', l, 512, self.depth_multiplier, downsample=False)
            l = AvgPooling('avg_pooling', l, [8,8])
            l = tf.contrib.layers.flatten(l)

        logits = FullyConnectedWithTrackedMults('linear', l, out_dim=10, nl=tf.identity, network_complexity=complexity)
        prob = tf.nn.softmax(logits, name='output')

        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')

        wrong = prediction_incorrect(logits, label)
        add_moving_summary(tf.reduce_mean(wrong, name='train_error'))

        # weight decay on all W of fc layers
        wd_w = tf.train.exponential_decay(0.0002, get_global_step_var(), 480000, 0.2, True)
        wd_cost = tf.multiply(wd_w, regularize_cost('.*/W', tf.nn.l2_loss), name='wd_cost')
        add_moving_summary(cost, wd_cost)

        add_param_summary(('.*/W', ['histogram']))   # monitor W
        self.cost = tf.add_n([cost, wd_cost], name='cost')
        tf.constant([self.network_complexity['mults']], name='TotalMults')
        tf.constant([self.network_complexity['weights']], name='TotalWeights')
        logger.info("Parameter count: {}".format(self.network_complexity))

    def _get_optimizer(self):
        lr = get_scalar_var('learning_rate', 0.01, summary=True)
        opt = tf.train.MomentumOptimizer(lr, 0.9)
        return opt

def create_dataflow(partition):
    ds = dataset.Cifar10(partition)
    pp_mean = ds.get_per_pixel_mean()
    isTrain = partition == 'train'
    if isTrain:
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
    ds = BatchData(ds, batch_size, remainder=isTrain)
    if isTrain:
        ds = PrefetchData(ds, 3, 2)
    return ds

if __name__ == '__main__':
    if FLAGS.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    depth_multiplier = FLAGS.depth_multiplier

    logger.set_logger_dir('train_logs/cifar10_mobilenet_{}'.format(depth_multiplier), action='d')
    batch_size = 128; load_ckpt = None

    train_dataflow = create_dataflow('train')
    val_dataflow = create_dataflow('test')

    resnet_model = MobilenetModel(depth_multiplier)
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
