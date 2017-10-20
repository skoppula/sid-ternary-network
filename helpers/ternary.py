#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf

G = tf.get_default_graph()


def tw_ternarize(x, thre):

    shape = x.get_shape()

    thre_x = tf.stop_gradient(tf.reduce_max(tf.abs(x)) * thre)

    w_p = tf.get_variable('Wp', collections=[tf.GraphKeys.GLOBAL_VARIABLES, 'positives'], initializer=1.0)
    w_n = tf.get_variable('Wn', collections=[tf.GraphKeys.GLOBAL_VARIABLES, 'negatives'], initializer=1.0)

    mask = tf.ones(shape)
    mask_p = tf.where(x > thre_x, tf.ones(shape) * w_p, mask)
    mask_np = tf.where(x < -thre_x, tf.ones(shape) * w_n, mask_p)
    mask_z = tf.where((x < thre_x) & (x > - thre_x), tf.zeros(shape), mask)

    with G.gradient_override_map({"Sign": "Identity", "Mul": "Add"}):
        w =  tf.sign(x) * tf.stop_gradient(mask_z)

    w = w * mask_np
    final_w = tf.identity(w, name='ternarized_W')
    
    t = tf.to_float(tf.reduce_sum(tf.ones(w.get_shape()), name='sparsity'))
    tf.summary.scalar(t.name, tf.to_float(tf.count_nonzero(w))/t)

    return final_w

def p_ternarize(x, p):

    x = tf.tanh(x)
    shape = x.get_shape()

    thre = tf.get_variable('T', trainable=False, collections=[tf.GraphKeys.VARIABLES, 'thresholds'],
            initializer=0.05)
    flat_x = tf.reshape(x, [-1])
    k = int(flat_x.get_shape().dims[0].value * (1 - p))
    topK, _ = tf.nn.top_k(tf.abs(flat_x), k)
    update_thre = thre.assign(topK[-1])
    tf.add_to_collection('update_thre_op', update_thre)

    mask = tf.zeros(shape)
    mask = tf.select((x > thre) | (x < -thre), tf.ones(shape), mask)

    with G.gradient_override_map({"Sign": "Identity", "Mul": "Add"}):
        w =  tf.sign(x) * tf.stop_gradient(mask)

    tf.histogram_summary(w.name, w)
    return w

def tw_ternarize_same_w(x, thre):

    shape = x.get_shape()

    thre_x = tf.stop_gradient(tf.reduce_max(tf.abs(x)) * thre)

    w_pn = tf.get_variable('Wpn', collections=[tf.GraphKeys.GLOBAL_VARIABLES, 'positives'], initializer=1.0)

    tf.summary.scalar(w_pn.name, w_pn)

    mask = tf.ones(shape)
    mask_p = tf.where(x > thre_x, tf.ones(shape) * w_pn, mask)
    mask_np = tf.where(x < -thre_x, tf.ones(shape) * -1 * w_pn, mask_p)
    mask_z = tf.where((x < thre_x) & (x > - thre_x), tf.zeros(shape), mask)

    with G.gradient_override_map({"Sign": "Identity", "Mul": "Add"}):
        w =  tf.sign(x) * tf.stop_gradient(mask_z)

    w = w * mask_np

    tf.summary.histogram(w.name, w)
    return w

