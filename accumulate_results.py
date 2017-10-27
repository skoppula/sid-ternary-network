#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
import numpy as np
from functools import reduce

d = 'train_log'
dirs = [os.path.join(d, o) for o in os.listdir(d) if os.path.isdir(os.path.join(d,o))]

out = "{:>35} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12}".format('model', 'best eer', 'epoch #', '# mults', '# bin ops', '# params', 'bin sz (kb)', 'energy (uJ)')
print(out)

for d in sorted(dirs):

  if 'sentfilt25' not in d: continue

  f = os.path.join(d, 'log.log')
  vals = []
  with open(f, 'r') as f2:
    lines = f2.readlines()
    total_num_activations = 0
    for line in lines:
        if 'val-utt-error' in line:
            vals.append(float(line.split(': ')[-1].strip()))
        elif 'Parameter count' in line:
            mult = int(float(line.split("mults': ")[1].split(',')[0]))
            param = int(float(line.split("weights': ")[1][:-2]))
        # to deal with dsc nets having two parts in their dsc conv layers
        elif ('output: [' in line and 'Pointwise' not in line) or 'PointwiseConv2D input' in line:
            shape_str = line.split('[None, ')[-1].strip()[:-1]
            dims = [int(x) for x in shape_str.split(', ')]
            prod = reduce((lambda x, y: x * y), dims)
            total_num_activations += prod
  
  dr = d.split('/')[-1]
  is_twn = 'twnTrue' in dr
  if vals:
    energy = 0
    if is_twn:
        bytesize = (float(param))*2
        binops = mult*2
        fl_mults = total_num_activations

        energy += fl_mults*3.7 # mults
        energy += binops*0.1 # adds
        energy += (bytesize/4)*5 # sram loads for weights
        energy += (total_num_activations)*5 # sram activation loads
        energy /= 1e6 # convert from pico to micro joules

        out = "{:>35} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12}".format(dr, min(vals), np.argmin(np.array(vals)), fl_mults, binops, param, bytesize/1000, energy)
        print(out)
    else:
        bytesize = (float(param))*32
        energy += mult*3.7 # mults
        energy += mult*0.9 # adds
        energy += (bytesize/4)*640 # dram loads for weights (shifting out the current set of weights in on-chip cache)
        energy += (total_num_activations)*5 # sram activation loads
        energy /= 1e6 # convert from pico to micro joules

        out = "{:>35} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12}".format(dr, min(vals), np.argmin(np.array(vals)), mult, 0, param, bytesize/1000, energy)
        print(out)
