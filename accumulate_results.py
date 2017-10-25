#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
import numpy as np

d = 'train_log'
dirs = [os.path.join(d, o) for o in os.listdir(d) if os.path.isdir(os.path.join(d,o))]
for d in sorted(dirs):
  f = os.path.join(d, 'log.log')
  vals = []
  with open(f, 'r') as f2:
    lines = f2.readlines()
    for line in lines:
        if 'val-utt-error' in line:
            vals.append(float(line.split(': ')[-1].strip()))
  if vals:
    print(d, min(vals), np.argmin(np.array(vals)))
