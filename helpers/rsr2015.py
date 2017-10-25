#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: rsr2015.py
# Author: Skanda Koppula  <skanda.koppula@gmail.com>
import os
import random
import pickle
import numpy as np
import hashlib
from ast import literal_eval as make_tuple

from tensorpack.dataflow import RNGDataFlow

__all__ = ['WholeUtteranceAsFrameRsr2015', 'RandomFramesBatchRsr2015', 'WholeUtteranceAsBatchRsr2015', 'RsrMfccFiles', 'RandomFramesBatchFromCacheRsr2015', 'get_n_spks']

# /data/sls/scratch/skoppula/mfcc-nns/rsr-experiments/create_rsr_data_cache/generator/spk_mappings.pickle
# /data/sls/scratch/skoppula/mfcc-nns/rsr-experiments/dorefa/train_cache/rsr_smlspk_mappings.pickle
def get_n_spks(save_path):
    if os.path.isfile(save_path):
        with open(save_path, "rb") as f:
            mapping = pickle.load(f)
    return len(mapping)

def create_label_mapping(labels, save_path='generator/spk_mappings.pickle'):
    if os.path.isfile(save_path):
        with open(save_path, "rb") as f:
            mapping = pickle.load(f)
    else:
        print("Loading mapping...")
        mapping = {label: i for i, label in enumerate(set(labels))}
        with open(save_path, "wb") as f:
            pickle.dump(mapping, f)
    mapped_labels = np.array([mapping[label] for label in labels], dtype='int32')
    return mapping, mapped_labels

def get_shapes(base_dir, partition):
    shapes_path = os.path.join(base_dir, partition + '.shapes')
    assert os.path.isfile(shapes_path)
    print(shapes_path)
    with open(shapes_path, 'r') as f:
        lines = f.readlines()
    return [make_tuple(line.strip().split('.npy ')[-1]) for line in lines]

class RsrMfccFiles(RNGDataFlow):
    """
    Expects a $partition.idx file inside the base_dir fodler
    which contains the path to each example file
    """
    def __init__(self, base_dir, partition, spk_mappings_path, shuffle=None, sentfilt=None):
        assert partition in ['train', 'test', 'val']
        assert os.path.isdir(base_dir)
        self.base_dir = base_dir
        self.partition = partition
        self.index = os.path.join(base_dir, partition + '.idx')
        self.sentfilt = sentfilt
        assert os.path.isfile(self.index)

        if shuffle is None:
            shuffle = name == 'train'
        self.shuffle = shuffle

        with open(self.index, 'r') as f:
            lines = f.readlines()

        self.labels = []; self.files = []
        for line in lines:
            label = line.split()[0].strip()
            fyle = line.split()[1].strip()
            sent_id = int(fyle.split('/')[-1].split('-')[-2].split('_')[1])
            if sentfilt is not None and sentfilt != sent_id:
                continue
            self.labels.append(label)
            self.files.append(fyle)

        self.mapping, self.mapped_labels = create_label_mapping(self.labels, spk_mappings_path)

    def size(self):
        return len(self.files)

    def get_data(self):
        idxs = np.arange(len(self.files))
        if self.shuffle:
            self.rng.shuffle(idxs)
        for i in idxs:
            fname, label = self.files[i], self.mapped_labels[i]
            fname = os.path.join(self.base_dir, fname)
            yield [fname, label]

#############################################################
# For shuffled frame batches from pre-made old-formatted cache
#############################################################

class RandomFramesBatchFromCacheRsr2015(RsrMfccFiles):
    def __init__(self, cache_dir):
        self.cache_base_dir = cache_dir
        all_files = os.listdir(self.cache_base_dir)
        self.batch_x_paths = sorted([os.path.join(self.cache_base_dir, f) for f in all_files if 'txt' not in f and '_x' in f])
        self.batch_y_paths = sorted([os.path.join(self.cache_base_dir, f) for f in all_files if 'txt' not in f and '_y' in f])
        self.num_batches_in_epoch = len(self.batch_x_paths)
        for i, path in enumerate(self.batch_x_paths):
            x_num = os.path.basename(path).split('_')[1]
            y_num = os.path.basename(self.batch_y_paths[i]).split('_')[1]
            assert x_num == y_num

        if not os.path.exists(self.cache_base_dir):
            print("Creating", self.cache_base_dir)
            os.makedirs(self.cache_base_dir)

    def get_n_spks(self):
        return len(self.mapping)

    def get_data(self):
        idxs = np.arange(len(self.batch_x_paths))
        random.shuffle(idxs)
        for i, idx in enumerate(idxs):
            cached_path_x = self.batch_x_paths[i]
            cached_path_y = self.batch_y_paths[i]

            if os.path.isfile(cached_path_x) and os.path.isfile(cached_path_y):
                batch_x = np.load(cached_path_x)
                batch_y = np.load(cached_path_y)
                # old cache format stores the y-label as a one hot vector instead of number
                batch_y = np.argmax(batch_y, axis=1)
            else:
                print("Couldn't find batch:", cached_path_x, cached_path_y)
                continue

            yield [batch_x, batch_y]


    def size(self):
        return self.num_batches_in_epoch


#############################################################
# For new shuffled frame batches
#############################################################

class RandomFramesBatchRsr2015(RsrMfccFiles):
    def __init__(self, base_dir, partition, batch_size, context=20, n_mfccs=20):
        super(RandomFramesBatchRsr2015, self).__init__(base_dir, partition, True)
        self.shapes = get_shapes(base_dir, partition)
        self.num_batches_in_epoch = len(self.shapes)
        self.context = context
        self.mfcc_size = n_mfccs
        self.batch_size = batch_size
        self.partition = partition
        self.cache_base_dir = os.path.join('..', self.partition + '_cache')
            
        if not os.path.exists(self.cache_base_dir):
            print("Creating", self.cache_base_dir)
            os.makedirs(self.cache_base_dir)
        assert context > 0

    def get_data(self):
        idxs = range(0, self.batch_size*self.num_batches_in_epoch, self.batch_size)
        random.shuffle(idxs)
        for i, idx in enumerate(idxs):
            cached_path_x = os.path.join(self.cache_base_dir, 'batch_' + str(idx) + '_x.npy')
            cached_path_y = os.path.join(self.cache_base_dir, 'batch_' + str(idx) + '_y.npy')

            if os.path.isfile(cached_path_x) and os.path.isfile(cached_path_y):
                batch_x = np.load(cached_path_x)
                batch_y = np.load(cached_path_y)
            else:
                print("Creating batch", i)
                batch_x = np.zeros((self.batch_size, self.context*self.mfcc_size))
                batch_y = np.zeros((self.batch_size,))

                for j in range(self.batch_size):
                    file_out = next(super(RandomFramesBatchRsr2015, self).get_data())
                    fname, label = file_out[0], file_out[1]
                    utt_data = np.load(fname)[:,0:self.mfcc_size]
                    i = random.randint(0,utt_data.shape[0] - self.context)
                    batch_x[j] = utt_data[i:i+self.context,:].reshape(1, self.context*self.mfcc_size)
                    batch_y[j] = label

                    np.save(cached_path_x, batch_x)
                    np.save(cached_path_y, batch_y)

            yield [batch_x, batch_y]


    def size(self):
        return self.num_batches_in_epoch

#############################################################
# For nonshuffled frames
############################################################

class WholeUtteranceAsFrameRsr2015(RsrMfccFiles):
    """
    Produces MFCC frames of size [context*mfcc_size], and corresponding
    numeric label based on mapping create from RsrMfccFiles. mfcc_size
    is n_mfccs=20 if not including double deltas otherwise n_mfccs*3
    (stacked mfcc, deltas, and double deltas, each of size n_mfccs)

    FEEDS CONSECUTIVE FRAMES IN AN UTTERANCE

    $partition.shapes must be exist and describe the shapes of each MFCC
    utterance matrix. It is recommended (but not necessary) to  
    follow the same ordering of utterances as the index
    """
    def __init__(self, base_dir, partition, context=20, n_mfccs=20, include_dd=False, shuffle=None):
        super(WholeUtteranceAsFrameRsr2015, self).__init__(base_dir, partition, shuffle)
        self.shapes = get_shapes(base_dir, partition)
        self.num_examples_in_epoch = sum([abs(x[0] - context) for x in self.shapes])
        self.context = context
        self.mfcc_size = n_mfccs*3 if include_dd else n_mfccs
        assert context > 0

    def get_data(self):
        for fname, label in super(WholeUtteranceAsFrameRsr2015, self).get_data():
            utt_data = np.load(fname)[:,0:self.mfcc_size]
            # otherwise, we feed in utterance after utterance, one per batch
            for i in range(utt_data.shape[0] - self.context):
                yield [utt_data[i:(i+self.context),:].flatten(), label]

    def size(self):
        return self.num_examples_in_epoch

#############################################################
# For nonshuffled frames AS A BATCH
############################################################

# magic little function that gets [width] number of 
# consecutive sliding windows. Based on
# https://stackoverflow.com/questions/15722324/sliding-window-in-numpy
def window_stack(a, stepsize=1, width=3):
    return np.hstack(a[i:1+i-width or None:stepsize] for i in range(0,width))

class WholeUtteranceAsBatchRsr2015(RsrMfccFiles):
    """
    Produces MFCC frames of size [context*mfcc_size], and corresponding
    numeric label based on mapping create from RsrMfccFiles. mfcc_size
    is n_mfccs=20 if not including double deltas otherwise n_mfccs*3
    (stacked mfcc, deltas, and double deltas, each of size n_mfccs)

    FEEDS CONSECUTIVE FRAMES IN AN UTTERANCE

    $partition.shapes must be exist and describe the shapes of each MFCC
    utterance matrix. It is recommended (but not necessary) to  
    follow the same ordering of utterances as the index
    """
    def __init__(self, base_dir, partition, spk_map_path, context=20, n_mfccs=20, include_dd=False, shuffle=None, sentfilt=None):
        super(WholeUtteranceAsBatchRsr2015, self).__init__(base_dir, partition, spk_map_path, shuffle, sentfilt)
        self.num_examples_in_epoch = len(self.files)
        self.context = context
        self.mfcc_size = n_mfccs*3 if include_dd else n_mfccs
        assert context > 0

    def get_data(self):
        for fname, label in super(WholeUtteranceAsBatchRsr2015, self).get_data():
            utt_data = np.load(fname)[:,0:self.mfcc_size]
            if utt_data.shape[0] <= self.context: continue
            # otherwise, we feed in utterance after utterance, one per batch
            out = window_stack(utt_data, stepsize=1, width=self.context)
            labels = np.array([label]*out.shape[0])
            yield [out, labels]

    def size(self):
        return self.num_examples_in_epoch


if __name__ == '__main__':
    ds = Rsr2015('./fake_data/', 'train', shuffle=False)
    ds.reset_state()

    for k in ds.get_data():
        from IPython import embed
        embed()
        break
