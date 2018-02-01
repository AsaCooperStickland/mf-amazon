# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 16:20:12 2015
@author: Balázs Hidasi
"""
import sys
import os
import hashlib
import struct
import subprocess
import collections
import tensorflow as tf

import random
import numpy as np
import pandas as pd
import datetime as dt
import pickle

PATH_TO_ORIGINAL_DATA = './'

gap = np.array([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
# gap = [2, 7, 15, 30, 60,]
# gap.extend( range(90, 4000, 200) )
# gap = np.array(gap)
def get_key(item):
    return item[2]


def proc_time_emb(hist_t, cur_t):
     hist_t = [cur_t - i + 1 for i in hist_t]
     hist_t = [np.sum(i >= gap) for i in hist_t]
     return hist_t


def gen_data(data_list):
    store_users = {}
    store_items = {}
    with open('../data/ratings_Musical_Instruments.csv', 'r') as f:
        num_user = 0
        num_item = 0
        num_rating = 0
        for line in f.readlines():
            tokens = line.strip().split(',')
            if tokens[0] not in store_users:
                store_users[tokens[0]] = num_user
                num_user += 1
            if tokens[1] not in store_items:
                store_items[tokens[1]] = num_item
                num_item += 1
            user_id = store_users[tokens[0]]
            item_id = store_items[tokens[1]]
            rating = int(float(tokens[2]))
            time = int(tokens[3])
            data_list.append((user_id, item_id, time, rating))
            num_rating += 1

    print('Num users: {}, num items: {}, num ratings: {}'.format(num_user,
                                                                 num_item,
                                                                 num_rating))

    return data_list, num_rating


def split_data(data_list, num_rating):
    data_list = sorted(data_list, key=get_key)
    train_list = data_list[:int(0.8*num_rating)]
    test = data_list[int(0.8*num_rating):]
    random.shuffle(test)
    val_list, test_list = test[:int(0.5*len(test))], test[int(0.5*len(test)):]
    val_list, test_list = sorted(val_list, key=get_key), sorted(test_list, key=get_key)
    return train_list, val_list, test_list, test


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def write_to_bin(data_list, out_file):
    """Reads the tokenized .story files corresponding to the urls listed in the url_file and writes them to a out_file."""
    writer = tf.python_io.TFRecordWriter(out_file)
    for idx, example in enumerate(data_list):
        example = tf.train.Example(features=tf.train.Features(feature={
                                   'UserId': _int64_feature([example[0]]),
                                   'ItemId': _int64_feature([example[1]]),
                                   'time': _int64_feature([example[2]]),
                                   'rating': _int64_feature([example[3]])}))
        tf_example_str = example.SerializeToString()
        writer.write(tf_example_str)
    writer.close()
    sys.stdout.flush()
    print("Finished writing file %s\n" % out_file)


print('Generatating data...')
all_train_set = []
all_train_list, ratings = gen_data(all_train_set)
train_list, val_list, test_list, test = split_data(all_train_list, ratings)
print('Writing to tfrecords...')
write_to_bin(train_list, '../data/train_dataset.tfrecords')
write_to_bin(val_list, '../data/val_dataset.tfrecords')
write_to_bin(test_list, '../data/test_dataset.tfrecords')
write_to_bin(test, '../data/all_test_dataset.tfrecords')
