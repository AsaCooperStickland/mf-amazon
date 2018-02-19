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

data_dir = '../data/ml/'

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
    with open('../data/ml/ratings.dat', 'r') as f:
        num_user = 0
        num_item = 0
        num_rating = 0
        for line in f.readlines():
            tokens = line.strip().split('::')
            #print(tokens)
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
            day = time // 3600 // 24
            data_list.append((user_id, item_id, time, rating, day))
            num_rating += 1

    print('Num users: {}, num items: {}, num ratings: {}'.format(num_user,
                                                                 num_item,
                                                                 num_rating))

    return data_list, num_rating


def split_data(data_list, num_rating):
    data_list = sorted(data_list, key=get_key)
<<<<<<< HEAD
=======
    '''for i, ex in enumerate(data_list):
        if i % 10000 == 0:
            time = ex[2]
            print(i)
            print(time)
            print(time // 3600 // 24 // 365 - 28 + 1997)'''

>>>>>>> 98b0610b2b9d4fe4ffb470fc28443e3555965971
    train_list = data_list[:int(0.9*num_rating)]
    test = data_list[int(0.9*num_rating):]
    val_list, test_list = test[:int(0.5*len(test))], test[int(0.5*len(test)):]
    return train_list, val_list, test_list, test


def process_for_rnn(data_list):
    labels = ['UserId', 'ItemId', 'time', 'rating', 'day']
    data = pd.DataFrame.from_records(data_list, columns=labels)
    new_list = []
    for UserID, data_point in data.groupby('UserId'):
        for day, day_data in data_point.groupby('day'):
            item_list = day_data['ItemId'].tolist()
            ratings_list = day_data['rating'].tolist()
            for i in range(len(item_list)):
                new_list.append((UserID, item_list[:i], ratings_list[:i], item_list[i], ratings_list[i], day))
    return new_list


def write_to_pd(data_list, out_file):
    """Reads the tokenized .story files corresponding to the urls listed in the url_file and writes them to a out_file."""
    labels = ['UserId', 'ItemList', 'RatingList', 'item', 'rating', 'day']
    data = pd.DataFrame.from_records(data_list, columns=labels)

    with open(data_dir + out_file + '.pkl', 'wb') as f:
        pickle.dump(data, f)
    print("Finished writing files %s\n" % out_file)


def write_to_list(data_list, out_file):
    """Reads the tokenized .story files corresponding to the urls listed in the url_file and writes them to a out_file."""

    with open(data_dir + out_file + '.pkl', 'wb') as f:
        pickle.dump(data_list, f)
    print("Finished writing files %s\n" % out_file)


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def write_to_bin(data_list, out_file):
    """Reads the tokenized .story files corresponding to the urls listed in the url_file and writes them to a out_file."""
    writer = tf.python_io.TFRecordWriter(data_dir + out_file + '.tfrecords')
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


def write_days_to_bin(data_list, out_file):
    """Reads the tokenized .story files corresponding to the urls listed in the url_file and writes them to a out_file."""
    day_dict = {}
    number_dict = {}
    for example in data_list:
        day = str(example[4])
        if day not in day_dict:
            day_dict[day] = [example[:4]]
        else:
            day_dict[day].append(example[:4])
        if day not in number_dict:
            number_dict[day] = 1
        else:
            number_dict[day] += 1
    for day, examples in day_dict.items():
        writer = tf.python_io.TFRecordWriter(data_dir + 'days/' + day + '.tfrecords')
        for idx, example in enumerate(examples):
            example = tf.train.Example(features=tf.train.Features(feature={
                                       'UserId': _int64_feature([example[0]]),
                                       'ItemId': _int64_feature([example[1]]),
                                       'time': _int64_feature([example[2]]),
                                       'rating': _int64_feature([example[3]])}))
            tf_example_str = example.SerializeToString()
            writer.write(tf_example_str)
        writer.close()
        sys.stdout.flush()
    names = [int(name) for name, examples in day_dict.items()]
    names = sorted(names)

    with open(data_dir + out_file + '_filenames.txt', 'w') as writer:
        for name in names:
            string_to_write = '\t'.join([data_dir + 'days/' + out_file +
                                        str(name) + '.tfrecords',
                                        str(number_dict[str(name)])])
            writer.write(string_to_write + '\n')
    print("Finished writing files %s\n" % out_file)


def av_error(train, val):
    user_av = {}
    item_av = {}
    all_ratings = []

    for idx, example in enumerate(train):
        user = example[0]
        item = example[1]
        rating = example[3]
        all_ratings.append(rating)
        if user not in user_av:
            user_av[user] = [rating]
        else:
            user_av[user].append(rating)
        if item not in item_av:
            item_av[item] = [rating]
        else:
            item_av[item].append(rating)
    MSE_user_av = 0.
    MSE_item_av = 0.
    MSE_rating_av = 0.
    mean_rating = sum(all_ratings)/len(all_ratings)
    new_users, new_items = 0, 0
    old_users, old_items = 0, 0
    print('Mean rating {}'.format(mean_rating))
    for idx, example in enumerate(val):
        user = example[0]
        item = example[1]
        rating = example[3]
        if user in user_av:
            MSE_user_av += (sum(user_av[user])/len(user_av[user]) - rating)**2
            old_users += 1
        else:
            MSE_user_av += (mean_rating - rating)**2
            new_users += 1
        if item in item_av:
            MSE_item_av += (sum(item_av[item])/len(item_av[item]) - rating)**2
            old_items += 1
        else:
            MSE_item_av += (mean_rating - rating)**2
            new_items += 1
        MSE_rating_av += (mean_rating - rating)**2
    MSE_user_av /= len(val)
    MSE_item_av /= len(val)
    MSE_rating_av /= len(val)
    print('New users {}% New Items {}%'.format(100*new_users/(new_users + old_users),
                                               100*new_items/(new_items + old_items)))
    print('User av error {} Item av error {}'.format(MSE_user_av**0.5, MSE_item_av**0.5))
    print('Ratings av error {}'.format(MSE_rating_av**0.5))


print('Generatating data...')
all_train_set = []
all_train_list, ratings = gen_data(all_train_set)
train_list, val_list, test_list, test = split_data(all_train_list, ratings)
av_error(train_list, val_list)
<<<<<<< HEAD
print('Writing to tfrecords...')
=======
rnn_train_list = process_for_rnn(train_list)
rnn_val_list = process_for_rnn(val_list)
rnn_test_list = process_for_rnn(test_list)
rnn_test = process_for_rnn(test)
print('Writing to list...')
'''write_to_list(rnn_train_list, 'train_dataset')
write_to_list(rnn_val_list, 'val_dataset')
write_to_list(rnn_test_list, 'test_dataset')
write_to_list(rnn_test, 'all_test_dataset')'''
'''print('Writing to tfrecords...')
>>>>>>> 98b0610b2b9d4fe4ffb470fc28443e3555965971
write_to_bin(train_list, 'train_dataset')
write_to_bin(val_list, 'val_dataset')
write_to_bin(test_list, 'test_dataset')
write_to_bin(test, 'all_test_dataset')
'''print('Writing to day tfrecords...')
write_days_to_bin(train_list, 'train_dataset')
write_days_to_bin(val_list, 'val_dataset')
write_days_to_bin(test_list, 'test_dataset')
write_days_to_bin(test, 'all_test_dataset')'''
