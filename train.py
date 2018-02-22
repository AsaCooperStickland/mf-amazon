import itertools
import os

import argparse
import tqdm
import pickle
import numpy as np
import tensorflow as tf
from utils.parser_utils import ParserClass
from utils.storage import build_experiment_folder, save_statistics
from sklearn.model_selection import KFold

from input_data import DataInput, DataInputTest
from smf import RNNSMF, SMF, BaseSMF
from rnn import JustRNNSMF

tf.reset_default_graph()  # resets any previous graphs to clear memory
parser = argparse.ArgumentParser(description='Welcome to MF experiments script')  # generates an argument parser
parser_extractor = ParserClass(parser=parser)  # creates a parser class to process the parsed input

rnn, dataset, batch_size, seed, epochs, logs_path, continue_from_epoch,\
    tensorboard_enable, experiment_prefix, day_split, l2_weight, latent_dim,\
    learning_rate, train_fraction = parser_extractor.get_argument_variables()

# No. updates to check val set after
update_number = 20000

if dataset == "ml1m":
    offset = 3.5906066288375316
    num_users = 6040
    num_items = 3706
    num_ratings = 1000209
    data_dir = "/home/s1302760/mf-amazon/data/ml/"
    print("Using movie-lens 1M dataset.")

elif dataset == "amazon":
    offset = 4.240331172610
    num_users = 339231
    num_items = 83046
    num_ratings = 500176
    data_dir = "/home/s1302760/mf-amazon/data/"
    print("Using Amazon datset.")

else:
    raise NameError("Unrecognized dataset!")


experiment_name = "rnn_{}_experiment_{}_batch_size_{}_l2_{}_dim_{}_frac_{}_lr_{}".format(rnn,
                                                                   experiment_prefix,
                                                                   batch_size, l2_weight,
                                                                   latent_dim, train_fraction,
                                                                   learning_rate)
batch_sizes = [256, 512, 1024]
# reg_lambdas = [0, 1e-10, 1e-7, 1e-5]
reg_lambdas = [2, 1e-5, 1e-3, 1e-1, 1]
learning_rates = [0.001, 0.0001]
latent_dims = [50, 100]

print("Running {}".format(experiment_name))
print("Starting from epoch {}".format(continue_from_epoch))

saved_models_filepath, logs_filepath = build_experiment_folder(experiment_name, logs_path)  # generate experiment dir


def val_check(sess, best_val_RMSE_loss):
    total_val_MSE_loss = 0.
    total_val_MAE_loss = 0.
    with tqdm.tqdm(total=total_val_batches) as pbar_val:
        for batch_idx, data in DataInput(val_data, batch_size):
            iter_id = e * total_train_batches + batch_idx
            rmse, mae_val, cost_val, summary_op = smf.eval(sess, data)
            total_val_MSE_loss += cost_val
            total_val_MAE_loss += mae_val
            iter_out = "val_rmse: {}, val_mae: {} val num: {}".format(total_val_MSE_loss / (batch_idx + 1),
                                                          total_val_MAE_loss / (batch_idx + 1),
                                                          total_val_batches * batch_size)
            pbar_val.set_description(iter_out)
            pbar_val.update(1)
    val_size
    total_val_MSE_loss /= val_size
    total_val_RMSE_loss = total_val_MSE_loss**0.5
    total_val_MAE_loss /= total_val_batches

    if best_val_RMSE_loss < total_val_RMSE_loss:  # check if val acc better than the previous best and if
        # so save current as best and save the model as the best validation model to be used on the test set
        #  after the final epoch
        best_val_RMSE_loss = total_val_RMSE_loss
        best_epoch = e
        save_path = val_saver.save(sess,
                                   "{}/best_validation_{}_{}.ckpt".format(saved_models_filepath,
                                                                          experiment_name, e))
        print("Saved best validation score model at", save_path)

    epoch_pbar.update(1)
    # save statistics of this epoch, train and val without test set performance
    save_statistics(logs_filepath, "result_summary_statistics",
                    [e, total_RMSE_loss, total_MAE_loss, total_val_RMSE_loss, total_val_MAE_loss,
                     -1, -1])
    return best_val_RMSE_loss

if rnn:
    smf = JustRNNSMF(num_users, num_items, latent_dim, learning_rate=learning_rate,
             reg_lambda=l2_weight)
else:
    smf = SMF(num_users, num_items, latent_dim, learning_rate=learning_rate,
             reg_lambda=l2_weight)
smf.build_graph(offset)

if continue_from_epoch == -1:  # if this is a new experiment and not
    # continuation of a previous one then generate a new
    # statistics file
    save_statistics(logs_filepath, "result_summary_statistics",
                    ["epoch", "train_c_loss", "train_c_accuracy",
                     "val_c_loss", "val_c_accuracy",
                     "test_c_loss", "test_c_accuracy"], create=True)

start_epoch = continue_from_epoch if continue_from_epoch != -1 else 0  # if new experiment start from 0 otherwise
# continue where left off

if day_split:
    with open(data_dir + 'train_datasetfilenames.txt', 'r') as f:
        filenames = []
        example_counter = []
        for line in f:
            filename, examples = line.strip().split('\t')
            filenames.append(filename)
            example_counter.append(int(examples))
        #filenames = [line.rstrip() for line in f]
        file_number = int((1. - train_fraction) * len(filenames))
        filenames = filenames[file_number:]
        example_counter = example_counter[file_number:]
        train_examples = sum(example_counter)
        print('loading {} examples'.format(train_examples))
else:
    print('loading all examples')
    with open(data_dir + 'train_dataset.pkl', 'rb') as f:
        train_data = pickle.load(f)
    with open(data_dir + 'val_dataset.pkl', 'rb') as f:
        val_data = pickle.load(f)

global_step = tf.Variable(0, name='global_step', trainable=False)
val_saver = tf.train.Saver()

train_size = int(0.9*num_ratings)
val_size = int(0.5*(num_ratings - train_size))

if day_split:
    total_train_batches = int(train_examples/batch_size)
    print('Using {} examples out of total {}'.format(train_examples, train_size))
else:
    total_train_batches = int(train_size/batch_size)
    print('Using {} examples out of total {}'.format(train_size, train_size))
total_val_batches = int(val_size/batch_size)
print('val size', val_size)
print('divide', (total_val_batches * batch_size))
#total_test_batches = = int(int(0.8*num_ratings)/batch_size)

best_epoch = 0

if tensorboard_enable:
    print("saved tensorboard file at", logs_filepath)
    writer = tf.summary.FileWriter(logs_filepath, graph=tf.get_default_graph())

init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())  # initialization op for the graph
with tf.Session() as sess:
    #training_handle = sess.run(init_train.string_handle())
    #val_handle = sess.run(init_val.string_handle())
    sess.run(init_op) # actually running the initialization op
    train_saver = tf.train.Saver()  # saver object that will save our graph so we can reload it later
    val_saver = tf.train.Saver()
    #  training or inference
    continue_from_epoch = -1
    if continue_from_epoch != -1:
        # restore previous graph to continue operations
        train_saver.restore(sess, "{}/{}_{}.ckpt".format(saved_models_filepath, experiment_name,
                                                   continue_from_epoch))
    best_val_RMSE_loss = 0.

    with tqdm.tqdm(total=epochs) as epoch_pbar:
        for e in range(start_epoch, epochs):
            total_RMSE_loss = 0.
            total_MAE_loss = 0.
            with tqdm.tqdm(total=total_train_batches) as pbar_train:
                for batch_idx, data in DataInput(train_data, batch_size):
                    iter_id = e * total_train_batches + batch_idx
                    rmse_train, mae_train, cost, summary_op = smf.train(sess, data, dropout=0.9)
                    # Here we execute u_update, v_update which train the network and also the ops that compute
                    # rmse and mae.
                    total_RMSE_loss += rmse_train
                    total_MAE_loss += mae_train

                    iter_out = "iter_num: {}, train_RMSE: {},"\
                                "train_MAE: {}, batch_RMSE: {}".format(iter_id,
                                                             total_RMSE_loss / (batch_idx + 1),
                                                             total_MAE_loss / (batch_idx + 1),
                                                             rmse_train)
                    pbar_train.set_description(iter_out)
                    pbar_train.update(1)
                    if batch_idx % int(update_number/batch_size) == 0:
                        best_val_RMSE_loss = val_check(sess, best_val_RMSE_loss)
                    if tensorboard_enable and batch_idx % 25 == 0:  # save tensorboard summary every 25 iterations
                        _summary = sess.run([summary_op],
                                            feed_dict={handle: training_handle})
                        writer.add_summary(_summary, global_step=iter_id)

            total_RMSE_loss /= total_train_batches  # compute mean of los
            total_MAE_loss /= total_train_batches # compute mean of accuracy

            best_val_RMSE_loss = val_check(sess, best_val_RMSE_loss)

        val_saver.restore(sess, "{}/best_validation_{}_{}.ckpt".format(saved_models_filepath,
                                                                       experiment_name, best_epoch))
        # restore model with best performance on validation set
        '''total_test_c_loss = 0.
        total_test_accuracy = 0.
        # computer test loss and accuracy and save
        with tqdm.tqdm(total=total_test_batches) as pbar_test:
            for batch_id, (x_batch, y_batch) in enumerate(test_data):
                c_loss_value, acc = sess.run(
                    [losses_ops["crossentropy_losses"], losses_ops["accuracy"]],
                    feed_dict={dropout_rate: dropout_rate_value, data_inputs: x_batch,
                               data_targets: y_batch, training_phase: False, rotate_data: False})
                total_test_c_loss += c_loss_value
                total_test_accuracy += acc
                iter_out = "test_loss: {}, test_accuracy: {}".format(total_test_c_loss / (batch_idx + 1),
                                                                     acc / (batch_idx + 1))
                pbar_test.set_description(iter_out)
                pbar_test.update(1)

        total_test_c_loss /= total_test_batches
        total_test_accuracy /= total_test_batches

        save_statistics(logs_filepath, "result_summary_statistics",
                        ["test set performance", -1, -1, -1, -1,
                         total_test_c_loss, total_test_accuracy])'''
