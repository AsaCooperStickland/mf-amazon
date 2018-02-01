import itertools
import os

import argparse
import tqdm
import numpy as np
import tensorflow as tf
from utils.parser_utils import ParserClass
from utils.storage import build_experiment_folder, save_statistics
from sklearn.model_selection import KFold

from smf import SMF

tf.reset_default_graph()  # resets any previous graphs to clear memory
parser = argparse.ArgumentParser(description='Welcome to CNN experiments script')  # generates an argument parser
parser_extractor = ParserClass(parser=parser)  # creates a parser class to process the parsed input

batch_size, seed, epochs, logs_path, continue_from_epoch, tensorboard_enable,\
 experiment_prefix, l2_weight, latent_dim,\
 learning_rate = parser_extractor.get_argument_variables()


num_users = 339231
num_items = 83046
num_ratings = 500176
experiment_name = "experiment_{}_batch_size_{}_l2_{}_dim_{}".format(experiment_prefix,
																   batch_size, l2_weight,
																   latent_dim)
#num_item = 83046

n_trained_data = 10000000
batch_sizes = [256, 512, 1024]
# reg_lambdas = [0, 1e-10, 1e-7, 1e-5]
reg_lambdas = [2, 1e-5, 1e-3, 1e-1, 1]
learning_rates = [0.001, 0.0001]
latent_dims = [50, 100]

print("Running {}".format(experiment_name))
print("Starting from epoch {}".format(continue_from_epoch))

saved_models_filepath, logs_filepath = build_experiment_folder(experiment_name, logs_path) # generate experiment dir

def decode(serialized_example):
	features = tf.parse_single_example(
						  serialized_example,
						  features={
						  'UserId': tf.FixedLenFeature([], tf.int64),
						  'ItemId': tf.FixedLenFeature([], tf.int64),
						  'time': tf.FixedLenFeature([], tf.int64),
						  'rating': tf.FixedLenFeature([], tf.int64)})

	# Convert label from a scalar uint8 tensor to an int32 scalar.
	UserId = tf.cast(features['UserId'], tf.int64)
	ItemId = tf.cast(features['ItemId'], tf.int64)
	time = tf.cast(features['time'], tf.float32)
	rating = tf.cast(features['rating'], tf.float32)
	return UserId, ItemId, time, rating


def create_dataset(filename):
	dataset = tf.contrib.data.TFRecordDataset(filename)
	dataset = dataset.map(decode)
	dataset = dataset.repeat(epochs)
	dataset = dataset.batch(batch_size)
	return dataset

'''
# Placeholder setup
data_inputs = tf.placeholder(tf.float32, [batch_size, train_data.inputs.shape[1], train_data.inputs.shape[2],
										  train_data.inputs.shape[3]], 'data-inputs')
data_targets = tf.placeholder(tf.int32, [batch_size], 'data-targets')

training_phase = tf.placeholder(tf.bool, name='training-flag')
rotate_data = tf.placeholder(tf.bool, name='rotate-flag')
dropout_rate = tf.placeholder(tf.float32, name='dropout-prob')
'''
#with tf.Graph().as_default():
smf = SMF(num_users, num_items, latent_dim, learning_rate=learning_rate,
		  batch_size=256, reg_lambda=l2_weight)

if continue_from_epoch == -1:  # if this is a new experiment and not continuation of a previous one then generate a new
	# statistics file
	save_statistics(logs_filepath, "result_summary_statistics", ["epoch", "train_c_loss", "train_c_accuracy",
																 "val_c_loss", "val_c_accuracy",
																 "test_c_loss", "test_c_accuracy"], create=True)

start_epoch = continue_from_epoch if continue_from_epoch != -1 else 0  # if new experiment start from 0 otherwise
# continue where left off

dataset = create_dataset('./data/train_dataset.tfrecords')
# For adding val dataset: https://stackoverflow.com/questions/47356764/how-to-use-tensorflow-dataset-api-with-training-and-validation-sets
it = tf.contrib.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
init_train = it.make_initializer(dataset)
#iterator = dataset.make_initializable_iterator()
u_idx, v_idx, time, r = it.get_next()

MSE, MAE, summary_op, train_step_u, train_step_v, v = smf.build_graph(u_idx, v_idx, r)  # get graph operations (ops)

global_step = tf.Variable(0, name='global_step', trainable=False)
val_saver = tf.train.Saver()

train_size = int(0.8*num_ratings)
val_size = int(0.5*(num_ratings - train_size))

total_train_batches = int(int(0.8*num_ratings)/batch_size)
total_val_batches = int(val_size/batch_size)
#total_test_batches = = int(int(0.8*num_ratings)/batch_size)

best_epoch = 0

if tensorboard_enable:
	print("saved tensorboard file at", logs_filepath)
	writer = tf.summary.FileWriter(logs_filepath, graph=tf.get_default_graph())

init_op = tf.group(tf.global_variables_initializer(),
				   tf.local_variables_initializer())  # initialization op for the graph
scaffold = tf.train.Scaffold(local_init_op=tf.group(tf.local_variables_initializer(),
									 init_train), saver = val_saver)
#with tf.Session() as sess:
with tf.train.MonitoredTrainingSession(scaffold=scaffold,
									   checkpoint_dir=saved_models_filepath) as sess:
	while not sess.should_stop():
		sess.run(init_op)  # actually running the initialization op
		#train_saver = tf.train.Saver()  # saver object that will save our graph so we can reload it later for continuation of
		#val_saver = tf.train.Saver()
		#  training or inference

		continue_from_epoch = -1

		if continue_from_epoch != -1:
			train_saver.restore(sess, "{}/{}_{}.ckpt".format(saved_models_filepath, experiment_name,
													   continue_from_epoch))  # restore previous graph to continue operations

		best_val_RMSE = 0.
		with tqdm.tqdm(total=epochs) as epoch_pbar:
			for e in range(start_epoch, epochs):
				total_RMSE_loss = 0.
				total_MAE_loss = 0.
				with tqdm.tqdm(total=total_train_batches) as pbar_train:
					for batch_idx in range(total_train_batches):
						iter_id = e * total_train_batches + batch_idx
						'''_u_idx, _v_idx, _time, _r, _v = sess.run([u_idx, v_idx, time, r, v])
						print(_v)'''
						#feed_dict = smf.construct_feeddict(u_idx, v_idx, r)
						rmse_train, mae_train, summary_str, u_update, v_update =\
							sess.run([MSE, MAE, summary_op, train_step_u, train_step_v])
						# Here we execute the c_error_opt_op which trains the network and also the ops that compute the
						# loss and accuracy, we save those in _, c_loss_value and acc respectively.
						total_RMSE_loss += rmse_train  # add loss of current iter to sum
						total_MAE_loss += mae_train # add acc of current iter to sum

						iter_out = "iter_num: {}, train_RMSE: {},"\
									"train_MAE: {}, batch_RMSE: {}".format(iter_id,
																 total_RMSE_loss / (batch_idx + 1),
																 total_MAE_loss / (batch_idx + 1),
																 rmse_train)
						pbar_train.set_description(iter_out)
						pbar_train.update(1)
						if tensorboard_enable and batch_idx % 25 == 0:  # save tensorboard summary every 25 iterations
							feed_dict = smf.construct_feeddict(valid_u_idx, valid_v_idx, validr)
							_summary = sess.run(
								[self.summary_op],
								feed_dict=feed_dict)
							writer.add_summary(_summary, global_step=iter_id)

				total_RMSE_loss /= total_train_batches  # compute mean of los
				total_MAE_loss /= total_train_batches # compute mean of accuracy

				total_val_c_loss = 0.
				total_val_accuracy = 0. #  run validation stage, note how training_phase placeholder is set to False
				# and that we do not run the c_error_opt_op which runs gradient descent, but instead only call the loss ops
				#  to collect losses on the validation set
				with tqdm.tqdm(total=total_val_batches) as pbar_val:
					for batch_idx, (x_batch, y_batch) in enumerate(val_data):
						c_loss_value, acc = sess.run(
							[losses_ops["crossentropy_losses"], losses_ops["accuracy"]],
							feed_dict={dropout_rate: dropout_rate_value, data_inputs: x_batch,
									   data_targets: y_batch, training_phase: False, rotate_data: False})
						total_val_c_loss += c_loss_value
						total_val_accuracy += acc
						iter_out = "val_loss: {}, val_accuracy: {}".format(total_val_c_loss / (batch_idx + 1),
																		   total_val_accuracy / (batch_idx + 1))
						pbar_val.set_description(iter_out)
						pbar_val.update(1)

				total_val_c_loss /= total_val_batches
				total_val_accuracy /= total_val_batches

				if best_val_accuracy < total_val_accuracy:  # check if val acc better than the previous best and if
					# so save current as best and save the model as the best validation model to be used on the test set
					#  after the final epoch
					best_val_accuracy = total_val_accuracy
					best_epoch = e
					save_path = val_saver.save(sess, "{}/best_validation_{}_{}.ckpt".format(saved_models_filepath, experiment_name, e))
					print("Saved best validation score model at", save_path)

				epoch_pbar.update(1)
				# save statistics of this epoch, train and val without test set performance
				save_statistics(logs_filepath, "result_summary_statistics",
								[e, total_c_loss, total_accuracy, total_val_c_loss, total_val_accuracy,
								 -1, -1])

			val_saver.restore(sess, "{}/best_validation_{}_{}.ckpt".format(saved_models_filepath, experiment_name, best_epoch))
			# restore model with best performance on validation set
			total_test_c_loss = 0.
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
							 total_test_c_loss, total_test_accuracy])
