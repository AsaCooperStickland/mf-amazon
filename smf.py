import tensorflow as tf
import numpy as np

def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.001)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name):
    b_init = tf.constant_initializer(0.)
    return tf.get_variable(name, shape, initializer=b_init)


def calc_running_avg_loss(loss, running_avg_loss, step, decay=0.99):
    """Calculate the running average loss via exponential decay.
    This is used to implement early stopping w.r.t. a more smooth loss curve than the raw loss curve.

    Args:
      loss: loss on the most recent eval step
      running_avg_loss: running_avg_loss so far
      summary_writer: FileWriter object to write for tensorboard
      step: training iteration step
      decay: rate of exponential decay, a float between 0 and 1. Larger is smoother.

    Returns:
      running_avg_loss: new running average loss
    """
    if running_avg_loss == 0:  # on the first iteration just take the loss
        running_avg_loss = loss
    else:
        running_avg_loss = running_avg_loss * decay + (1 - decay) * loss
    #running_avg_loss = min(running_avg_loss, 12)  # clip
    #loss_sum = tf.Summary()
    #tag_name = 'running_avg_loss/decay=%f' % (decay)
    #loss_sum.value.add(tag=tag_name, simple_value=running_avg_loss)
    #summary_writer.add_summary(loss_sum, step)
    #tf.logging.info('running_avg_loss: %f', running_avg_loss)
    return running_avg_loss


class SMF:
    def __init__(self, num_users, num_items, latent_dim,
                 learning_rate=0.001, batch_size=256, reg_lambda=0.01):
        self.num_users = num_users
        self.num_items = num_items
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.reg_lambda = tf.constant(reg_lambda, dtype=tf.float32)
        #self.build_graph()

    def build_graph(self, u_idx, v_idx, r):
        #u_idx = tf.placeholder(tf.int32, [None])
        #v_idx = tf.placeholder(tf.int32, [None])
        #r = tf.placeholder(tf.float32, [None])

        self.U = weight_variable([self.num_users, self.latent_dim], 'U')
        self.V = weight_variable([self.num_items, self.latent_dim], 'V')
        self.U_bias = weight_variable([self.num_users], 'U_bias')
        self.V_bias = weight_variable([self.num_items], 'V_bias')

        self.U_embed = tf.nn.embedding_lookup(self.U, u_idx)
        self.V_embed = tf.nn.embedding_lookup(self.V, v_idx)
        self.U_bias_embed = tf.nn.embedding_lookup(self.U_bias, u_idx)
        self.V_bias_embed = tf.nn.embedding_lookup(self.V_bias, v_idx)
        self.r_hat = tf.reduce_sum(tf.multiply(self.U_embed, self.V_embed), reduction_indices=1)
        self.r_hat = tf.add(self.r_hat, self.U_bias_embed)
        self.r_hat = tf.add(self.r_hat, self.V_bias_embed)

        self.RMSE = tf.sqrt(tf.losses.mean_squared_error(r, self.r_hat))
        self.l2_loss = tf.nn.l2_loss(tf.subtract(r, self.r_hat))
        self.MAE = tf.reduce_mean(tf.abs(tf.subtract(r, self.r_hat)))
        self.reg = tf.add(tf.multiply(self.reg_lambda, tf.nn.l2_loss(self.U)), tf.multiply(self.reg_lambda, tf.nn.l2_loss(self.V)))
        self.reg_loss = tf.add(self.l2_loss, self.reg)

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        # self.train_step = self.optimizer.minimize(self.reg_loss)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # Needed for correct batch norm usage
        with tf.control_dependencies(update_ops):
            self.train_step_u = self.optimizer.minimize(self.reg_loss,
                                                        var_list=[self.U, self.U_bias],
                                                        colocate_gradients_with_ops=True)
            self.train_step_v = self.optimizer.minimize(self.reg_loss,
                                                        var_list=[self.V, self.V_bias],
                                                        colocate_gradients_with_ops=True)

        tf.summary.scalar("RMSE", self.RMSE)
        tf.summary.scalar("MAE", self.MAE)
        tf.summary.scalar("L2-Loss", self.l2_loss)
        tf.summary.scalar("Reg-Loss", self.reg_loss)

        # add op for merging summary
        self.summary_op = tf.summary.merge_all()

        # add Saver ops
        self.saver = tf.train.Saver()

        return self.RMSE, self.MAE, self.summary_op, \
            self.train_step_u, self.train_step_v, self.V_bias

    def construct_feeddict(self, u_idx, v_idx, r):
        return {self.u_idx: u_idx, self.v_idx: v_idx, self.r: r}

    def train_test_validation(self, M, train_idx, test_idx, valid_idx,
                              n_steps=100000, result_path='result/'):
        #nonzero_u_idx = M.nonzero()[0]
        #nonzero_v_idx = M.nonzero()[1]

        train_size = train_idx.size
        valid_size = valid_idx.size
        #train_uidx = self.nonzero_u_idx[train_idx]
        #train_vidx = self.nonzero_v_idx[train_idx]
        trainr = np.zeros(self.batch_size)
        validr = np.zeros(self.batch_size)

        best_val_rmse = np.inf
        best_val_mae = np.inf
        best_test_rmse = 0
        best_test_mae = 0
        valid_running_av = 0
        train_running_av = 0

        train_writer = tf.summary.FileWriter(result_path + '/train', graph=self.sess.graph)
        valid_writer = tf.summary.FileWriter(result_path + '/validation', graph=self.sess.graph)
        test_writer = tf.summary.FileWriter(result_path + '/test', graph=self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        print('Starting to train...')
        for step in range(1, n_steps):
            batch_idx = np.random.randint(train_size, size=self.batch_size)
            u_idx = self.nonzero_u_idx[train_idx[batch_idx]]
            v_idx = self.nonzero_v_idx[train_idx[batch_idx]]
            for i, (u, v) in enumerate(zip(u_idx, v_idx)):
                trainr[i] = M[(u, v)]
            feed_dict = self.construct_feeddict(u_idx, v_idx, trainr)

            self.sess.run(self.train_step_v, feed_dict=feed_dict)
            _, rmse, mae, summary_str = self.sess.run(
                [self.train_step_u, self.RMSE, self.MAE, self.summary_op], feed_dict=feed_dict)
            train_running_av = calc_running_avg_loss(rmse,
                                                     train_running_av, step)
            train_writer.add_summary(summary_str, step)
            if step % 50 == 0:
                batch_idx = np.random.randint(valid_size, size=self.batch_size)
                valid_u_idx = self.nonzero_u_idx[valid_idx[batch_idx]]
                valid_v_idx = self.nonzero_v_idx[valid_idx[batch_idx]]
                for i, (u, v) in enumerate(zip(valid_u_idx, valid_v_idx)):
                    validr[i] = M[(u, v)]
                feed_dict = self.construct_feeddict(valid_u_idx, valid_v_idx, validr)
                rmse_valid, mae_valid, summary_str = self.sess.run(
                    [self.RMSE, self.MAE, self.summary_op], feed_dict=feed_dict)
                valid_running_av = calc_running_avg_loss(rmse_valid,
                                                         valid_running_av, step,
                                                         decay=0.9)
                valid_writer.add_summary(summary_str, step)
                '''
                test_u_idx = nonzero_u_idx[test_idx]
                test_v_idx = nonzero_v_idx[test_idx]
                feed_dict = self.construct_feeddict(test_u_idx, test_v_idx, M)
                rmse_test, mae_test, summary_str = self.sess.run(
                    [self.RMSE, self.MAE, self.summary_op], feed_dict=feed_dict)

                test_writer.add_summary(summary_str, step)'''

                print("Step {0} | Train RMSE: {1:3.4f}, MAE: {2:3.4f}".format(
                    step, rmse, mae))
                print("Step {0} | Train running av: {1:3.4f}".format(
                    step, train_running_av))
                print("Step {0} | Valid  RMSE: {0:3.4f}, MAE: {1:3.4f}".format(
                    rmse_valid, mae_valid))
                print("Step {0} | Valid running av: {1:3.4f}".format(
                    step, valid_running_av))
                '''print("         | Test  RMSE: {0:3.4f}, MAE: {1:3.4f}".format(
                    rmse_test, mae_test))'''
                if step % 500 == 0:
                    if best_val_rmse > valid_running_av:
                        best_val_rmse = valid_running_av
                        print('Saving best model...')
                        self.saver.save(self.sess, result_path + "/model" + str(step) + ".ckpt")
                    #best_test_rmse = rmse_test

                if best_val_mae > mae_valid:
                    best_val_mae = mae_valid
                    #best_test_mae = mae_test

        self.saver.save(self.sess, result_path + "/model.ckpt")
        return best_test_rmse, best_test_mae
