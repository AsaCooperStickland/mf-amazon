import tensorflow as tf
import numpy as np

def weight_variable(shape, name, trainable=True):
    initial = tf.truncated_normal(shape, stddev=0.001)
    return tf.Variable(initial, name=name, trainable=trainable)


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


class BaseSMF:
    def __init__(self, num_users, num_items, latent_dim,
                 learning_rate=0.001, batch_size=256, reg_lambda=0.01):
        self.num_users = num_users
        self.num_items = num_items
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.reg_lambda = tf.constant(reg_lambda, dtype=tf.float32)
        #self.lr = tf.maximum(1e-5,tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps, self.decay, staircase=True))
        #self.build_graph()

    def build_graph(self, u_idx, v_idx, r):
        #u_idx = tf.placeholder(tf.int32, [None])
        #v_idx = tf.placeholder(tf.int32, [None])
        #r = tf.placeholder(tf.float32, [None])

        self.U_bias = weight_variable([self.num_users], 'U_bias')

        self.U_bias_embed = tf.nn.embedding_lookup(self.U_bias, u_idx)
        #self.V_bias_embed = tf.nn.embedding_lookup(self.V_bias, v_idx)
        self.r_hat = self.U_bias_embed
        #self.r_hat = tf.add(self.r_hat, self.V_bias_embed)

        self.RMSE = tf.sqrt(tf.losses.mean_squared_error(r, self.r_hat))
        self.l2_loss = tf.nn.l2_loss(tf.subtract(r, self.r_hat))
        self.MAE = tf.reduce_mean(tf.abs(tf.subtract(r, self.r_hat)))
        #self.reg = tf.add(tf.multiply(self.reg_lambda, tf.nn.l2_loss(self.U)), tf.multiply(self.reg_lambda, tf.nn.l2_loss(self.V)))
        self.reg_loss = self.l2_loss

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        # self.train_step = self.optimizer.minimize(self.reg_loss)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # Needed for correct batch norm usage
        with tf.control_dependencies(update_ops):
            self.train_step_u = self.optimizer.minimize(self.reg_loss,
                                                        var_list=[self.U_bias],
                                                        colocate_gradients_with_ops=True)


        tf.summary.scalar("RMSE", self.RMSE)
        tf.summary.scalar("MAE", self.MAE)
        tf.summary.scalar("L2-Loss", self.l2_loss)
        tf.summary.scalar("Reg-Loss", self.reg_loss)

        # add op for merging summary
        self.summary_op = tf.summary.merge_all()

        # add Saver ops
        self.saver = tf.train.Saver()

        return self.RMSE, self.MAE, self.l2_loss, self.summary_op,\
            self.train_step_u


class SMF:
    def __init__(self, num_users, num_items, latent_dim,
                 learning_rate=0.001, batch_size=256, reg_lambda=0.01):
        self.num_users = num_users
        self.num_items = num_items
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.reg_lambda = tf.constant(reg_lambda, dtype=tf.float32)
        #self.lr = tf.maximum(1e-5,tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps, self.decay, staircase=True))
        #self.build_graph()

    def build_graph(self, u_idx, v_idx, r, offset):
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
        self.r_hat = tf.add(self.r_hat, self.V_bias_embed) + offset

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

        return self.RMSE, self.MAE, self.l2_loss, self.summary_op,\
            self.train_step_u, self.train_step_v, self.V_bias


class RNNSMF:
    def __init__(self, num_users, num_items, latent_dim, rnn_dim=20,
                 learning_rate=0.001, batch_size=256, reg_lambda=0.01,
                 dropout_p_hidden=0.8, rnn_dim=20):
        self.num_users = num_users
        self.num_items = num_items
        self.latent_dim = latent_dim
        self.rnn_dim = rnn_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.reg_lambda = tf.constant(reg_lambda, dtype=tf.float32)
        self.dropout_p_hidden = dropout_p_hidden
        self.rnn_dim = rnn_dim
        #self.lr = tf.maximum(1e-5,tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps, self.decay, staircase=True))
        #self.build_graph()

    def build_graph(self, u_idx, u_t_itemidx, u_t_rs, u_state, v_idx,  r):

        #self.U_t = weight_variable([self.num_users, self.rnn_dim], 'U_t', trainable=False)
        self.U_W = weight_variable([self.num_items + 3, self.rnn_dim], 'U_W')
        # Should this be items * rnn_dim not users * rnn_dim???

        self.U = weight_variable([self.num_users, self.latent_dim], 'U')
        self.V = weight_variable([self.num_items, self.latent_dim], 'V')
        self.U_bias = weight_variable([self.num_users], 'U_bias')
        self.V_bias = weight_variable([self.num_items], 'V_bias')

        self.U_embed = tf.nn.embedding_lookup(self.U, u_idx)
        self.V_embed = tf.nn.embedding_lookup(self.V, v_idx)
        self.U_bias_embed = tf.nn.embedding_lookup(self.U_bias, u_idx)
        self.V_bias_embed = tf.nn.embedding_lookup(self.V_bias, v_idx)

        self.U_y_t = tf.nn.embedding_lookup_sparse(self.U_W, sp_ids=u_t_itemidx,
                                                   sp_weights=u_t_rs, combiner="sum")
        self.U_final = weight_variable([self.num_users, self.rnn_size * self.latent_dim], 'U_final')
        self.U_final_bias = weight_variable([self.num_items, self.latent_dim], 'U_final_bias')

        self.U_final_embed = tf.nn.embedding_lookup(self.U_final, u_idx)
        self.U_final_embed = tf.reshape(self.U_final_embed, [self.rnn_size, self.latent_dim])
        self.U_final_bias_embed = tf.nn.embedding_lookup(self.U_final_bias, u_idx)

        self.V_t = weight_variable([self.num_items, self.latent_dim], 'V_t')
        self.V_t_embed = tf.nn.embedding_lookup(self.V_t, v_idx)

        # in u_t_itemidx put newbie, timestamps - with corresponding values in
        # u_t_rs.
        with tf.name_scope("rnn"):

            cell = rnn_cell.GRUCell(self.rnn_dim, activation=self.hidden_act)
            drop_cell = rnn_cell.DropoutWrapper(cell, output_keep_prob=self.dropout_p_hidden)
            stacked_cell = rnn_cell.MultiRNNCell([drop_cell] * self.layers)
            output, state = stacked_cell(self.U_y_t, tuple(self.state))
            self.final_state = state
            self.U_t = output

        u_affine = tf.matmul(output, self.U_final_embed) + U_final_bias_embed
        t_contrib = tf.reduce_sum(tf.multiply(u_affine, self.V_t_embed), reduction_indices=1)
        s_contrib = tf.reduce_sum(tf.multiply(self.U_embed, self.V_embed), reduction_indices=1)
        s_contrib = tf.add(self.r_hat, self.U_bias_embed)
        s_contrib = tf.add(self.r_hat, self.V_bias_embed)

        self.r_hat = tf.add(t_contrib, s_contrib)

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

        return self.RMSE, self.MAE, self.l2_loss, self.summary_op,\
            self.train_step_u, self.train_step_v, self.V_bias
