
import tensorflow as tf
import numpy as np

from tensorflow.python.ops.rnn_cell import GRUCell
from tensorflow.python.ops.rnn_cell import LSTMCell
from tensorflow.python.ops.rnn_cell import MultiRNNCell


def weight_variable(shape, name, trainable=True):
    initial = tf.truncated_normal(shape, stddev=0.001)
    return tf.Variable(initial, name=name, trainable=trainable)


def bias_variable(shape, name):
    b_init = tf.constant_initializer(0.)
    return tf.get_variable(name, shape, initializer=b_init)


def build_single_cell(hidden_units):
    cell_type = LSTMCell
    # cell_type = GRUCell
    cell = cell_type(hidden_units)
    return cell


def extract_axis_1(data, ind):
    batch_range = tf.range(tf.shape(data)[0])
    indices = tf.stack([batch_range, ind], axis=1)
    res = tf.gather_nd(data, indices)
    return res


def extract_axis_1_vanilla(data, ind):
    batch_range = tf.range(tf.shape(data)[0])
    indices = tf.stack([batch_range, ind], axis=1)
    res = tf.gather_nd(data, indices)
    return res


def build_cell(hidden_units, depth=1):
    cell_list = [build_single_cell(hidden_units) for i in range(depth)]
    return MultiRNNCell(cell_list)


def vanilla_attention(queries, keys, keys_length):
    '''
      queries:     [B, H]
      keys:        [B, T, H]
      keys_length: [B]
    '''
    #queries = tf.tile(queries, [1, 2])
    queries = tf.expand_dims(queries, 1) # [B, 1, H]
    # Multiplication
    outputs = tf.matmul(queries, tf.transpose(keys, [0, 2, 1])) # [B, 1, T]

    # Mask
    key_masks = tf.sequence_mask(keys_length, tf.shape(keys)[1])   # [B, T]
    key_masks = tf.expand_dims(key_masks, 1) # [B, 1, T]
    paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
    outputs = tf.where(key_masks, outputs, paddings)  # [B, 1, T]

    # Scale
    outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5)

    # Activation
    outputs = tf.nn.softmax(outputs)  # [B, 1, T]

    # Weighted sum
    outputs = tf.matmul(outputs, keys)  # [B, 1, H]

    return outputs


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
                 learning_rate=0.001, reg_lambda=0.01,
                 dropout_p_hidden=0.8, item_rnn_dim=10, rating_rnn_dim=2):
        self.num_users = num_users
        self.num_items = num_items
        self.latent_dim = latent_dim
        self.item_rnn_dim = item_rnn_dim
        self.rating_rnn_dim = rating_rnn_dim
        self.rnn_dim = item_rnn_dim + rating_rnn_dim
        self.learning_rate = learning_rate
        self.reg_lambda = tf.constant(reg_lambda, dtype=tf.float32)


        self.dropout_p = tf.placeholder(tf.float32, [])
        self.u_idx = tf.placeholder(tf.int32, [None, ])  # [B]
        self.i_idx = tf.placeholder(tf.int32, [None, ])  # [B]
        #self.j = tf.placeholder(tf.int32, [None,]) # [B]
        self.r = tf.placeholder(tf.float32, [None, ])  # [B]
        self.i_hist = tf.placeholder(tf.int32, [None, None])  # [B, T]
        self.r_hist = tf.placeholder(tf.int32, [None, None])  # [B, T]
        self.sl = tf.placeholder(tf.int32, [None, ])  # [B]
        self.lr = tf.placeholder(tf.float64, [])
        #self.lr = tf.maximum(1e-5,tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps, self.decay, staircase=True))
        #self.build_graph()

    def build_graph(self, offset):

        self.U_bias = bias_variable([self.num_users], 'U_bias')
        self.V_bias = bias_variable([self.num_items], 'V_bias')
        self.U_bias_embed = tf.nn.embedding_lookup(self.U_bias, self.u_idx)
        self.V_bias_embed = tf.nn.embedding_lookup(self.V_bias, self.i_idx)

        self.r_hat = offset + self.U_bias_embed + self.V_bias_embed

        self.RMSE = tf.sqrt(tf.losses.mean_squared_error(self.r, self.r_hat))
        self.cost = 2 * tf.nn.l2_loss(tf.subtract(self.r, self.r_hat))
        self.MAE = tf.reduce_mean(tf.abs(tf.subtract(self.r, self.r_hat)))

        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        # self.train_step = self.optimizer.minimize(self.reg_loss)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # Needed for correct batch norm usage
        with tf.control_dependencies(update_ops):
            self.train_step = self.optimizer.minimize(self.cost,
                                                        colocate_gradients_with_ops=True)

        tf.summary.scalar("RMSE", self.RMSE)
        tf.summary.scalar("MAE", self.MAE)
        tf.summary.scalar("Cost", self.cost)
        tf.summary.scalar("Reg-Loss", self.cost)

        # add op for merging summary
        self.summary_op = tf.summary.merge_all()

        # add Saver ops
        self.saver = tf.train.Saver()

        return self.RMSE, self.MAE, self.cost, self.summary_op,\
            self.train_step

    def train(self, sess, data, dropout):
        RMSE, MAE, cost, summary_op,\
          train_step = sess.run([self.RMSE, self.MAE, self.cost, self.summary_op,
                           self.train_step], feed_dict={
                           self.u_idx: data[0],
                           self.i_idx: data[1],
                           self.r: data[2],
                           self.i_hist: data[3],
                           self.r_hist: data[4],
                           self.sl: data[5],
                           self.dropout_p: dropout
                           })
        return RMSE, MAE, cost, summary_op

    def eval(self, sess, data, dropout=1.0):
        RMSE, MAE, cost, summary_op,\
           = sess.run([self.RMSE, self.MAE, self.cost, self.summary_op],
                      feed_dict={
                           self.u_idx: data[0],
                           self.i_idx: data[1],
                           self.r: data[2],
                           self.i_hist: data[3],
                           self.r_hist: data[4],
                           self.sl: data[5],
                           self.dropout_p: dropout
                           })
        return RMSE, MAE, cost, summary_op


class SMF:
    def __init__(self, num_users, num_items, latent_dim,
                 learning_rate=0.001, batch_size=256, reg_lambda=0.01):
        self.num_users = num_users
        self.num_items = num_items
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.reg_lambda = tf.constant(reg_lambda, dtype=tf.float32)

        self.u_idx = tf.placeholder(tf.int32, [None, ])  # [B]
        self.i_idx = tf.placeholder(tf.int32, [None, ])  # [B]
        self.r = tf.placeholder(tf.float32, [None, ])
        self.dropout_p = tf.placeholder(tf.float32, [])


    def build_graph(self, offset):
        #u_idx = tf.placeholder(tf.int32, [None])
        #v_idx = tf.placeholder(tf.int32, [None])
        #r = tf.placeholder(tf.float32, [None])

        self.U = bias_variable([self.num_users, self.latent_dim], 'U')
        self.V = bias_variable([self.num_items, self.latent_dim], 'V')
        self.U_bias = bias_variable([self.num_users], 'U_bias')
        self.V_bias = bias_variable([self.num_items], 'V_bias')

        self.U_embed = tf.nn.embedding_lookup(self.U, self.u_idx)
        #self.U_embed = tf.nn.dropout(self.U_embed, 0.9)
        self.V_embed = tf.nn.embedding_lookup(self.V, self.i_idx)
        #self.V_embed = tf.nn.dropout(self.V_embed, 0.9)
        self.U_bias_embed = tf.nn.embedding_lookup(self.U_bias, self.u_idx)
        self.V_bias_embed = tf.nn.embedding_lookup(self.V_bias, self.i_idx)
        self.r_hat = tf.reduce_sum(tf.multiply(self.U_embed, self.V_embed), reduction_indices=1)
        self.r_hat = tf.add(self.r_hat, self.U_bias_embed)
        self.r_hat = tf.add(self.r_hat, self.V_bias_embed) + offset

        self.RMSE = tf.sqrt(tf.losses.mean_squared_error(self.r, self.r_hat))
        self.cost = 2. * tf.nn.l2_loss(tf.subtract(self.r, self.r_hat))
        self.MAE = tf.reduce_mean(tf.abs(tf.subtract(self.r, self.r_hat)))
        self.reg = tf.add(tf.multiply(self.reg_lambda, tf.nn.l2_loss(self.U)), tf.multiply(self.reg_lambda, tf.nn.l2_loss(self.V)))
        self.reg_loss = tf.add(self.cost, self.reg)

        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
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
        tf.summary.scalar("L2-Loss", self.cost)
        tf.summary.scalar("Reg-Loss", self.reg_loss)

        # add op for merging summary
        self.summary_op = tf.summary.merge_all()

        # add Saver ops
        self.saver = tf.train.Saver()

        return self.RMSE, self.MAE, self.cost, self.summary_op,\
            self.train_step_u, self.train_step_v, self.V_bias


    def train(self, sess, data, dropout):
        RMSE, MAE, cost, summary_op,\
          train_step_u, train_step_v = sess.run([self.RMSE, self.MAE,
                           self.cost, self.summary_op,
                           self.train_step_u, self.train_step_v], feed_dict={
                           self.u_idx: data[0],
                           self.i_idx: data[1],
                           self.r: data[2],
                           self.dropout_p: dropout
                           })
        return RMSE, MAE, cost, summary_op


    def eval(self, sess, data, dropout=1.0):
        RMSE, MAE, cost, summary_op,\
           = sess.run([self.RMSE, self.MAE, self.cost, self.summary_op],
                      feed_dict={
                           self.u_idx: data[0],
                           self.i_idx: data[1],
                           self.r: data[2],
                           self.dropout_p: dropout
                           })
        return RMSE, MAE, cost, summary_op

class RNNSMF:
    def __init__(self, num_users, num_items, latent_dim,
                 learning_rate=0.001, reg_lambda=0.01,
                 dropout_p_hidden=0.8, item_rnn_dim=10, rating_rnn_dim=2):
        self.num_users = num_users
        self.num_items = num_items
        self.latent_dim = latent_dim
        self.item_rnn_dim = item_rnn_dim
        self.rating_rnn_dim = rating_rnn_dim
        self.rnn_dim = item_rnn_dim + rating_rnn_dim
        self.learning_rate = learning_rate
        self.reg_lambda = tf.constant(reg_lambda, dtype=tf.float32)


        self.dropout_p = tf.placeholder(tf.float32, [])
        self.u_idx = tf.placeholder(tf.int32, [None, ])  # [B]
        self.i_idx = tf.placeholder(tf.int32, [None, ])  # [B]
        #self.j = tf.placeholder(tf.int32, [None,]) # [B]
        self.r = tf.placeholder(tf.float32, [None, ])  # [B]
        self.i_hist = tf.placeholder(tf.int32, [None, None])  # [B, T]
        self.r_hist = tf.placeholder(tf.int32, [None, None])  # [B, T]
        self.sl = tf.placeholder(tf.int32, [None, ])  # [B]
        self.lr = tf.placeholder(tf.float64, [])
        #self.lr = tf.maximum(1e-5,tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps, self.decay, staircase=True))
        #self.build_graph()

    def build_graph(self, offset):

        #self.U_t = weight_variable([self.num_users, self.rnn_dim], 'U_t', trainable=False)

        # Should this be items * rnn_dim not users * rnn_dim???

        self.U = weight_variable([self.num_users, self.latent_dim], 'U')
        self.V = weight_variable([self.num_items, self.latent_dim], 'V')
        self.U_bias = bias_variable([self.num_users], 'U_bias')
        self.V_bias = bias_variable([self.num_items], 'V_bias')

        self.U_W = weight_variable([self.num_items, self.item_rnn_dim], 'U_W')
        self.R_W = weight_variable([5, self.rating_rnn_dim], 'R_W')

        self.U_embed = tf.nn.embedding_lookup(self.U, self.u_idx)
        self.V_embed = tf.nn.embedding_lookup(self.V, self.i_idx)
        self.U_bias_embed = tf.nn.embedding_lookup(self.U_bias, self.u_idx)
        self.V_bias_embed = tf.nn.embedding_lookup(self.V_bias, self.i_idx)
        self.i_hist_emb = tf.nn.embedding_lookup(self.U_W, self.i_hist)
        self.r_hist_emb = tf.nn.embedding_lookup(self.R_W, self.r_hist)
        '''
        self.U_final = weight_variable([self.num_users, 2 * self.rnn_dim * self.latent_dim], 'U_final')
        self.U_final_bias = weight_variable([self.num_items, self.latent_dim], 'U_final_bias')

        self.U_final_embed = tf.nn.embedding_lookup(self.U_final, self.u_idx)
        self.U_final_embed = tf.reshape(self.U_final_embed, [-1, self.latent_dim, 2 * self.rnn_dim])
        self.U_final_bias_embed = tf.nn.embedding_lookup(self.U_final_bias, self.u_idx)

        self.V_t = weight_variable([self.num_items, self.latent_dim], 'V_t')
        self.V_t_embed = tf.nn.embedding_lookup(self.V_t, self.i_idx)'''

        self.U_final = weight_variable([self.rnn_dim, self.latent_dim], 'U_final')
        self.U_final_bias = bias_variable([self.latent_dim], 'U_final_bias')

        self.V_t = weight_variable([self.latent_dim], 'V_t')
        concat_emb = tf.concat([self.i_hist_emb, self.r_hist_emb], 2)
        with tf.name_scope("rnn"):

            cell_fw = tf.contrib.rnn.DropoutWrapper(build_cell(self.rnn_dim),
                                                    variational_recurrent=True,
                                                    input_keep_prob=self.dropout_p,
                                                    output_keep_prob=self.dropout_p,
                                                    state_keep_prob=self.dropout_p,
                                                    input_size=(self.rnn_dim),
                                                    dtype=tf.float32)
            '''cell_bw = tf.contrib.rnn.DropoutWrapper(build_cell(2 * self.rnn_dim),
                                                    variational_recurrent=True,
                                                    input_keep_prob=0.95,
                                                    output_keep_prob=0.95,
                                                    state_keep_prob=0.95,
                                                    input_size=(2 * self.rnn_dim),
                                                    dtype=tf.float32)'''
            rnn_output, _ = tf.nn.dynamic_rnn(cell_fw,
                                              concat_emb, self.sl,
                                              dtype=tf.float32)
            rnn_output = extract_axis_1(rnn_output, self.sl-1)
        #print(rnn_output)
        #print(self.U_final_embed)
        '''u_affine = tf.tensordot(rnn_output, self.U_final_embed, axes=[[0, 1], [0, 2]])\
            + self.U_final_bias_embed
        print(u_affine)
        u_affine = tf.reshape(u_affine, [-1, self.latent_dim])
        t_contrib = tf.reduce_sum(tf.multiply(u_affine, self.V_t_embed), reduction_indices=1)'''
        u_affine = tf.nn.relu(tf.matmul(rnn_output, self.U_final) + self.U_final_bias)
        t_contrib = tf.reduce_sum(tf.multiply(u_affine, self.V_t), reduction_indices=1)
        s_contrib = tf.reduce_sum(tf.multiply(self.U_embed, self.V_embed), reduction_indices=1)
        s_contrib = tf.add(s_contrib, self.U_bias_embed)
        s_contrib = tf.add(s_contrib, self.V_bias_embed)
        self.r_hat = tf.add(t_contrib, s_contrib) + offset
        #self.r_hat = t_contrib + offset

        self.RMSE = tf.sqrt(tf.losses.mean_squared_error(self.r, self.r_hat))
        self.cost = 2 * tf.nn.l2_loss(tf.subtract(self.r, self.r_hat))
        self.MAE = tf.reduce_mean(tf.abs(tf.subtract(self.r, self.r_hat)))
        self.reg = tf.add(tf.multiply(self.reg_lambda, tf.nn.l2_loss(self.U)), tf.multiply(self.reg_lambda, tf.nn.l2_loss(self.V)))
        self.reg_loss = tf.add(self.cost, self.reg)
        #self.reg_loss = self.cost

        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        # self.train_step = self.optimizer.minimize(self.reg_loss)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # Needed for correct batch norm usage
        with tf.control_dependencies(update_ops):
            self.train_step = self.optimizer.minimize(self.reg_loss,
                                                        colocate_gradients_with_ops=True)

        tf.summary.scalar("RMSE", self.RMSE)
        tf.summary.scalar("MAE", self.MAE)
        tf.summary.scalar("Cost", self.cost)
        tf.summary.scalar("Reg-Loss", self.reg_loss)

        # add op for merging summary
        self.summary_op = tf.summary.merge_all()

        # add Saver ops
        self.saver = tf.train.Saver()

        return self.RMSE, self.MAE, self.cost, self.summary_op,\
            self.train_step

    def train(self, sess, data, dropout):
        RMSE, MAE, cost, summary_op,\
          train_step = sess.run([self.RMSE, self.MAE, self.cost, self.summary_op,
                           self.train_step], feed_dict={
                           self.u_idx: data[0],
                           self.i_idx: data[1],
                           self.r: data[2],
                           self.i_hist: data[3],
                           self.r_hist: data[4],
                           self.sl: data[5],
                           self.dropout_p: dropout
                           })
        return RMSE, MAE, cost, summary_op

    def eval(self, sess, data, dropout=1.0):
        RMSE, MAE, cost, summary_op,\
           = sess.run([self.RMSE, self.MAE, self.cost, self.summary_op],
                      feed_dict={
                           self.u_idx: data[0],
                           self.i_idx: data[1],
                           self.r: data[2],
                           self.i_hist: data[3],
                           self.r_hist: data[4],
                           self.sl: data[5],
                           self.dropout_p: dropout
                           })
        return RMSE, MAE, cost, summary_op


class JustRNNSMF:
    def __init__(self, num_users, num_items, latent_dim,
                 learning_rate=0.001, reg_lambda=0.01,
                 dropout_p_hidden=0.8, item_rnn_dim=10, rating_rnn_dim=2):
        self.num_users = num_users
        self.num_items = num_items
        self.latent_dim = latent_dim
        self.item_rnn_dim = item_rnn_dim
        self.rating_rnn_dim = rating_rnn_dim
        self.rnn_dim = item_rnn_dim + rating_rnn_dim
        self.learning_rate = learning_rate
        self.reg_lambda = tf.constant(reg_lambda, dtype=tf.float32)


        self.dropout_p = tf.placeholder(tf.float32, [])
        self.u_idx = tf.placeholder(tf.int32, [None, ])  # [B]
        self.i_idx = tf.placeholder(tf.int32, [None, ])  # [B]
        #self.j = tf.placeholder(tf.int32, [None,]) # [B]
        self.r = tf.placeholder(tf.float32, [None, ])  # [B]
        self.i_hist = tf.placeholder(tf.int32, [None, None])  # [B, T]
        self.r_hist = tf.placeholder(tf.int32, [None, None])  # [B, T]
        self.sl = tf.placeholder(tf.int32, [None, ])  # [B]
        self.lr = tf.placeholder(tf.float64, [])
        #self.lr = tf.maximum(1e-5,tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps, self.decay, staircase=True))
        #self.build_graph()

    def build_graph(self, offset):

        #self.U_t = weight_variable([self.num_users, self.rnn_dim], 'U_t', trainable=False)

        # Should this be items * rnn_dim not users * rnn_dim???

        self.U_W = weight_variable([self.num_items, self.item_rnn_dim], 'U_W')
        self.R_W = weight_variable([5, self.rating_rnn_dim], 'R_W')

        self.U_bias = bias_variable([self.num_users], 'U_bias')
        self.V_bias = bias_variable([self.num_items], 'V_bias')
        self.U_bias_embed = tf.nn.embedding_lookup(self.U_bias, self.u_idx)
        self.V_bias_embed = tf.nn.embedding_lookup(self.V_bias, self.i_idx)

        self.i_hist_emb = tf.nn.embedding_lookup(self.U_W, self.i_hist)
        self.r_hist_emb = tf.nn.embedding_lookup(self.R_W, self.r_hist)

        self.U_final = weight_variable([self.rnn_dim, self.latent_dim], 'U_final')
        self.U_final_bias = bias_variable([self.latent_dim], 'U_final_bias')

        self.V_t = weight_variable([self.latent_dim], 'V_t')
        concat_emb = tf.concat([self.i_hist_emb, self.r_hist_emb], 2)
        with tf.name_scope("rnn"):

            cell_fw = tf.contrib.rnn.DropoutWrapper(build_cell(self.rnn_dim),
                                                    variational_recurrent=True,
                                                    input_keep_prob=self.dropout_p,
                                                    output_keep_prob=self.dropout_p,
                                                    state_keep_prob=self.dropout_p,
                                                    input_size=(self.rnn_dim),
                                                    dtype=tf.float32)
            rnn_output, _ = tf.nn.dynamic_rnn(cell_fw,
                                              concat_emb, self.sl,
                                              dtype=tf.float32)
            rnn_output = extract_axis_1(rnn_output, self.sl-1)
        #print(rnn_output)
        #print(self.U_final_embed)
        u_affine = tf.nn.relu(tf.matmul(rnn_output, self.U_final) + self.U_final_bias)
        t_contrib = tf.reduce_sum(tf.multiply(u_affine, self.V_t), reduction_indices=1)
        self.r_hat = t_contrib + offset + self.U_bias_embed + self.V_bias_embed

        self.RMSE = tf.sqrt(tf.losses.mean_squared_error(self.r, self.r_hat))
        self.cost = 2 * tf.nn.l2_loss(tf.subtract(self.r, self.r_hat))
        self.MAE = tf.reduce_mean(tf.abs(tf.subtract(self.r, self.r_hat)))

        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        # self.train_step = self.optimizer.minimize(self.reg_loss)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # Needed for correct batch norm usage
        with tf.control_dependencies(update_ops):
            self.train_step = self.optimizer.minimize(self.cost,
                                                        colocate_gradients_with_ops=True)

        tf.summary.scalar("RMSE", self.RMSE)
        tf.summary.scalar("MAE", self.MAE)
        tf.summary.scalar("Cost", self.cost)
        tf.summary.scalar("Reg-Loss", self.cost)

        # add op for merging summary
        self.summary_op = tf.summary.merge_all()

        # add Saver ops
        self.saver = tf.train.Saver()

        return self.RMSE, self.MAE, self.cost, self.summary_op,\
            self.train_step

    def train(self, sess, data, dropout):
        RMSE, MAE, cost, summary_op,\
          train_step = sess.run([self.RMSE, self.MAE, self.cost, self.summary_op,
                           self.train_step], feed_dict={
                           self.u_idx: data[0],
                           self.i_idx: data[1],
                           self.r: data[2],
                           self.i_hist: data[3],
                           self.r_hist: data[4],
                           self.sl: data[5],
                           self.dropout_p: dropout
                           })
        return RMSE, MAE, cost, summary_op

    def eval(self, sess, data, dropout=1.0):
        RMSE, MAE, cost, summary_op,\
           = sess.run([self.RMSE, self.MAE, self.cost, self.summary_op],
                      feed_dict={
                           self.u_idx: data[0],
                           self.i_idx: data[1],
                           self.r: data[2],
                           self.i_hist: data[3],
                           self.r_hist: data[4],
                           self.sl: data[5],
                           self.dropout_p: dropout
                           })
        return RMSE, MAE, cost, summary_op

class RNNSMF:
    def __init__(self, num_users, num_items, latent_dim,
                 learning_rate=0.001, reg_lambda=0.01,
                 dropout_p_hidden=0.8, item_rnn_dim=10, rating_rnn_dim=2):
        self.num_users = num_users
        self.num_items = num_items
        self.latent_dim = latent_dim
        self.item_rnn_dim = item_rnn_dim
        self.rating_rnn_dim = rating_rnn_dim
        self.rnn_dim = item_rnn_dim + rating_rnn_dim
        self.learning_rate = learning_rate
        self.reg_lambda = tf.constant(reg_lambda, dtype=tf.float32)


        self.dropout_p = tf.placeholder(tf.float32, [])
        self.u_idx = tf.placeholder(tf.int32, [None, ])  # [B]
        self.i_idx = tf.placeholder(tf.int32, [None, ])  # [B]
        #self.j = tf.placeholder(tf.int32, [None,]) # [B]
        self.r = tf.placeholder(tf.float32, [None, ])  # [B]
        self.i_hist = tf.placeholder(tf.int32, [None, None])  # [B, T]
        self.r_hist = tf.placeholder(tf.int32, [None, None])  # [B, T]
        self.sl = tf.placeholder(tf.int32, [None, ])  # [B]
        self.lr = tf.placeholder(tf.float64, [])
        #self.lr = tf.maximum(1e-5,tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps, self.decay, staircase=True))
        #self.build_graph()

    def build_graph(self, offset):

        #self.U_t = weight_variable([self.num_users, self.rnn_dim], 'U_t', trainable=False)

        # Should this be items * rnn_dim not users * rnn_dim???

        self.U = weight_variable([self.num_users, self.latent_dim], 'U')
        self.V = weight_variable([self.num_items, self.latent_dim], 'V')
        self.U_bias = bias_variable([self.num_users], 'U_bias')
        self.V_bias = bias_variable([self.num_items], 'V_bias')

        self.U_W = weight_variable([self.num_items, self.item_rnn_dim], 'U_W')
        self.R_W = weight_variable([5, self.rating_rnn_dim], 'R_W')

        self.U_embed = tf.nn.embedding_lookup(self.U, self.u_idx)
        self.V_embed = tf.nn.embedding_lookup(self.V, self.i_idx)
        self.U_bias_embed = tf.nn.embedding_lookup(self.U_bias, self.u_idx)
        self.V_bias_embed = tf.nn.embedding_lookup(self.V_bias, self.i_idx)
        self.i_hist_emb = tf.nn.embedding_lookup(self.U_W, self.i_hist)
        self.i_next_emb = tf.nn.embedding_lookup(self.U_W, self.i_idx)
        self.r_hist_emb = tf.nn.embedding_lookup(self.R_W, self.r_hist)
        '''
        self.U_final = weight_variable([self.num_users, 2 * self.rnn_dim * self.latent_dim], 'U_final')
        self.U_final_bias = weight_variable([self.num_items, self.latent_dim], 'U_final_bias')

        self.U_final_embed = tf.nn.embedding_lookup(self.U_final, self.u_idx)
        self.U_final_embed = tf.reshape(self.U_final_embed, [-1, self.latent_dim, 2 * self.rnn_dim])
        self.U_final_bias_embed = tf.nn.embedding_lookup(self.U_final_bias, self.u_idx)

        self.V_t = weight_variable([self.num_items, self.latent_dim], 'V_t')
        self.V_t_embed = tf.nn.embedding_lookup(self.V_t, self.i_idx)'''

        self.U_final = weight_variable([self.rnn_dim, self.latent_dim], 'U_final')
        self.U_final_bias = bias_variable([self.latent_dim], 'U_final_bias')

        self.V_t = weight_variable([self.latent_dim], 'V_t')
        concat_emb = tf.concat([self.i_hist_emb, self.r_hist_emb], 2)
        with tf.name_scope("rnn"):

            cell_fw = tf.contrib.rnn.DropoutWrapper(build_cell(self.rnn_dim),
                                                    variational_recurrent=True,
                                                    input_keep_prob=self.dropout_p,
                                                    output_keep_prob=self.dropout_p,
                                                    state_keep_prob=self.dropout_p,
                                                    input_size=(self.rnn_dim),
                                                    dtype=tf.float32)
            '''cell_bw = tf.contrib.rnn.DropoutWrapper(build_cell(2 * self.rnn_dim),
                                                    variational_recurrent=True,
                                                    input_keep_prob=0.95,
                                                    output_keep_prob=0.95,
                                                    state_keep_prob=0.95,
                                                    input_size=(2 * self.rnn_dim),
                                                    dtype=tf.float32)'''
            rnn_output, _ = tf.nn.dynamic_rnn(cell_fw,
                                              concat_emb, self.sl,
                                              dtype=tf.float32)
            rnn_output = extract_axis_1(rnn_output, self.sl-1)
        #print(rnn_output)
        #print(self.U_final_embed)
        '''u_affine = tf.tensordot(rnn_output, self.U_final_embed, axes=[[0, 1], [0, 2]])\
            + self.U_final_bias_embed
        print(u_affine)
        u_affine = tf.reshape(u_affine, [-1, self.latent_dim])
        t_contrib = tf.reduce_sum(tf.multiply(u_affine, self.V_t_embed), reduction_indices=1)'''
        rnn_concat = tf.concat([rnn_output, self.i_next_emb], 1)
        u_affine = tf.layers.dense(rnn_concat, self.latent_dim, activation = tf.nn.relu)
        #u_affine = tf.nn.relu(tf.matmul(rnn_output, self.U_final) + self.U_final_bias)
        t_contrib = tf.reduce_sum(tf.multiply(u_affine, self.V_t), reduction_indices=1)
        s_contrib = tf.reduce_sum(tf.multiply(self.U_embed, self.V_embed), reduction_indices=1)
        s_contrib = tf.add(s_contrib, self.U_bias_embed)
        s_contrib = tf.add(s_contrib, self.V_bias_embed)
        self.r_hat = tf.add(t_contrib, s_contrib) + offset
        #self.r_hat = t_contrib + offset

        self.RMSE = tf.sqrt(tf.losses.mean_squared_error(self.r, self.r_hat))
        self.cost = 2 * tf.nn.l2_loss(tf.subtract(self.r, self.r_hat))
        self.MAE = tf.reduce_mean(tf.abs(tf.subtract(self.r, self.r_hat)))
        self.reg = tf.add(tf.multiply(self.reg_lambda, tf.nn.l2_loss(self.U)), tf.multiply(self.reg_lambda, tf.nn.l2_loss(self.V)))
        self.reg_loss = tf.add(self.cost, self.reg)
        #self.reg_loss = self.cost

        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        # self.train_step = self.optimizer.minimize(self.reg_loss)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # Needed for correct batch norm usage
        with tf.control_dependencies(update_ops):
            self.train_step = self.optimizer.minimize(self.reg_loss,
                                                        colocate_gradients_with_ops=True)

        tf.summary.scalar("RMSE", self.RMSE)
        tf.summary.scalar("MAE", self.MAE)
        tf.summary.scalar("Cost", self.cost)
        tf.summary.scalar("Reg-Loss", self.reg_loss)

        # add op for merging summary
        self.summary_op = tf.summary.merge_all()

        # add Saver ops
        self.saver = tf.train.Saver()

        return self.RMSE, self.MAE, self.cost, self.summary_op,\
            self.train_step

    def train(self, sess, data, dropout):
        RMSE, MAE, cost, summary_op,\
          train_step = sess.run([self.RMSE, self.MAE, self.cost, self.summary_op,
                           self.train_step], feed_dict={
                           self.u_idx: data[0],
                           self.i_idx: data[1],
                           self.r: data[2],
                           self.i_hist: data[3],
                           self.r_hist: data[4],
                           self.sl: data[5],
                           self.dropout_p: dropout
                           })
        return RMSE, MAE, cost, summary_op

    def eval(self, sess, data, dropout=1.0):
        RMSE, MAE, cost, summary_op,\
           = sess.run([self.RMSE, self.MAE, self.cost, self.summary_op],
                      feed_dict={
                           self.u_idx: data[0],
                           self.i_idx: data[1],
                           self.r: data[2],
                           self.i_hist: data[3],
                           self.r_hist: data[4],
                           self.sl: data[5],
                           self.dropout_p: dropout
                           })
        return RMSE, MAE, cost, summary_op


class AttnRNNSMF:
    def __init__(self, num_users, num_items, latent_dim,
                 learning_rate=0.001, reg_lambda=0.01,
                 dropout_p_hidden=0.8, item_rnn_dim=10, rating_rnn_dim=2):
        self.num_users = num_users
        self.num_items = num_items
        self.latent_dim = latent_dim
        self.item_rnn_dim = item_rnn_dim
        self.rating_rnn_dim = rating_rnn_dim
        self.rnn_dim = item_rnn_dim + rating_rnn_dim
        self.learning_rate = learning_rate
        self.reg_lambda = tf.constant(reg_lambda, dtype=tf.float32)


        self.dropout_p = tf.placeholder(tf.float32, [])
        self.batch_size = tf.placeholder(tf.int32, [])
        self.u_idx = tf.placeholder(tf.int32, [None, ])  # [B]
        self.i_idx = tf.placeholder(tf.int32, [None, ])  # [B]
        self.r = tf.placeholder(tf.float32, [None, ])  # [B]
        self.i_hist = tf.placeholder(tf.int32, [None, None])  # [B, T]
        self.r_hist = tf.placeholder(tf.int32, [None, None])  # [B, T]
        self.sl = tf.placeholder(tf.int32, [None, ])  # [B]
        self.lr = tf.placeholder(tf.float64, [])

    def build_graph(self, offset):

        self.U = weight_variable([self.num_users, self.latent_dim], 'U')
        self.V = weight_variable([self.num_items, self.latent_dim], 'V')
        self.U_bias = bias_variable([self.num_users], 'U_bias')
        self.V_bias = bias_variable([self.num_items], 'V_bias')
        self.rating_const = bias_variable([self.rating_rnn_dim], 'rating_const')
        #self.rating_const = tf.expand_dims(self.rating_const, 0) # [B, 1, T]
        #self.rating_const = tf.tile(self.rating_const, self.batch_size)
        #self.rating_const = tf.reshape(self.rating_const, [tf.shape(self.i_hist)[0], self.rating_rnn_dim])
        #print(self.rating_const)
        #print(tf.shape(self.i_hist)[0])

        self.U_W = weight_variable([self.num_items, self.item_rnn_dim], 'U_W')
        self.R_W = weight_variable([5, self.rating_rnn_dim], 'R_W')

        self.U_embed = tf.nn.embedding_lookup(self.U, self.u_idx)
        self.V_embed = tf.nn.embedding_lookup(self.V, self.i_idx)
        self.U_bias_embed = tf.nn.embedding_lookup(self.U_bias, self.u_idx)
        self.V_bias_embed = tf.nn.embedding_lookup(self.V_bias, self.i_idx)
        self.i_next_emb = tf.nn.embedding_lookup(self.U_W, self.i_idx)
        paddings = tf.constant([[0, 0,], [0, 2]])
        self.i_next_emb = tf.pad(self.i_next_emb, paddings)
        #self.i_next_emb = tf.concat([self.i_next_emb, self.rating_const], 1)
        #self.i_next_emb = tf.reshape(self.i_next_emb, [-1])
        '''self.i_next_emb = tf.concat([self.i_next_emb, self.rating_const], 1)
        self.i_next_emb = tf.reshape(self.i_next_emb, [-1, self.rnn_dim])'''
        print(self.i_next_emb)
        self.i_hist_emb = tf.nn.embedding_lookup(self.U_W, self.i_hist)
        self.r_hist_emb = tf.nn.embedding_lookup(self.R_W, self.r_hist)

        self.U_final = weight_variable([self.rnn_dim, self.latent_dim], 'U_final')
        self.U_final_bias = bias_variable([self.latent_dim], 'U_final_bias')

        self.V_t = weight_variable([self.latent_dim], 'V_t')
        concat_emb = tf.concat([self.i_hist_emb, self.r_hist_emb], 2)
        with tf.name_scope("rnn"):

            cell_fw = tf.contrib.rnn.DropoutWrapper(build_cell(self.rnn_dim),
                                                    variational_recurrent=True,
                                                    input_keep_prob=self.dropout_p,
                                                    output_keep_prob=self.dropout_p,
                                                    state_keep_prob=self.dropout_p,
                                                    input_size=(self.rnn_dim),
                                                    dtype=tf.float32)

            rnn_output, _ = tf.nn.dynamic_rnn(cell_fw,
                                              concat_emb, self.sl,
                                              dtype=tf.float32)
            #rnn_output = extract_axis_1(rnn_output, self.sl-1)
        print(rnn_output) 

        print(self.i_next_emb)
        rnn_output = vanilla_attention(self.i_next_emb, rnn_output, self.sl)
        rnn_output = tf.reshape(rnn_output, [-1, self.rnn_dim])
        u_affine = tf.nn.relu(tf.matmul(rnn_output, self.U_final) + self.U_final_bias)
        t_contrib = tf.reduce_sum(tf.multiply(u_affine, self.V_t), reduction_indices=1)
        s_contrib = tf.reduce_sum(tf.multiply(self.U_embed, self.V_embed), reduction_indices=1)
        s_contrib = tf.add(s_contrib, self.U_bias_embed)
        s_contrib = tf.add(s_contrib, self.V_bias_embed)
        self.r_hat = tf.add(t_contrib, s_contrib) + offset
        #self.r_hat = t_contrib + offset

        self.RMSE = tf.sqrt(tf.losses.mean_squared_error(self.r, self.r_hat))
        self.cost = 2 * tf.nn.l2_loss(tf.subtract(self.r, self.r_hat))
        self.MAE = tf.reduce_mean(tf.abs(tf.subtract(self.r, self.r_hat)))
        self.reg = tf.add(tf.multiply(self.reg_lambda, tf.nn.l2_loss(self.U)), tf.multiply(self.reg_lambda, tf.nn.l2_loss(self.V)))
        self.reg_loss = tf.add(self.cost, self.reg)
        #self.reg_loss = self.cost

        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        # self.train_step = self.optimizer.minimize(self.reg_loss)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # Needed for correct batch norm usage
        with tf.control_dependencies(update_ops):
            self.train_step = self.optimizer.minimize(self.reg_loss,
                                                        colocate_gradients_with_ops=True)

        tf.summary.scalar("RMSE", self.RMSE)
        tf.summary.scalar("MAE", self.MAE)
        tf.summary.scalar("Cost", self.cost)
        tf.summary.scalar("Reg-Loss", self.reg_loss)

        # add op for merging summary
        self.summary_op = tf.summary.merge_all()

        # add Saver ops
        self.saver = tf.train.Saver()

        return self.RMSE, self.MAE, self.cost, self.summary_op,\
            self.train_step

    def train(self, sess, data, dropout):
        RMSE, MAE, cost, summary_op,\
          train_step = sess.run([self.RMSE, self.MAE, self.cost, self.summary_op,
                           self.train_step], feed_dict={
                           self.u_idx: data[0],
                           self.i_idx: data[1],
                           self.r: data[2],
                           self.i_hist: data[3],
                           self.r_hist: data[4],
                           self.sl: data[5],
                           self.dropout_p: dropout
                           })
        return RMSE, MAE, cost, summary_op

    def eval(self, sess, data, dropout=1.0):
        RMSE, MAE, cost, summary_op,\
           = sess.run([self.RMSE, self.MAE, self.cost, self.summary_op],
                      feed_dict={
                           self.u_idx: data[0],
                           self.i_idx: data[1],
                           self.r: data[2],
                           self.i_hist: data[3],
                           self.r_hist: data[4],
                           self.sl: data[5],
                           self.dropout_p: dropout
                           })
        return RMSE, MAE, cost, summary_op
