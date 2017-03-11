import net

import numpy as np
import tensorflow as tf

class DataLoader(object):
    def __init__(
            self, n_batch, n_summary_features, n_const_features,
            max_constituents):
        self.n_batch = n_batch
        self.t_label = tf.placeholder(tf.int32, (n_batch,))
        self.t_summary_features = tf.placeholder(
                tf.float32, (n_batch, n_summary_features))
        self.t_const_features = tf.placeholder(
                tf.float32, (n_batch, max_constituents, n_const_features))
        self.t_const_len = tf.placeholder(tf.int32, (n_batch))
        self.random = np.random.RandomState(0)

    def load(self, data, sample=True):
        label = np.zeros(self.t_label.get_shape(), dtype=np.int32)
        summary_features = np.zeros(self.t_summary_features.get_shape())
        const_features = np.zeros(self.t_const_features.get_shape())
        const_len = np.zeros(self.t_const_len.get_shape(), dtype=np.int32)

        if sample:
            indices = self.random.randint(len(data), size=self.n_batch)
        else:
            assert len(data) == self.n_batch
            indices = list(range(self.n_batch))
        for i, i_d in enumerate(indices):
            datum = data[i_d]
            label[i] = datum.label
            summary_features[i, :] = datum.summary_features

            l = datum.const_features.shape[0]
            const_len[i] = l
            const_features[i, :l, :] = datum.const_features
        
        return {
            self.t_label: label,
            self.t_summary_features: summary_features,
            self.t_const_features: const_features,
            self.t_const_len: const_len
        }

class MlpModel(object):
    def __init__(self, layers, loader):
        self.t_scores, v_model = net.mlp(loader.t_summary_features, layers)
        self.t_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.t_scores, labels=loader.t_label))
        self.t_acc = tf.reduce_mean(tf.cast(
                tf.nn.in_top_k(self.t_scores, loader.t_label, 1), tf.float32))

        optimizer = tf.train.AdamOptimizer(0.00003)
        self.o_train = optimizer.minimize(self.t_loss, var_list=v_model)

class RnnModel(object):
    def __init__(self, n_hidden, n_labels, loader):
        cell = tf.contrib.rnn.GRUCell(n_hidden)
        t_drop = tf.nn.dropout(loader.t_const_features, 0.9)
        _, t_state = tf.nn.dynamic_rnn(
                cell, t_drop, loader.t_const_len, dtype=tf.float32)
        t_inputs = tf.concat((loader.t_summary_features, t_state), axis=1)
        self.t_scores, _ = net.mlp(t_inputs, (n_labels,))

        v_model = tf.get_collection(tf.GraphKeys.VARIABLES)

        self.t_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.t_scores, labels=loader.t_label))
        self.t_acc = tf.reduce_mean(tf.cast(
                tf.nn.in_top_k(self.t_scores, loader.t_label, 1), tf.float32))

        optimizer = tf.train.AdamOptimizer(0.0003)
        self.o_train = optimizer.minimize(self.t_loss, var_list=v_model)
