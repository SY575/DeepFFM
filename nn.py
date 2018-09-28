# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 20:33:07 2018

@author: SY
"""

import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score
from time import time
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
import config as cfg
from tqdm import tqdm

class sFFM(BaseEstimator, TransformerMixin):
    def __init__(self, field_sizes, total_feature_sizes, new_feature_size=0,
                 dynamic_max_len=30, extern_lr_size=0, extern_lr_feature_size=0,
                 embedding_size=8, dropout_fm=[1.0, 1.0],
                 deep_layers=[128], dropout_deep=[1.0, 1.0],obj='binary', 
                 val_batch_size=128,
                 deep_layers_activation=tf.nn.relu,
                 epoch=10, batch_size=256,
                 learning_rate=0.001, optimizer_type="adam",
                 batch_norm=1, batch_norm_decay=0.995,
                 verbose=True, random_seed=465,
                 loss_type="logloss", eval_metric=roc_auc_score,
                 l2_reg=0.0, greater_is_better=True):
        assert loss_type in ["logloss", "mse"], \
            "loss_type can be either 'logloss' for classification task or 'mse' for regression task"
        
        self.field_sizes = field_sizes
        self.total_field_size = sum(field_sizes)
        self.total_feature_sizes = total_feature_sizes
        self.embedding_size = embedding_size
        self.extern_lr_size = extern_lr_size
        self.extern_lr_feature_size = extern_lr_feature_size

        self.dropout_fm = dropout_fm
        self.deep_layers = deep_layers

        self.dropout_deep = dropout_deep
        self.deep_layers_activation = deep_layers_activation

        self.epoch = epoch
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type

        self.batch_norm = batch_norm
        self.batch_norm_decay = batch_norm_decay

        self.verbose = verbose
        self.random_seed = random_seed
        self.loss_type = loss_type
        self.eval_metric = eval_metric
        self.greater_is_better = greater_is_better
        self.train_result, self.valid_result = [], []
        
        self.obj = obj
        
        self._init_graph()


    def _init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(self.random_seed)
            '''placeholder'''
            self.label = tf.placeholder(tf.float32, shape=[None, cfg.NUM_CLASS], name="label")
            self.static_index = tf.placeholder(tf.int32, shape=[None, len(cfg.NUMERIC_COL)+len(cfg.ONEHOT_COL)], name="static_index")
            if len(cfg.VECTOR_COL) != 0:
                self.dynamic_index = tf.placeholder(tf.int32, shape=[None, len(cfg.VECTOR_COL), cfg.MAX_LEN], name="dynamic_index")
                self.dynamic_len = tf.placeholder(tf.int32, shape=[None, len(cfg.VECTOR_COL)], name="dynamic_len")
            if len(cfg.LR_COL) != 0:
                self.lr = tf.placeholder(tf.float32, shape=[None, len(cfg.LR_COL)], name="lr")
            self.dropout_keep_fm = tf.placeholder(tf.float32, shape=[None], name="dropout_keep_fm")
            self.dropout_keep_deep = tf.placeholder(tf.float32, shape=[None], name="dropout_keep_deep")
            self.train_phase = tf.placeholder(tf.bool, name="train_phase")

            self.weights = self._initialize_weights()
            
            # static
            self.static_ffm_embs = tf.nn.embedding_lookup(self.weights["static_ffm_embeddings"],
                                                          self.static_index) # None * static_feature_size * [k * F]
            # dynamic
            if len(cfg.VECTOR_COL) != 0:
                self.dynamic_ffm_embs = tf.nn.embedding_lookup(self.weights["dynamic_ffm_embeddings"],
                                                              self.dynamic_index) # None * [dynamic_feature_size * max_len] * [k * F]
                self.ffm_mask = tf.sequence_mask(tf.reshape(self.dynamic_len,[-1]), maxlen= cfg.MAX_LEN) # [None * dynamic_feature] * max_len
                self.ffm_mask = tf.expand_dims(self.ffm_mask, axis=-1) # [None * dynamic_feature] * max_len * 1
                self.ffm_mask = tf.concat([self.ffm_mask for i in range(self.embedding_size * self.total_field_size)], axis = -1) # [None * dynamic_feature] * max_len * [k * F]
                self.dynamic_ffm_embs = tf.reshape(self.dynamic_ffm_embs,[-1, cfg.MAX_LEN, self.embedding_size * self.total_field_size]) # [None * dynamic_feature] * max_len * [k * F]
                self.dynamic_ffm_embs = tf.multiply(self.dynamic_ffm_embs, tf.to_float(self.ffm_mask)) # [None * dynamic_feature] * max_len * [k * F]
                self.dynamic_ffm_embs = tf.reshape(tf.reduce_sum(self.dynamic_ffm_embs, axis=1),[-1, self.field_sizes[1],
                                                                                                 self.embedding_size * self.total_field_size]) # None * dynamic_feature_size * [k * F]
                self.padding_lengths = tf.concat([tf.expand_dims(self.dynamic_len, axis=-1)
                                                  for i in range(self.embedding_size * self.total_field_size)],axis=-1) # None * dynamic_feature_size * [k * F]
                self.dynamic_ffm_embs = tf.div(self.dynamic_ffm_embs, tf.to_float(self.padding_lengths)) # None * dynamic_feature_size * [k * F]
                # concat
                self.ffm_embs = tf.concat([self.static_ffm_embs, self.dynamic_ffm_embs], axis=1)
            else:
                self.ffm_embs = self.static_ffm_embs
            
            # 矩阵乘
            self.ffm_embs_col = tf.reshape(self.ffm_embs,
                                           [-1, self.total_field_size, self.total_field_size, self.embedding_size]) # None * F * F * k
            self.ffm_embs_row = tf.transpose(self.ffm_embs_col, [0, 2, 1, 3]) # None * F * F * k
            self.ffm_embs_out = tf.multiply(self.ffm_embs_col, self.ffm_embs_row) # None *F * F * k
            self.ones = tf.ones_like(self.ffm_embs_out)
            self.op = tf.contrib.linalg.LinearOperatorTriL(tf.transpose(self.ones,[0,3,1,2])) # None *k * F *F
            self.upper_tri_mask = tf.less(tf.transpose(self.op.to_dense(), [0,2,3,1]), self.ones) # None *F * F * k

            self.ffm_embs_out = tf.boolean_mask(self.ffm_embs_out, self.upper_tri_mask) # [None * F * (F-1) * k]
            self.ffm_embs_out = tf.reshape(self.ffm_embs_out, [-1, self.total_field_size * (self.total_field_size-1) // 2
                                                              * self.embedding_size]) # None * [F * (F-1) / 2 * k]
    
            self.y_deep = self.ffm_embs_out #tf.reshape(self.ffm_embs_col,[-1, self.total_field_size * self.total_field_size * self.embedding_size])
            # lr
            if len(cfg.LR_COL) != 0:
                self.lr_emb = tf.multiply(self.lr, self.weights["lr_embeddings"])
                self.y_deep = tf.concat([self.y_deep, self.lr_emb], axis=1)
    
            self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_keep_deep[0])
            for i in range(0, len(self.deep_layers)):
                self.y_deep = tf.matmul(self.y_deep, self.weights["layer_%d" % i])
                #self.y_deep = tf.add(tf.matmul(self.y_deep, self.weights["layer_%d" %i]), self.weights["bias_%d"%i]) # None * layer[i] * 1
                if self.batch_norm:
                    self.y_deep = self.batch_norm_layer(self.y_deep, train_phase=self.train_phase, scope_bn="bn_%d" %i) # None * layer[i] * 1
                self.y_deep = self.deep_layers_activation(self.y_deep)
                self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_keep_deep[1+i]) # dropout at each Deep layer
            self.y_deep = tf.layers.dense(inputs=self.y_deep, units=self.deep_layers[-1], activation=self.deep_layers_activation)
            
            self.out = tf.add(tf.matmul(self.y_deep, self.weights["concat_projection"]), self.weights["concat_bias"])
            self.out = tf.reshape(self.out, shape=[-1, cfg.NUM_CLASS])
            self.label = tf.reshape(self.label, shape=[-1, cfg.NUM_CLASS])

            # loss
            if self.loss_type == "logloss":
                self.out = tf.nn.sigmoid(self.out)
                self.loss = tf.losses.log_loss(self.label, self.out)
            elif self.loss_type == "mse":
                self.loss = tf.nn.l2_loss(tf.subtract(self.label, self.out))

            # optimizer
            if self.optimizer_type == "adam":
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,
                                                        epsilon=1e-8).minimize(self.loss)
            elif self.optimizer_type == "adagrad":
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                           initial_accumulator_value=1e-8).minimize(self.loss)
            elif self.optimizer_type == "gd":
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            elif self.optimizer_type == "momentum":
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95).minimize(
                    self.loss)
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess = self._init_session()
            self.sess.run(init)

            total_parameters = 0
            for variable in self.weights.values():
                shape = variable.get_shape()
                variable_parameters = 1
                for dim in shape:
                    variable_parameters *= dim.value
                total_parameters += variable_parameters
            if self.verbose > 0:
                print("#params: %d" % total_parameters)


    def _init_session(self):
        config = tf.ConfigProto(device_count={"gpu": 0})
        return tf.Session(config=config)


    def _initialize_weights(self):
        weights = dict()
        # FFM embeddings
        input_size = self.embedding_size * self.total_field_size
        glorot = 0.0001#np.sqrt(2.0 / (input_size * self.total_feature_sizes))
        weights["static_ffm_embeddings"] = tf.Variable(
            tf.random_normal([self.total_feature_sizes[0], input_size], 0.0, glorot),
            name="static_ffm_embeddings")  # static_feature_size * [K * F]
        if len(cfg.VECTOR_COL) != 0:
            weights["dynamic_ffm_embeddings"] = tf.Variable(
                tf.random_normal([self.total_feature_sizes[1], input_size], 0.0, glorot),
                name="dynamic_ffm_embeddings")  # dynamic_feature_size * [K * F]
        if len(cfg.LR_COL) != 0:
            weights["lr_embeddings"] = tf.Variable(
                tf.random_normal([len(cfg.LR_COL)], 0.0, glorot),
                name="lr_embeddings")
        # deep layers
        num_layer = len(self.deep_layers)
        input_size = self.total_field_size * (self.total_field_size -1) // 2 * self.embedding_size + len(cfg.LR_COL)
        glorot = np.sqrt(2.0 / (input_size))
        weights["layer_0"] = tf.Variable(
            np.random.normal(loc=0, scale=glorot, size=(input_size, self.deep_layers[0])), dtype=np.float32)
        weights["bias_0"] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[0])),
                                                        dtype=np.float32)  # 1 * layers[0]
        for i in range(1, num_layer):
            glorot = np.sqrt(2.0 / (self.deep_layers[i-1]))
            weights["layer_%d" % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(self.deep_layers[i-1], self.deep_layers[i])),
                dtype=np.float32)  # layers[i-1] * layers[i]

        # final concat projection layer
        input_size = self.deep_layers[-1]
        if self.extern_lr_size:
            input_size += self.extern_lr_feature_size
        glorot = np.sqrt(2.0 / (input_size + 1))
        weights["concat_projection"] = tf.Variable(
                        np.random.normal(loc=0, scale=glorot, size=(input_size, cfg.NUM_CLASS)),
                        dtype=np.float32)  # layers[i-1]*layers[i]
        weights["concat_bias"] = tf.Variable(tf.constant(-3.5), dtype=np.float32)

        return weights


    def batch_norm_layer(self, x, train_phase, scope_bn):
        bn_train = batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True, updates_collections=None,
                              is_training=True, reuse=None, trainable=True, scope=scope_bn)
        bn_inference = batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True, updates_collections=None,
                                  is_training=False, reuse=True, trainable=True, scope=scope_bn)
        z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
        return z


    
    def feed_data(self, 
                  train_static, train_dynamic, train_lr, train_y, 
                  val_static, val_dynamic, val_lr, val_y, 
                  test_static, test_dynamic, test_lr, test_y):
        
        self.train_static = train_static.values
        self.val_static = val_static.values
        self.test_static = test_static.values
        
        self.train_lr = train_lr
        self.val_lr = val_lr
        self.test_lr = test_lr
        
        
        self.train_y = train_y.values
        self.val_y = val_y.values
        self.test_y = test_y.values
        
        self.train_dynamic_len = train_dynamic.copy()
        self.val_dynamic_len   = val_dynamic.copy()
        self.test_dynamic_len  = test_dynamic.copy()
        for df in [self.train_dynamic_len, self.val_dynamic_len, self.test_dynamic_len]:
            for col in cfg.VECTOR_COL:
                df[col] = df[col].apply(lambda x :len(x))
        self.train_dynamic_len = self.train_dynamic_len.values
        self.val_dynamic_len = self.val_dynamic_len.values
        self.test_dynamic_len = self.test_dynamic_len.values
        
        
        self.train_dynamic = train_dynamic.copy()
        self.val_dynamic = val_dynamic.copy()
        self.test_dynamic = test_dynamic.copy()
        for df in [self.train_dynamic, self.val_dynamic, self.test_dynamic]:
            for col in cfg.VECTOR_COL:
                df[col] = df[col].apply(lambda x:[x[i] if i < len(x) else 0 for i in range(cfg.MAX_LEN)])
        self.train_dynamic = self.train_dynamic.values
        self.val_dynamic = self.val_dynamic.values
        self.test_dynamic = self.test_dynamic.values
        
    

    def fit(self,
            early_stopping=False, refit=False):
        print('======== training ========')
        for epoch in range(self.epoch):
            t1 = time()
            label_lst, out_lst = None, None
            label_e, out_e = None, None
            for ix, batch_ix in enumerate(range(0, self.len_train, self.batch_size)):
                batch_static, batch_dynamic, batch_dynamic_len, batch_lr, batch_y = self.get_train_batch(batch_ix)
                batch_static = np.array(batch_static)
                batch_dynamic = np.array(batch_dynamic.tolist())
                batch_dynamic_len = np.array(batch_dynamic_len)
                batch_lr = np.array(batch_lr)
                batch_y = np.array(batch_y.tolist())
                
                f_dict = {self.label:batch_y,
                          self.static_index: batch_static, 
                          self.dropout_keep_fm:self.dropout_fm,
                          self.dropout_keep_deep:self.dropout_deep,
                          self.train_phase: True}
                if len(cfg.VECTOR_COL) != 0:
                    f_dict.update({self.dynamic_index:batch_dynamic,
                                   self.dynamic_len:batch_dynamic_len})
                if len(cfg.LR_COL) != 0:
                    f_dict.update({self.lr:batch_lr})
                    
                loss_, _ ,label_, out_= self.sess.run((self.loss, self.optimizer, self.label, self.out),
                                                      feed_dict=f_dict)
                if label_e is None:
                    label_e = label_
                    out_e = out_
                else:
                    label_e = np.concatenate((label_e, label_), axis=0)
                    out_e = np.concatenate((out_e, out_), axis=0)
                if label_lst is None:
                    label_lst = label_
                    out_lst = out_
                else:
                    label_lst = np.concatenate((label_lst, label_), axis=0)
                    out_lst = np.concatenate((out_lst, out_), axis=0)
                if cfg.is_print and ix % 100 == 99:
                    print('[%d/%d] AUC = %.4f %.4f [ %ds ]'
                          %(ix+1, self.len_train//self.batch_size, 
                            self.eval_metric(label_lst, out_lst), 
                            self.eval_metric(label_e, out_e), 
                            int(time()-t1)))
                    label_e, out_e = None, None
            if cfg.is_print:
                print('----- predict val -----')
                valid_result = self.evaluate()
                print("[%d] valid-result=%.4f [ %d s ]"
                      %(epoch + 1, valid_result, int(time() - t1)))
            break
        
    def get_train_batch(self, _i):
        if _i + self.batch_size > self.len_train:
            end = self.len_train
        else:
            end = _i + self.batch_size
        return (self.train_static[_i:end], 
                self.train_dynamic[_i:end], 
                self.train_dynamic_len[_i:end], 
                self.train_lr[_i:end], 
                self.train_y[_i:end])
        
    def get_val_batch(self, _i):
        if _i + self.val_batch_size > self.len_val:
            end = self.len_val
        else:
            end = _i + self.val_batch_size
        return (self.val_static[_i:end], 
                self.val_dynamic[_i:end], 
                self.val_dynamic_len[_i:end], 
                self.val_lr[_i:end], 
                self.val_y[_i:end])
    
    def get_test_batch(self, _i):
        if _i + self.val_batch_size > self.len_test:
            end = self.len_test
        else:
            end = _i + self.val_batch_size
        return (self.test_static[_i:end], 
                self.test_dynamic[_i:end], 
                self.test_dynamic_len[_i:end], 
                self.test_lr[_i:end], 
                self.test_y[_i:end])
    

    def evaluate(self):
        y_pred, y = None, None
        for batch_ix in tqdm(list(range(0, self.len_val, self.val_batch_size))):
            batch_static, batch_dynamic, batch_dynamic_len, batch_lr, batch_y = self.get_val_batch(batch_ix)
            batch_static = np.array(batch_static)
            batch_dynamic = np.array(batch_dynamic.tolist())
            batch_dynamic_len = np.array(batch_dynamic_len)
            batch_lr = np.array(batch_lr)
            batch_y = np.array(batch_y.tolist())
            
            f_dict = {self.label:batch_y,
                      self.static_index: batch_static, 
                      self.dropout_keep_fm:self.dropout_fm,
                      self.dropout_keep_deep:self.dropout_deep,
                      self.train_phase: True}
            if len(cfg.VECTOR_COL) != 0:
                f_dict.update({self.dynamic_index:batch_dynamic,
                               self.dynamic_len:batch_dynamic_len})
            if len(cfg.LR_COL) != 0:
                f_dict.update({self.lr:batch_lr})
            label_, out_ = self.sess.run([self.label, self.out],
                                         feed_dict=f_dict)
            if y_pred is None:
                y_pred = out_
                y = label_
            else:
                y_pred = np.concatenate((y_pred, out_), axis=0)
                y = np.concatenate((y, label_), axis=0)
        self.val_pred = y_pred # [:self.len_val]
        return self.eval_metric(y, y_pred)


    def get_result(self):
        y_pred = None
        for batch_ix in range(0, self.len_test, self.val_batch_size):
            batch_static, batch_dynamic, batch_dynamic_len, batch_lr, batch_y = self.get_test_batch(batch_ix)
            batch_static = np.array(batch_static)
            batch_dynamic = np.array(batch_dynamic.tolist())
            batch_dynamic_len = np.array(batch_dynamic_len)
            batch_lr = np.array(batch_lr)
            batch_y = np.array(batch_y.tolist())
            
            f_dict = {self.label:batch_y,
                      self.static_index: batch_static, 
                      self.dropout_keep_fm:self.dropout_fm,
                      self.dropout_keep_deep:self.dropout_deep,
                      self.train_phase: True}
            if len(cfg.VECTOR_COL) != 0:
                f_dict.update({self.dynamic_index:batch_dynamic,
                               self.dynamic_len:batch_dynamic_len})
            if len(cfg.LR_COL) != 0:
                f_dict.update({self.lr:batch_lr})
            out_ = self.sess.run(self.out,
                                 feed_dict=f_dict)
            if y_pred is None:
                y_pred = out_
            else:
                y_pred = np.concatenate((y_pred, out_), axis=0)
        return y_pred
