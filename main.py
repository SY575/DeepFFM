# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 21:11:26 2018

@author: SY
"""

import warnings
warnings.filterwarnings("ignore")
import sys
import tensorflow as tf
import pandas as pd
import numpy as np
import config
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import train_test_split
sys.path.append(".")
from nn import sFFM
from tqdm import tqdm
import math
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from sklearn.model_selection import StratifiedKFold


def to_bins(df_, bins=config.N_BINS):
    try:
        MAX = df_.max()
        MIN = df_.min()
        df_ = df_ - MIN
        df_ = df_ // ((MAX - MIN)/bins)
        df_ = df_.astype(np.int16)
        return df_
    except:
        df_ = df_.fillna(-999)
        df_ = df_.apply(lambda x:-999 if math.isinf(x) else x)
        MAX = df_.max()
        MIN = df_.min()
        df_ = df_ - MIN
        df_ = df_ // ((MAX - MIN)/bins)
        df_ = df_.astype(np.int16)
        return df_

def normalization(_df, n_features):
    _lst = list(set(_df))
    _c = {}
    for i in _lst:
        _c[i] = n_features
        n_features += 1
    _df = _df.apply(lambda x:_c[x])
    return _df, n_features

def load_data():
    if 'dealed_data' not in os.listdir('./'):
        os.makedirs('dealed_data')
    if 'data.csv' in os.listdir('./dealed_data'):
        data = pd.read_csv('./dealed_data/data.csv', nrows=None)
        n_features = [np.sum([len(set(data[col])) for col in config.NUMERIC_COL + config.ONEHOT_COL]),
                      np.sum([len(set(' '.join(data[col].apply(lambda x:x.replace(',', ' '))).split(' '))) 
                      for col in config.VECTOR_COL])]
    else:
        data = pd.read_csv(config.TRAIN_PATH)
        test = pd.read_csv(config.TEST_PATH)
        test[config.LABEL_NAME] = -1
        data = data.append(test).reset_index(drop=True)
        print('len(data) = %d, len(data.columns) = %d'
              %(len(data), len(data.columns)))
        
        n_features = [0, 0]
        '''LR_COL'''
        if len(config.LR) != 0:
            for col in config.LR:
                data[col+'_lr'] = data[col]
        '''CROSS_COL'''
        if len(config.CROSS_COL) != 0:
            for cols in config.CROSS_COL:
                data[cols[0]+'_'+cols[1]] = data[cols[0]] * data[cols[1]]
        '''NUMERIC_COL'''
        if len(config.NUMERIC_COL) != 0:
            '''deal extream values'''
            for col in tqdm(config.NUMERIC_COL):
                data[col] = data[col].astype(float)
                _mean = np.mean(data[col])
                _percentile_99 = np.percentile(data[col], 0.99)
                _percentile_01 = np.percentile(data[col], 0.01)
                if _percentile_99 > 100 * _mean:
                    data[col].loc[data[col] > _percentile_99] = _percentile_99
                if _percentile_01 < 0.01 * _mean:
                    data[col].loc[data[col] < _percentile_01] = _percentile_01
            '''to bins, normalization'''
            data = data.fillna(-1)
            for col in tqdm(config.NUMERIC_COL):
                if len(set(data[col])) > 20:
                    data[col] = to_bins(data[col])
                data[col], n_features[0] = normalization(data[col], n_features[0])
                data[col] = data[col].astype(np.int32)
        '''ONEHOT_COL'''
        if len(config.ONEHOT_COL) != 0:
            for col in tqdm(config.ONEHOT_COL):
                data[col] = data[col].fillna(-1)
                lst = list(set(data[col]))
                temp_dict = {}
                for i, j in enumerate(lst):
                    temp_dict[j] = n_features[0] + i + 1
                n_features[0] += len(lst)
                data[col] = data[col].apply(lambda x:temp_dict[x])
        '''VECTOR_COL'''
        if len(config.VECTOR_COL) != 0:
            for col in tqdm(config.VECTOR_COL):
                data[col] = data[col].fillna('-1')
                data[col] = data[col].apply(lambda x:x.replace(',', ' '))
                lst = list(set(' '.join(data[col]).split(' ')))
                temp_dict = {}
                for i, j in enumerate(lst):
                    temp_dict[j] = n_features[1] + i + 1
                n_features[1] += len(lst)
                data[col] = data[col].apply(lambda x:' '.join([str(temp_dict[j]) for j in x.split(' ')]))
        data[config.LABEL_NAME] = data[config.LABEL_NAME].astype(np.float32)
        
        data = data.sample(frac=1)
        data.to_csv('./dealed_data/data.csv', index=False)
    
    if len(config.VECTOR_COL) != 0:
        for col in config.VECTOR_COL:
            data[col] = data[col].apply(lambda x:[int(i) for i in x.split(' ')])
    train = data.loc[data[config.LABEL_NAME]!=-1].reset_index(drop=True)
    train_label = train[config.LABEL_NAME].tolist()
    train[config.LABEL_NAME] = train[config.LABEL_NAME].astype(int).apply(lambda x:[1 if i == x else 0 for i in range(config.NUM_CLASS)])
    test = data.loc[data[config.LABEL_NAME]==-1].reset_index(drop=True)
    test[config.LABEL_NAME] = test[config.LABEL_NAME].apply(lambda x:[0 for i in range(config.NUM_CLASS)])
    
    return (train[config.NUMERIC_COL+config.ONEHOT_COL+config.VECTOR_COL+config.LR_COL+[config.LABEL_NAME]], train_label, 
            test[ config.NUMERIC_COL+config.ONEHOT_COL+config.VECTOR_COL+config.LR_COL+[config.LABEL_NAME]], n_features)
    

def run_model(params):
    assert params['obj'] in ['binary', 'multi-logloss']
    print('load data...')
    train, train_label, test, n_features = load_data()
    print(len(train))
    params['total_feature_sizes'] = n_features
    
    print('***** DeepFFM *****')
    print(params)
    result_train = pd.DataFrame(index=train.index, columns=['pred'])
    result_train[config.LABEL_NAME] = train[config.LABEL_NAME]
    result_lst = []
    skf = StratifiedKFold(n_splits = config.fold, random_state=config.RANDOM_SEED, shuffle=True)
    for train_ix, val_ix in skf.split(train[config.NUMERIC_COL], train_label):
        train_static = train[config.NUMERIC_COL+config.ONEHOT_COL].loc[train_ix]
        val_static = train[config.NUMERIC_COL+config.ONEHOT_COL].loc[val_ix]
        
        train_dynamic = train[config.VECTOR_COL].loc[train_ix]
        val_dynamic = train[config.VECTOR_COL].loc[val_ix]
        
        train_lr = train[config.LR_COL].loc[train_ix]
        val_lr = train[config.LR_COL].loc[val_ix]
        
        train_y = train[config.LABEL_NAME].loc[train_ix]
        val_y = train[config.LABEL_NAME].loc[val_ix]
        
        sffm = sFFM(**params)
        sffm.len_train = len(train_y)
        sffm.len_val = len(val_y)
        sffm.len_test = len(test)
        
        sffm.feed_data(train_static, train_dynamic, train_lr, train_y, 
                       val_static, val_dynamic, val_lr, val_y, 
                       test[config.NUMERIC_COL+config.ONEHOT_COL], test[config.VECTOR_COL], 
                       test[config.LR_COL], test[config.LABEL_NAME])
        sffm.fit()
        result_train['pred'].loc[val_ix] = sffm.val_pred.tolist()
        result_lst.append(sffm.get_result())
    result_train[config.LABEL_NAME] = result_train[config.LABEL_NAME].apply(lambda x:x.index(1))
    result_train['pred'] = result_train['pred'].apply(lambda x:[i/sum(x) for i in x])
    for i in range(config.NUM_CLASS):
        result_train['prob'+str(i)] = np.array(result_train.pred.values.tolist())[:,i]
    try:
        print(log_loss(result_train[config.LABEL_NAME], result_train['pred'].values.tolist()))
    except:
        1
    return result_train, result_lst


params = {
    "field_sizes":[len(config.NUMERIC_COL)+len(config.ONEHOT_COL), len(config.VECTOR_COL)],
    'obj':'binary',
    "embedding_size": 4,
    "dropout_fm": [1., 1.],
    "deep_layers": [32, 16],
    "dropout_deep": [1., 1., 1., 1.],
    "deep_layers_activation": tf.nn.relu,
    "epoch": 2,
    "batch_size": 256,
    "val_batch_size":256,
    "learning_rate": 0.001,
    "optimizer_type": "adam",
    "batch_norm": 10,
    "batch_norm_decay": 0.995,
    "l2_reg": 0.,
    "verbose": True,
    "eval_metric": roc_auc_score,
    "random_seed": config.RANDOM_SEED
}

result_train, result_lst = run_model(params)

'''
from sklearn.metrics import f1_score
preds = np.argmax(np.array(result_train.pred.values.tolist()), axis=1)
print(f1_score(y_true=result_train.current_service, 
                   y_pred=preds, average='weighted'))
'''