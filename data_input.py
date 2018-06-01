import tensorflow as tf
import pandas as pd
import numpy as np
# 超参配置文件
from config import cfg
from scipy import sparse


HASH_LEN = 3188
if cfg.data_source == 1:
    DATA = '_9_2162.npz'
    LGB_LEN = 2162
elif cfg.data_source == 2:
    DATA = '_5_4236.npz'
    LGB_LEN = 4236
elif cfg.data_source == 3:
    DATA = '_3_6789.npz'
    LGB_LEN = 6789
elif cfg.data_source == 4:
    DATA = '_2_9098.npz'
    LGB_LEN = 9098
elif cfg.data_source == 5:
    DATA = '_1_13475.npz'
    LGB_LEN = 13475


def read_data(dir, cls):
    y_file = None
    y = None
    if dir == 'test':
        directory = 'data/output/test/'
    elif dir == 'tr1':
        directory = 'data/output/tr1/'
    elif dir == 'tr2':
        directory = 'data/output/tr2/'
    elif dir == 'tr3':
        directory = 'data/output/tr3/'
    elif dir == 'tr4':
        directory = 'data/output/tr4/'
    else:
        return None

    if cls == 'tr':
        # X_file = dir + '_9_2162.npz'
        X_file = dir + DATA
        y_file = dir + '_y.csv'
    elif cls == 'tr_hash':
        X_file = dir + '_hash.npz'
        y_file = dir + '_y.csv'
    elif cls == 'test1':
        X_file = dir+'1'+ DATA
    elif cls == 'test1_hash':
        X_file = dir+'1_hash.npz'
    elif cls == 'test2':
        X_file = dir+'2'+ DATA
    elif cls == 'test2_hash':
        X_file = dir+'2_hash.npz'
    else:
        return None

    X = sparse.load_npz(directory + X_file)
    if y_file is not None:
        y = pd.read_csv(directory + y_file).values

    return X, y


def get_test_data(test_set):
    print('reading test data...')
    print('test set：', test_set)
    print('data type:', 'lgb_important_feature{}'.format(DATA) if cfg.feature == 1 else 'hash')
    if test_set == 1:
        test_x, _ = read_data('test', 'test1' if cfg.feature == 1 else 'test1_hash')
    elif test_set == 2:
        test_x, _ = read_data('test', 'test2' if cfg.feature == 1 else 'test2_hash')
    else:
        return None
    print('data shape:', test_x.shape)
    return test_x, test_x.shape[0]


def get_valid_data():
    print('reading valid data from fold', cfg.fold, '...')
    print('data type:', 'lgb_important_feature{}'.format(DATA) if cfg.feature == 1 else 'hash')
    if cfg.fold == 1:
        val_x, val_y = read_data('tr1', 'tr' if cfg.feature == 1 else 'tr_hash')
    elif cfg.fold == 2:
        val_x, val_y = read_data('tr2', 'tr' if cfg.feature == 1 else 'tr_hash')
    elif cfg.fold == 3:
        val_x, val_y = read_data('tr3', 'tr' if cfg.feature == 1 else 'tr_hash')
    elif cfg.fold == 4:
        val_x, val_y = read_data('tr4', 'tr' if cfg.feature == 1 else 'tr_hash')
    else:
        return None
    print('data shape:',val_x.shape)
    return val_x, val_y, val_x.shape[0]


def get_train_data(stage=0):
    print('reading train data from fold', cfg.fold, '...')
    print('data type: ', 'lgb_important_feature{}'.format(DATA) if cfg.feature == 1 else 'hash')
    if cfg.fold == 1:
        tr = ['tr2', 'tr3', 'tr4']
    elif cfg.fold == 2:
        tr = ['tr1', 'tr3', 'tr4']
    elif cfg.fold == 3:
        tr = ['tr1', 'tr2', 'tr4']
    elif cfg.fold == 4:
        tr = ['tr1', 'tr2', 'tr3']
    else:
        return None
    tr_x, tr_y = read_data(tr[stage], 'tr' if cfg.feature == 1 else 'tr_hash')
    print('data shape:', tr_x.shape)
    return tr_x, tr_y, tr_x.shape[0]










if __name__ == "__main__":
    tf.app.run()

