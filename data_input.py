import tensorflow as tf
import pandas as pd
import numpy as np
# 超参配置文件
from config import cfg
from scipy import sparse


LGB_LEN = 2162
HASH_LEN = 3188


def read_data(dir, cls):
    y_file = None
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
        X_file = dir + '_9_2162.npz'
        y_file = dir + '_y.csv'
    elif cls == 'tr_hash':
        X_file = dir + '_hash.npz'
        y_file = dir + '_y.csv'
    elif cls == 'test1':
        X_file = dir+'1_9_2162.npz'
    elif cls == 'test1_hash':
        X_file = dir+'1_hash.npz'
    elif cls == 'test2':
        X_file = dir+'2_9_2162.npz'
    elif cls == 'test2_hash':
        X_file = dir+'2_hash.npz'
    else:
        return None

    X = sparse.load_npz(directory + X_file)
    if y_file is not None:
        y = pd.read_csv(directory + y_file).as_matrix()

    return X, y


def get_test_data(test_set):
    print('reading test data...')
    if test_set == 1:
        test_x, _ = read_data('test', 'test1' if cfg.feature == 1 else 'test1_hash')
    elif test_set == 2:
        test_x, _ = read_data('test', 'test2' if cfg.feature == 1 else 'test2_hash')
    else:
        return None
    return test_x, test_x.shape[0]


def get_valid_data():
    print('reading valid data from fold', cfg.fold, '...')
    print('data type: ', 'lgb' if cfg.feature == 1 else 'hash')
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
    return val_x, val_y, val_x.shape[0]


def get_train_data(stage=0):
    print('reading train data from fold', cfg.fold, '...')
    print('data type: ', 'lgb' if cfg.feature == 1 else 'hash')
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
    return tr_x, tr_y, tr_x.shape[0]










if __name__ == "__main__":
    tf.app.run()

