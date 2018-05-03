from collections import Counter
from sklearn.metrics import roc_curve, auc

import pandas as pd
import numpy as np
import sys


def create_lookup_tables(words):
    word_counts = Counter(words)
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    vocab_to_int = {word: ii for ii, word in enumerate(sorted_vocab)}

    return vocab_to_int


def map_complex_elements(source, map_dic, fill_na='0'):
    mapped = source.copy()
    length = len(mapped)
    step = length // 10
    for i, e in enumerate(source.fillna(fill_na)):
        if e.isdigit():
            mapped[i] = str(map_dic[int(e)])
        else:
            elements = list()
            for ee in e.split('-'):
                elements.append(str(map_dic[int(ee)]))
            formed = '-'
            formed = formed.join(elements)
            mapped[i] = formed
        if i % step == 0:
            sys.stdout.write('progress: ' + str(10 * i // step) + '\r')
            sys.stdout.flush()
    sys.stdout.write('progress: 100\n')
    sys.stdout.flush()
    return mapped


def remove_rare_labels(source, remove_list, fill_na='0'):
    cleaned = source.copy()
    length = len(cleaned)
    step = length // 10
    for i, e in enumerate(source.fillna(fill_na)):
        if e.isdigit():
            if int(e) in remove_list:
                cleaned[i] = '0'
            if int(e) == 0:
                cleaned[i] = '0'
        else:
            elements = list()
            for ee in e.split('-'):
                if int(ee) not in remove_list:
                    elements.append(ee)
            if len(elements) == 0:
                cleaned[i] = '0'
            else:
                formed = '-'
                formed = formed.join(elements)
                cleaned[i] = formed
        if i % step == 0:
            sys.stdout.write('progress: ' + str(10 * i // step) + '\r')
            sys.stdout.flush()
    sys.stdout.write('progress: 100\n')
    sys.stdout.flush()
    return cleaned


def count_element(feature_box, fill_na='0'):
    collector = list()
    length = len(feature_box)
    step = length // 10
    filled_feature = feature_box.fillna(fill_na)
    for i, e in enumerate(feature_box.fillna(fill_na)):
        e = filled_feature[i]
        if e.isdigit():
            collector.append(int(e))
        else:
            for ee in e.split('-'):
                collector.append(int(ee))
        if i % step == 0:
            sys.stdout.write('progress: ' + str(10 * i // step) + '\r')
            sys.stdout.flush()
    sys.stdout.write('progress: 100\n')
    sys.stdout.flush()
    return collector


def count_baseline(counter, line):
    dic_less_than = dict()
    dic_more_than = dict()
    for e in counter.keys():
        if counter.get(e) < line:
            dic_less_than[e] = counter.get(e)
        else:
            dic_more_than[e] = counter.get(e)
    return dic_less_than, dic_more_than


def asset_clean_result(cleaned_count, boundary, less, more):
    dic_less_than_cleaned = dict()
    dic_more_than_cleaned = dict()

    for e in cleaned_count.keys():
        if cleaned_count.get(e) < boundary:
            dic_less_than_cleaned[e] = cleaned_count.get(e)
        elif cleaned_count.get(e) >= boundary:
            dic_more_than_cleaned[e] = cleaned_count.get(e)

    print(len(less))
    print(len(more))
    print(len(dic_less_than_cleaned))
    print(len(dic_more_than_cleaned))


def add_base_for_all(feature_box, base):
    added_feature = feature_box.copy()
    length = len(added_feature)
    step = length // 10
    for i, e in enumerate(feature_box.fillna('NaN')):
        if e == 'NaN':
            continue
        if e.isdigit():
            added_feature[i] = str(int(e) + base)
        else:
            elements = list()
            for ee in e.split('-'):
                elements.append(str(int(ee) + base))
            formed = '-'
            formed = formed.join(elements)
            added_feature[i] = formed
        if i % step == 0:
            sys.stdout.write('progress: ' + str(10 * i // step) + '%\r')
            sys.stdout.flush()
    sys.stdout.write('progress: 100%\n')
    sys.stdout.flush()
    return added_feature


def merge_feature(source, feature1, feature2):
    merged = source.copy()
    f1 = source[feature1].copy()
    f2 = source[feature2].fillna('NaN')
    length = len(f1)
    step = length // 10
    for i, e in enumerate(f1.fillna('NaN')):
        if e == 'NaN':
            f1[i] = f2[i]
        else:
            if f2[i] != 'NaN':
                formed = '-'
                formed = formed.join([e, f2[i]])
                f1[i] = formed
        if i % step == 0:
            sys.stdout.write('progress: ' + str(10 * i // step) + '%\r')
            sys.stdout.flush()
    sys.stdout.write('progress: 100%\n')
    sys.stdout.flush()
    merged[feature1] = f1
    return merged.drop(feature2, axis=1).rename(columns={feature1: feature1+feature2})


def cal_pos_rate(data_set):
    train_pos = data_set[data_set['label'] == 1]
    train_neg = data_set[data_set['label'] == -1]
    total_len = len(data_set)
    print('pos:', len(train_pos) / total_len)
    print('neg:', len(train_neg) / total_len)


def cal_estimated_pos_rate(data_set, threshold=0.5):
    estimated_pos = data_set[data_set['score'] >= threshold]
    pos = data_set[data_set['label'] == 1]
    total_len = len(data_set)
    return len(estimated_pos) / total_len, len(pos) / total_len


def get_pos_rate_by_aid(data_set, ads):
    pos_rates = list()
    for aid in ads:
        collect = data_set[data_set['aid'] == aid]
        pos_rates.append(len(collect[collect['label'] == 1]) / len(collect))
    return pos_rates


def cal_auc(labels, predict, pos_label=1):
    fpr, tpr, thresholds = roc_curve(labels, predict, pos_label=pos_label)
    return np.mean(fpr), np.mean(tpr), auc(fpr, tpr)


def cal_auc_by_aid_per_batch(aids, labels, pres):
    scores = list()
    aids = np.squeeze(aids)
    labels = np.squeeze(labels)
    pres = np.squeeze(pres)
    df = pd.DataFrame({'aid': aids, 'labels': labels, 'pres': pres})
    aid_types = df['aid'].drop_duplicates()
    for aid in aid_types:
        l = np.array(df[df['aid'] == aid]['labels'])
        p = np.array(df[df['aid'] == aid]['pres'])
        if 1 in l and 0 in l:
            _, _, _auc = cal_auc(l, p, 1)
            scores.append(_auc)
    if len(scores) == 0:
        return 0
    else:
        return np.mean(scores)


def cal_auc_by_aid(data_set, log=True):
    aids = data_set['aid'].drop_duplicates()
    auc_list = list()
    fpr_list = list()
    tpr_list = list()
    for aid in aids:
        group = data_set[data_set['aid'] == aid]
        labels = group['label']
        scores = group['score']
        fpr, tpr, _auc = cal_auc(labels, scores)
        if log:
            print('aid:{}, fpr:{:5.3f}, tpr:{:5.3f}, auc:{:5.3f}'
                  .format(aid, fpr, tpr, _auc))
        auc_list.append(_auc)
        fpr_list.append(fpr)
        tpr_list.append(tpr)

    print('auc of {} aids:{}'.format(len(aids), np.mean(auc_list)))
    print('fpr:{} tpr:{}'.format(np.mean(fpr_list), np.mean(tpr_list)))
