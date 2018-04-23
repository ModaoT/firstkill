from collections import Counter

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



