from collections import Counter


def create_lookup_tables(words):
    word_counts = Counter(words)
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    vocab_to_int = {word: ii for ii, word in enumerate(sorted_vocab)}

    return vocab_to_int


def remove_rare_labels(source, remove_list, fill_na='0'):
    cleaned = source.copy()
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
    return cleaned


def count_element(feature_box, fill_na='0'):
    collector = list()
    for i, e in enumerate(feature_box.fillna(fill_na)):
        if e.isdigit():
            collector.append(int(e))
        else:
            for ee in e.split('-'):
                collector.append(int(ee))

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

