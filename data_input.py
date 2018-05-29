import tensorflow as tf
import pandas as pd
import numpy as np
# 超参配置文件
from config import cfg

INTEREST_UNIQUE = 637
KW_UNIQUE = 16656
TOPIC_UNIQUE = 30001
APP_ID_UNIQUE = 15374
EMBEDDING = 500

TRAIN = True


def main(_):
    test_embedding()
    tf.reset_default_graph()
    with tf.get_default_graph().as_default():
        labels = 0
        if TRAIN:
            head, features, interest, kw, topic, appId = create_data_input('data/train_ad_user_all.csv', [], True, 2)
            ids, labels = tf.split(head, [2, 1], 1)
        else:
            head, features, interest, kw, topic, appId = create_data_input('data/test_ad_user_all.csv', [], False, 2)
    # just test
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        interest, kw, topic, appId, features = sess.run([interest, kw, topic, appId, features])
        print(interest)
        print(kw)
        print(topic)
        print(appId)
        print(features)


def get_data(train=True, valid=True):
    if train:
        if valid:
            tr_x = pd.read_csv('data/tr_tf_x.csv', iterator=True)
            tr_y = pd.read_csv('data/tr_tf_y.csv', iterator=True)
            return tr_x, tr_y
        else:
            train_x = pd.read_csv('data/train_x.csv', iterator=True)
            train_y = pd.read_csv('data/train_y.csv', iterator=True)
            return train_x, train_y
    else:
        if valid:
            val_x = pd.read_csv('data/val_tf_x.csv', iterator=True)
            val_y = pd.read_csv('data/val_tf_y.csv', iterator=True)
            return val_x, val_y
        else:
            test_x = pd.read_csv('data/test_x.csv', iterator=True)
            return test_x


ID_COLUMNS = ['aid', 'uid']
BASE_COLUMNS = ['advertiserId', 'campaignId', 'creativeId',
                'creativeSize', 'adCategoryId', 'productId', 'productType',
                'age', 'carrier', 'consumptionAbility', 'ct', 'education', 'gender', 'marriageStatus', 'os',
                'aid_', 'LBS', 'house']
LIST_COLUMNS = ['interest', 'kw', 'topic', 'app']
ADVERTISERID_LEN = 79
CAMPAIGNID_LEN = 138
CREATIVEID_LEN = 173
CREATIVESIZE_LEN = 15
ADCATEGORYID_LEN = 40
PRODUCTID_LEN = 33
PRODUCTTYPE_LEN = 4
LBS_LEN = 856
AGE_LEN = 6
CARRIER_LEN = 4
CONSUMPTIONABILITY_LEN = 3
CT_LEN = 16
EDUCATION_LEN = 8
GENDER_LEN = 3
HOUSE_LEN = 2
MARRIAGESTATUS_LEN = 26
OS_LEN = 4
AID__LEN = 173
EMBED_LEN = 36479


def parse_normal_feature(data, one_hot_length):
    one_hot = np.zeros([cfg.batch, one_hot_length])
    for i in range(cfg.batch):
        one_hot[i, data[i]] = 1
    return data, one_hot


def parse_list_feature(data, one_hot_length):
    one_hot = np.zeros([cfg.batch, one_hot_length])
    for i in range(cfg.batch):
        e = data[i].strip().split()
        for ee in e:
            one_hot[i, ee] = 1
    return data, one_hot


def parse_feature(data):
    baseline = ADVERTISERID_LEN
    data.campaignId += baseline

    baseline += CAMPAIGNID_LEN
    data.creativeId += baseline

    baseline += CREATIVEID_LEN
    data.creativeSize += baseline

    baseline += CREATIVESIZE_LEN
    data.adCategoryId += baseline

    baseline += ADCATEGORYID_LEN
    data.productId += baseline

    baseline += PRODUCTID_LEN
    data.productType += baseline

    baseline += PRODUCTTYPE_LEN
    data.age += baseline

    baseline += AGE_LEN
    data.carrier += baseline

    baseline += CARRIER_LEN
    data.consumptionAbility += baseline

    baseline += CONSUMPTIONABILITY_LEN
    data.ct += baseline

    baseline += CT_LEN
    data.education += baseline

    baseline += EDUCATION_LEN
    data.gender += baseline

    baseline += GENDER_LEN
    data.marriageStatus += baseline

    baseline += MARRIAGESTATUS_LEN
    data.os += baseline

    baseline += OS_LEN
    data.aid_ += baseline

    baseline += AID__LEN
    data.LBS += baseline

    baseline += LBS_LEN
    data.house += baseline

    base = data[BASE_COLUMNS]
    embed = data['embed']

    return base.as_matrix(), embed.as_matrix()


def get_data_(summary, handle, train=True, valid=True, batch_size=128):
    with tf.name_scope('input'):
        labels = 0
        training_iter, validation_iter = None, None
        if train:
            head, base, interest, kw, topic, appId, training_iter, validation_iter = \
                create_train_data_input(summary, handle, batch_size=batch_size, repeat=True, valid=valid)
            aid, uid, labels = tf.split(head, [1, 1, 1], 1)
        else:
            if valid:
                head, base, interest, kw, topic, appId = create_test_data_input('data/train_valid.csv',
                                                                                True, batch_size)
                aid, uid, labels = tf.split(head, [1, 1, 1], 1)
            else:
                head, base, interest, kw, topic, appId = create_test_data_input('data/test_ad_user_all.csv',
                                                                                False, batch_size)
                aid, uid = tf.split(head, [1, 1], 1)

        with tf.name_scope('input_embedding'):
            interest_embedded = create_embedding(interest, INTEREST_UNIQUE, INTEREST_EMBED, 'interest', summary,
                                                 'first')
            kw_embedded = create_embedding(kw, KW_UNIQUE, KW_EMBED, 'kw', summary, 'first')
            topic_embedded = create_embedding(topic, TOPIC_UNIQUE, TOPIC_EMBED, 'topic', summary, 'last')
            appId_embedded = create_embedding(appId, APP_ID_UNIQUE, APP_ID_EMBED, 'appId', summary, 'first')

            embedded = tf.concat([interest_embedded, kw_embedded, topic_embedded, appId_embedded], -1)

            summary.append(tf.summary.histogram('embedded', embedded))
        features = tf.concat([base, embedded], 1, name='features')

    return aid, uid, features, labels, training_iter, validation_iter


def create_train_data_input(summary, handle, batch_size=128, repeat=True, valid=True):
    with tf.name_scope('origin_data'):
        training_iter, validation_iter = None, None
        if valid:
            tr_dataset = read_data('data/train_train.csv', True, batch_size, repeat=repeat)
            val_dataset = read_data('data/train_valid.csv', True, batch_size, repeat=repeat)
            iterator = tf.data.Iterator.from_string_handle(handle, tr_dataset.output_types, tr_dataset.output_shapes)
            head, base, interest, kw, topic, appId = iterator.get_next()

            training_iter = tr_dataset.make_one_shot_iterator()
            validation_iter = val_dataset.make_one_shot_iterator()
        else:
            tr_dataset = read_data('data/train_ad_user_all_shuffle.csv', True, batch_size, repeat=repeat)
            head, base, interest, kw, topic, appId = tr_dataset.make_one_shot_iterator().get_next()

        interest = convert(interest)
        kw = convert(kw)
        topic = convert(topic)
        appId = convert(appId)
        summary.append(tf.summary.histogram('interest_input', interest))
        summary.append(tf.summary.histogram('kw_input', kw))
        summary.append(tf.summary.histogram('topic_input', topic))
        summary.append(tf.summary.histogram('appId_input', appId))
        summary.append(tf.summary.histogram('base', base))

    return head, base, interest, kw, topic, appId, training_iter, validation_iter


def create_test_data_input(files, valid, batch_size=128):
    with tf.name_scope('origin_data'):
        dataset = read_data(files, valid, batch_size, repeat=False)

        head, base, interest, kw, topic, appId = dataset.make_one_shot_iterator().get_next()
        interest = convert(interest)
        kw = convert(kw)
        topic = convert(topic)
        appId = convert(appId)

    return head, base, interest, kw, topic, appId


def create_embedding(inputs, words_unique, out_num, name, summary, remove='first'):
    with tf.get_default_graph().as_default():
        with tf.name_scope(name + '_embedding'):
            embedding = tf.Variable(tf.random_uniform([words_unique, out_num], -1, 1), name=name)
            # embedding = tf.Variable(tf.ones([words_unique, out_num]), name=name)  #测试embed
            remove_mask = tf.zeros([1, out_num])
            if remove == 'first':
                masked_embedding = tf.concat([remove_mask, embedding], 0)
            else:
                masked_embedding = tf.concat([remove_mask, embedding, remove_mask], 0)
            summary.append(tf.summary.histogram(name + '_embedding_var', masked_embedding))
            feature_embedded = tf.nn.embedding_lookup(masked_embedding, inputs)
            feature_embedded_sum = tf.reduce_sum(feature_embedded, -2)
            return feature_embedded_sum


def test_embedding():
    inputs = tf.constant([[1, 2, 3], [4, 5, 6]])
    with tf.get_default_graph().as_default():
        with tf.name_scope('embedding'):
            embedding = tf.Variable(np.identity(10))
            feature_embedded = tf.nn.embedding_lookup(embedding, inputs)
            feature_embedded_sum = tf.reduce_sum(feature_embedded, -2)
        # just test
        init_op = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init_op)
            for i in range(1):
                feature_embedded, feature_embedded_sum = sess.run([feature_embedded, feature_embedded_sum])
                print(feature_embedded)
                print(feature_embedded_sum)


def read_data(files, train, batch_size, repeat):

    def parser_v1(value):
        if train:
            COLUMN_DEFAULTS = [[0], [0], [0],
                               [0.], [0.], [0.], [0.], [0.], [0.], [0.],
                               [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.],
                               [''], [''], [''], ['']]
            aid, uid, label, \
            advertiserId, campaignId, creativeId, creativeSize, adCategoryId, productId, productType, \
            age, gender, marriageStatus, education, consumptionAbility, LBS, ct, os, carrier, house, \
            interest, kw, topic, appId = tf.decode_csv(value, COLUMN_DEFAULTS)

            label = tf.div(tf.add(label, 1), 2)
            head = tf.stack([aid, uid, label])
        else:
            COLUMN_DEFAULTS = [[0], [0],
                               [0.], [0.], [0.], [0.], [0.], [0.], [0.],
                               [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.],
                               [''], [''], [''], ['']]
            aid, uid, \
            advertiserId, campaignId, creativeId, creativeSize, adCategoryId, productId, productType, \
            age, gender, marriageStatus, education, consumptionAbility, LBS, ct, os, carrier, house, \
            interest, kw, topic, appId = tf.decode_csv(value, COLUMN_DEFAULTS)
            head = tf.stack([aid, uid])

        base = tf.stack([tf.log(tf.to_float(aid)), advertiserId, campaignId, creativeId, creativeSize, adCategoryId, productId,
                         productType, age, gender, marriageStatus, education, consumptionAbility,
                         LBS, ct, os, carrier, house])

        return head, base, interest, kw, topic, appId

    dataset = tf.data.TextLineDataset(files).skip(1).map(parser_v1).batch(batch_size).repeat(
        count=-1 if repeat else 1)

    return dataset


def convert(value):
    splits = tf.string_split(value)
    dense = tf.sparse_tensor_to_dense(splits, default_value='0')
    return tf.string_to_number(dense, out_type=tf.int32)


if __name__ == "__main__":
    tf.app.run()

