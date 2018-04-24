import tensorflow as tf
import numpy as np

INTEREST_UNIQUE = 637
INTEREST_EMBED = 5

KW_UNIQUE = 16656
KW_EMBED = 10

TOPIC_UNIQUE = 30001
TOPIC_EMBED = 10

APP_ID_UNIQUE = 15374
APP_ID_EMBED = 10

TRAIN = True


def main(_):
    # test_embedding()
    tf.reset_default_graph()
    with tf.get_default_graph().as_default():
        labels = 0
        if TRAIN:
            head, features = create_data_input('data/train_ad_user_all.csv', [], True, 5)
            ids, labels = tf.split(head, [2, 1], 1)
        else:
            head, features = create_data_input('data/test_ad_user_all.csv', [], False, 5)
    # just test
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        _head, label1 = sess.run([head, labels])
        print(_head)
        print(label1)
        # print(feature)

        # for i in range(1):
        #     _head, _ad_base, _user_base, _user_embedding, _feature = sess.run([head,
        #                                                                        ad_base,
        #                                                                        user_base,
        #                                                                        user_embedding,
        #                                                                        feature])
        #     print(_head)
        #     print(_ad_base)
        #     print(_user_base)
        #     print(_user_embedding)
        #     print(_feature)
        #     print()


def get_data(summary, train=True, batch_size=128):
    labels = 0
    if train:
        head, features = create_data_input('data/train_ad_user_all.csv', summary, True, batch_size)
        ids, labels = tf.split(head, [2, 1], 1)
    else:
        head, features = create_data_input('data/test_ad_user_all.csv', summary, False, batch_size)

    return features, labels


def create_data_input(files, summary, train=True, batch_size=128):
    with tf.name_scope('origin_data'):
        head, base, interest, kw, topic, appId = read_data(files, train, batch_size)
        summary.append(tf.summary.histogram('base', base))



    feature = tf.concat([base, embedded], 1)

    return head, feature


def create_embedding(inputs, words_unique, out_num, name):
    with tf.get_default_graph().as_default():
        with tf.name_scope(name + '_embedding'):
            embedding = tf.Variable(tf.random_uniform([words_unique, out_num], -1, 1), name=name)
            # embedding = tf.Variable(tf.ones([words_unique, out_num]), name=name)  测试embed
            feature_embedded = tf.nn.embedding_lookup(embedding, inputs)
            feature_embedded_sum = tf.reduce_sum(feature_embedded, -2)
            return tf.squeeze(feature_embedded_sum, -2)


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


def read_data(files, summary, train, batch_size):

    def parser(value):
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

        base = tf.stack([advertiserId, campaignId, creativeId, creativeSize, adCategoryId, productId,
                         productType, age, gender, marriageStatus, education, consumptionAbility,
                         LBS, ct, os, carrier, house])
        with tf.name_scope('input_embedding'):
            embedded = complex_data_embedding([interest, kw, topic, appId])
            summary.append(tf.summary.histogram('embedded', embedded))

        return head, base, embedded

    dataset = tf.data.TextLineDataset(files).skip(1).map(parser).batch(batch_size)
    if train:
        dataset = dataset.shuffle(100).repeat()

    data_input = dataset.make_one_shot_iterator().get_next()

    return data_input


def complex_data_embedding(data_origin):
    interest, kw, topic, appId = data_origin
    interest = convert(interest)
    kw = convert(kw)
    topic = convert(topic)
    appId = convert(appId)
    interest_embedded = create_embedding(interest, INTEREST_UNIQUE, INTEREST_EMBED, 'interest')
    kw_embedded = create_embedding(kw, KW_UNIQUE, KW_EMBED, 'kw')
    topic_embedded = create_embedding(topic, TOPIC_UNIQUE, TOPIC_EMBED, 'topic')
    appId_embedded = create_embedding(appId, APP_ID_UNIQUE, APP_ID_EMBED, 'appId')

    embedded = tf.concat([interest_embedded, kw_embedded, topic_embedded, appId_embedded], 0)

    return embedded


def convert(value):
    splits = tf.string_split(value, '-')
    dense = tf.sparse_tensor_to_dense(splits, default_value='0')
    return tf.string_to_number(dense, out_type=tf.int32)


if __name__ == "__main__":
    tf.app.run()
