import tensorflow as tf

INTEREST1_UNIQUE = 123
INTEREST1_EMBED = 1
INTEREST2_UNIQUE = 81
INTEREST2_EMBED = 1
INTEREST3_UNIQUE = 11
INTEREST3_EMBED = 1
INTEREST4_UNIQUE = 11
INTEREST4_EMBED = 1
INTEREST5_UNIQUE = 137
INTEREST5_EMBED = 1

KW1_UNIQUE = 123
KW1_EMBED = 12
KW2_UNIQUE = 123
KW2_EMBED = 12
KW3_UNIQUE = 123
KW3_EMBED = 12

TOPIC1_UNIQUE = 123
TOPIC1_EMBED = 12
TOPIC2_UNIQUE = 123
TOPIC2_EMBED = 12
TOPIC3_UNIQUE = 123
TOPIC3_EMBED = 12

APP_ID_INSTALL_UNIQUE = 123
APP_ID_INSTALL_EMBED = 1
APP_ID_ACTION_UNIQUE = 123
APP_ID_ACTION_EMBED = 1


def main(_):
    tf.reset_default_graph()
    with tf.get_default_graph().as_default():
        interest1, interest2, interest3, interest4, interest5, \
        kw1, kw2, kw3, \
        topic1, topic2, topic3, \
        appIdInstall, appIdAction = read_data('data/user_feature_embedding.csv', 2)

        interest1_embedded = create_embedding(interest1, 124, )

    # just test
    with tf.Session() as sess:
        for i in range(1):
            # interest1, interest2, interest3, interest4, interest5, \
            #        kw1, kw2, kw3, \
            #        topic1, topic2, topic3, \
            #        appIdInstall, appIdAction = sess.run([interest1, interest2, interest3, interest4, interest5, \
            #        kw1, kw2, kw3, \
            #        topic1, topic2, topic3, \
            #        appIdInstall, appIdAction])
            res = sess.run([interest1, interest2])
            print(res)
            print()


def create_embedding(inputs, words_unique, out_num, name):
    with tf.get_default_graph().as_default():
        with tf.name_scope(name + '_embedding'):
            embedding = tf.Variable(tf.random_uniform([words_unique, out_num], -1, 1), name=name)
            feature_embedded = tf.nn.embedding_lookup(embedding, inputs)
            feature_embedded_sum = tf.reduce_sum(feature_embedded, -2)
            return feature_embedded_sum


def read_data(files, batch_size):
    COLUMN_DEFAULTS = [[''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], ['']]
    COLUMNS = ['interest1', 'interest2', 'interest3', 'interest4', 'interest5',
               'kw1', 'kw2', 'kw3',
               'topic1', 'topic2', 'topic3',
               'appIdInstall', 'appIdAction']

    def convert(value):
        splits = tf.string_split([value], '-')
        dense = tf.sparse_tensor_to_dense(splits, default_value='0')
        return tf.string_to_number(dense, out_type=tf.int32)

    def parser(value):
        interest1, interest2, interest3, interest4, interest5, \
        kw1, kw2, kw3, \
        topic1, topic2, topic3, \
        appIdInstall, appIdAction = tf.decode_csv(value, COLUMN_DEFAULTS)
        return convert(interest1), convert(interest2), convert(interest3), convert(interest4), convert(interest5), \
               convert(kw1), convert(kw2), convert(kw3), \
               convert(topic1), convert(topic2), convert(topic3), \
               convert(appIdInstall), convert(appIdAction)

    dataset = tf.data.TextLineDataset(files).skip(1).map(parser).batch(batch_size)
    text_line = dataset.make_one_shot_iterator().get_next()

    return text_line


if __name__ == "__main__":
    tf.app.run()
