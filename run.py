import os

import numpy as np
import pandas as pd
import tensorflow as tf
# 进度条工具
from tqdm import tqdm

import data_input
import utils
# 超参配置文件
from config import cfg
from data_input import ADVERTISERID_LEN, CAMPAIGNID_LEN, CREATIVEID_LEN, CREATIVESIZE_LEN, ADCATEGORYID_LEN, PRODUCTID_LEN, PRODUCTTYPE_LEN, LBS_LEN, AGE_LEN, CARRIER_LEN, CONSUMPTIONABILITY_LEN, CT_LEN, EDUCATION_LEN, GENDER_LEN, HOUSE_LEN, MARRIAGESTATUS_LEN, OS_LEN, AID__LEN, EMBED_LEN

BASE_LEN = ADVERTISERID_LEN+CAMPAIGNID_LEN+CREATIVEID_LEN+CREATIVESIZE_LEN+ADCATEGORYID_LEN+PRODUCTID_LEN+PRODUCTTYPE_LEN+AGE_LEN+CARRIER_LEN+CONSUMPTIONABILITY_LEN+CT_LEN+EDUCATION_LEN+GENDER_LEN+MARRIAGESTATUS_LEN+OS_LEN+AID__LEN
HAS_NA = LBS_LEN+HOUSE_LEN
INPUT_LEN = BASE_LEN+HAS_NA+EMBED_LEN

TRAIN_NUM = 8798814
TRAIN_TRAIN_NUM = 7039051
TRAIN_VALID_NUM = 1759763
TEST_NUM = 2265989


def main(_):
    tf.reset_default_graph()
    graph = tf.Graph()

    if cfg.train:
        train(graph)
    else:
        evaluate(graph)


def train(graph):
    if cfg.valid:  # 使用训练集+验证集
        # 获取上一次保存的状态
        train_batch = TRAIN_TRAIN_NUM // cfg.batch
        train_valid_batch = TRAIN_VALID_NUM // cfg.batch
        ckpt, global_step, last_epoch, last_step = get_last_state(cfg.logdir, train_batch)
    else:  # 使用用全部训练集
        # 获取上一次保存的状态
        train_batch = TRAIN_NUM // cfg.batch  # 训练集大小//batch大小
        ckpt, global_step, last_epoch, last_step = get_last_state(cfg.logdir, train_batch)

    summary = []
    with graph.as_default():
        with tf.name_scope('Input'):
            feature_base = tf.placeholder(dtype=tf.int32, shape=[None, 18], name='base')
            feature_embed = tf.placeholder(dtype=tf.string, shape=[None], name='embed')
            labels = tf.placeholder(dtype=tf.int32, shape=[None, 1], name='labels')
            is_train = tf.placeholder(dtype=tf.bool, shape=[])

        # 构造网络结构
        logits, outputs = build_arch(feature_base, feature_embed, cfg.hidden, summary, is_train)

        if cfg.train:
            # 构造损失函数
            loss = build_loss(labels, logits, summary)

            merged_summary = tf.summary.merge(summary)
            # 构造学习器
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):  # 为了batch normalization能正常运行
                opt = tf.train.AdamOptimizer(cfg.lr).minimize(loss)

            init_op = tf.group([tf.global_variables_initializer(), tf.local_variables_initializer()])
            saver = tf.train.Saver(max_to_keep=10)
            auc_saver = get_auc_saver(cfg.auc_path)
            with tf.Session() as sess:
                sess.run(init_op)

                train_writer = tf.summary.FileWriter(cfg.logdir + '/train', sess.graph)
                valid_writer = tf.summary.FileWriter(cfg.logdir + '/valid')

                if ckpt and ckpt.model_checkpoint_path:
                    # 加载上次保存的模型
                    print('load model: ', ckpt.model_checkpoint_path)
                    saver.restore(sess, ckpt.model_checkpoint_path)
                # 计算图结构分析
                param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(
                    tf.get_default_graph(),
                    tfprof_options=tf.contrib.tfprof.model_analyzer.TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
                print('total_params: %d\n' % param_stats.total_parameters)

                for e in range(last_epoch, cfg.epoch):
                    print('Training for epoch ' + str(e + 1) + '/' + str(cfg.epoch) + ':')

                    tr_x, tr_y = data_input.get_data(True, cfg.valid)
                    if cfg.valid:
                        val_x, val_y = data_input.get_data(False, True)

                    bar = tqdm(range(0, train_batch+1), initial=0, total=train_batch, ncols=150, leave=False,
                               unit='b')

                    for _ in bar:
                        tr_x_batch = tr_x.get_chunk(cfg.batch)
                        tr_y_batch = tr_y.get_chunk(cfg.batch)
                        tr_y_batch = tr_y_batch.as_matrix()

                        tr_base, tr_embed = data_input.parse_feature(tr_x_batch)

                        if global_step % cfg.summary == 0:
                            tr_labels, tr_loss, tr_pre, summary_str = sess.run(
                                [labels, loss, outputs, merged_summary],
                                feed_dict={feature_base: tr_base,
                                           feature_embed: tr_embed,
                                           labels: tr_y_batch,
                                           is_train: False})
                            train_writer.add_summary(summary_str, global_step)
                            _, _, tr_auc = utils.cal_auc(tr_labels, tr_pre)

                            if cfg.valid:
                                try:
                                    val_x_batch = val_x.get_chunk(cfg.batch)
                                    val_y_batch = val_y.get_chunk(cfg.batch)
                                except StopIteration:
                                    val_x, val_y = data_input.get_data(False, True)
                                    val_x_batch = val_x.get_chunk(cfg.batch)
                                    val_y_batch = val_y.get_chunk(cfg.batch)

                                val_base, val_embed = data_input.parse_feature(val_x_batch)

                                val_labels, val_loss, val_pre, summary_str = sess.run(
                                    [labels, loss, outputs, merged_summary],
                                    feed_dict={feature_base: val_base,
                                               feature_embed: val_embed,
                                               labels: val_y_batch,
                                               is_train: False})
                                valid_writer.add_summary(summary_str, global_step)
                                _, _, val_auc = utils.cal_auc(val_labels, val_pre)
                                bar.set_description('t_l:{:5.3f},v_l:{:5.3f},t_a:{:5.3f},v_a:{:5.3f}'
                                                    .format(tr_loss, val_loss, tr_auc, val_auc))
                            else:
                                val_auc = 0
                                val_loss = 0
                                bar.set_description('tr_loss:{:5.3f},tr_auc:{:5.3f}'
                                                    .format(tr_loss, tr_auc))

                            auc_saver.write(str(global_step) + ','
                                            + str(tr_loss) + ','
                                            + str(val_loss) + ','
                                            + str(tr_auc) + ','
                                            + str(val_auc) + "\n")
                            auc_saver.flush()
                        else:
                            sess.run(opt, feed_dict={feature_base: tr_base,
                                                     feature_embed: tr_embed,
                                                     labels: tr_y_batch,
                                                     is_train: True})

                        global_step += 1
                        if global_step % cfg.checkpoint == 0:
                            saver.save(sess, cfg.logdir + '/model.ckpt', global_step=global_step)

                    saver.save(sess, cfg.logdir + '/model.ckpt', global_step=global_step)
                    bar.close()

                train_writer.close()
                valid_writer.close()
                auc_saver.close()


def evaluate(graph):
    with graph.as_default():
        if cfg.valid:
            # 获取上一次保存的状态
            batch_num = TRAIN_VALID_NUM // cfg.batch
            ckpt, _, _, _ = get_last_state(cfg.logdir, batch_num)
            # 读取数据
            X, y = data_input.get_data(False, True)

        else:  # 使用测试集生成submission
            # 获取上一次保存的状态
            batch_num = TEST_NUM // cfg.batch  # 测试集大小//batch大小
            ckpt, _, _, _ = get_last_state(cfg.logdir, batch_num)
            # 读取数据
            X = data_input.get_data(False, False)

        if ckpt is None or batch_num == 0:
            print('No ckpt found!')
            return

        result = np.array([], dtype=np.float32)

        # 构造网络结构
        feature_base = tf.placeholder(dtype=tf.int32, shape=[None, 18], name='base')
        feature_embed = tf.placeholder(dtype=tf.string, shape=[None], name='embed')

        # 构造网络结构
        logits, outputs = build_arch(feature_base, feature_embed, cfg.hidden, [], False)

        init_op = tf.group([tf.global_variables_initializer(), tf.local_variables_initializer()])
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(init_op)

            print('load model: ', ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)

            print('computing result ...')
            bar = tqdm(range(0, batch_num + 1), total=batch_num, ncols=100, leave=False,
                       unit='b')
            for _ in bar:
                x_batch = X.get_chunk(cfg.batch)

                base, embed = data_input.parse_feature(x_batch)

                _outputs = sess.run(outputs, feed_dict={feature_base: base,
                                                        feature_embed: embed})

                result = np.append(result, _outputs)

            bar.close()
            print('scores length: ', len(result))  # 2265989
            if cfg.valid:
                print('reading train_valid.csv ...')
                valid_data = pd.read_csv('data/val_x.csv')
                valid_label = pd.read_csv('data/val_y.csv')
                valid_data['score'] = np.array(result)
                valid_data['label'] = valid_label['label']
                print('computing auc ...')
                utils.cal_auc_by_aid(valid_data[['aid', 'uid', 'label', 'score']])
            else:
                print('reading res.csv ...')
                test_data = pd.read_csv('data/res.csv')
                test_data['score'] = np.array(result)
                print('writing results into submission.csv ...')
                test_data[['aid', 'uid', 'score']].to_csv('results/submission.csv', columns=['aid', 'uid', 'score'],
                                                          index=False)

        print('finish')


def get_auc_saver(path):
    if not os.path.exists('results'):
        os.mkdir('results')

    train_auc = path
    if os.path.exists(train_auc):
        os.remove(train_auc)

    fd_train_auc = open(train_auc, 'a')
    if not os.path.exists(train_auc):
        fd_train_auc.write('step,train_loss,valid_loss,train_auc,valid_auc\n')

    return fd_train_auc


def build_arch(base, embed, hide_layer, summary, is_training):
    with tf.name_scope('arch'):
        v_base = tf.Variable(tf.truncated_normal(shape=[BASE_LEN, cfg.embed], stddev=0.01), dtype=tf.float32)

        v_lbs = tf.Variable(tf.truncated_normal(shape=[LBS_LEN-1, cfg.embed], stddev=0.01), dtype=tf.float32)
        v_lbs = tf.concat([tf.zeros([1, cfg.embed]), v_lbs], 0)
        v_house = tf.Variable(tf.truncated_normal(shape=[HOUSE_LEN-1, cfg.embed], stddev=0.01), dtype=tf.float32)
        v_house = tf.concat([tf.zeros([1, cfg.embed]), v_house], 0)

        v_base = tf.concat([v_base, v_lbs, v_house], 0)

        v_embed = tf.Variable(tf.truncated_normal(shape=[EMBED_LEN-1, cfg.embed], stddev=0.01), dtype=tf.float32)
        v_embed = tf.concat([tf.zeros([1, cfg.embed]), v_embed], 0)

        with tf.variable_scope('FM'):
            b = tf.get_variable('bias', shape=[1], initializer=tf.zeros_initializer())
            w_base = tf.get_variable('w_base', shape=[BASE_LEN, 1],
                                     initializer=tf.truncated_normal_initializer(stddev=0.01))

            w_lbs = tf.Variable(tf.truncated_normal(shape=[LBS_LEN - 1, 1], stddev=0.01), dtype=tf.float32)
            w_lbs = tf.concat([tf.zeros([1, 1]), w_lbs], 0)
            w_house = tf.Variable(tf.truncated_normal(shape=[HOUSE_LEN - 1, 1], stddev=0.01), dtype=tf.float32)
            w_house = tf.concat([tf.zeros([1, 1]), w_house], 0)

            w_base = tf.concat([w_base, w_lbs, w_house], 0)

            w_embed = tf.get_variable('w_embed', shape=[EMBED_LEN-1, 1],
                                      initializer=tf.truncated_normal_initializer(stddev=0.01))
            w_embed = tf.concat([tf.zeros([1, 1]), w_embed], 0)

            converted_embed = data_input.convert(embed)

            # 等效于tf.matmul(X, w)
            w_lookup_base = tf.reduce_sum(tf.nn.embedding_lookup(w_base, base), -2)

            w_lookup_embed = tf.reduce_sum(tf.nn.embedding_lookup(w_embed, converted_embed), -2)

            linear_terms = tf.add(w_lookup_base, w_lookup_embed)
            linear_terms = tf.add(linear_terms, b, name='linear')
            summary.append(tf.summary.histogram('linear_terms', linear_terms))

            # 等效于tf.matmul(X, v)
            v_lookup_base1 = tf.reduce_sum(tf.nn.embedding_lookup(v_base, base), -2)
            v_lookup_embed1 = tf.reduce_sum(tf.nn.embedding_lookup(v_embed, converted_embed), -2)

            part1 = tf.add(v_lookup_base1, v_lookup_embed1)

            # 等效于tf.matmul(X^2, v^2)
            v_lookup_base2 = tf.reduce_sum(tf.pow(tf.nn.embedding_lookup(v_base, base), 2), -2)
            v_lookup_embed2 = tf.reduce_sum(tf.pow(tf.nn.embedding_lookup(v_embed, converted_embed), 2), -2)

            part2 = tf.add(v_lookup_base2, v_lookup_embed2)

            # interaction_terms = tf.multiply(0.5,
            #                                 tf.reduce_mean(tf.subtract(tf.pow(part1, 2), part2), 1, keepdims=True))
            interaction_terms = tf.multiply(0.5, tf.subtract(tf.pow(part1, 2), part2), name='interaction')
            summary.append(tf.summary.histogram('interaction_terms', interaction_terms))

            # y_fm = tf.add(linear_terms, interaction_terms)
            y_fm = tf.concat([linear_terms, interaction_terms], -1, name='fm_out')
            summary.append(tf.summary.histogram('fm_outputs', y_fm))

        with tf.variable_scope('DNN', reuse=False):
            # embedding layer
            embedding_input_base = tf.reshape(tf.gather(v_base, base), [-1, 18 * cfg.embed])  # 18为base特征数量
            embedding_input_embed = tf.reduce_mean(tf.nn.embedding_lookup(v_embed, converted_embed), -2)

            embedding_input = tf.concat([embedding_input_base, embedding_input_embed], -1)

            layer = embedding_input
            for i, num in enumerate(hide_layer):
                layer = tf.layers.dense(layer, num, activation=None, use_bias=True, name='layer' + str(i+1))
                layer = tf.layers.batch_normalization(layer, training=is_training, name='bn_layer' + str(i+1))
                layer = tf.nn.relu(layer, name='act_layer' + str(i+1))
                layer = tf.layers.dropout(layer, cfg.drop_out, is_training)
                # summary.append(tf.summary.histogram('layer' + str(i+1), layer))

            y_dnn = tf.layers.dense(layer, 1, activation=tf.nn.relu, use_bias=True, name='y_dnn')
            summary.append(tf.summary.histogram('dnn_outputs', y_dnn))

        # add FM output and DNN output
        # logits = tf.add(cfg.fm * y_fm, cfg.dnn * y_dnn, name='logits')
        combine = tf.layers.batch_normalization(tf.concat([y_fm, y_dnn], axis=1), training=is_training, name='combine')

        layer = combine
        for i, num in enumerate([128, 64, 32]):
            layer = tf.layers.dense(layer, num, activation=None, use_bias=True, name='layer' + str(i + 1))
            layer = tf.layers.batch_normalization(layer, training=is_training, name='bn_layer' + str(i + 1))
            layer = tf.nn.relu(layer, name='act_layer' + str(i + 1))
            layer = tf.layers.dropout(layer, cfg.drop_out, is_training)

        logits = tf.layers.dense(layer, 1, activation=None, use_bias=True, name='logits')
        summary.append(tf.summary.histogram('logits', logits))
        outputs = tf.sigmoid(logits, name='outputs')
        summary.append(tf.summary.histogram('outputs', outputs))

    return logits, outputs


def build_loss(labels, logits, summary):
    loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(
        tf.cast(labels, tf.float32), logits, pos_weight=cfg.pos_weight), name='loss')
    summary.append(tf.summary.scalar('loss', loss))
    return loss


def cal_auc(labels, outputs):
    # 获取正负样本的索引位置
    with tf.name_scope('auc'):
        labels_temp = labels[:, 0]
        pos_idx = tf.where(tf.equal(labels_temp, 1))[:, 0]
        neg_idx = tf.where(tf.equal(labels_temp, 0))[:, 0]

        # 获取正负样本位置对应的预测值
        pos_predict = tf.gather(outputs, pos_idx)
        pos_size = tf.shape(pos_predict)[0]
        neg_predict = tf.gather(outputs, neg_idx)
        neg_size = tf.shape(neg_predict)[0]
        # 按照论文'Optimizing Classifier Performance
        # via an Approximation to the Wilcoxon-Mann-Whitney Statistic'中的公式(7)计算loss_function
        pos_neg_diff = tf.reshape(
            -(tf.matmul(pos_predict, tf.ones([1, neg_size])) -
              tf.matmul(tf.ones([pos_size, 1]), tf.reshape(neg_predict, [1, neg_size])) - cfg.gammar),
            [-1, 1]
        )
        pos_neg_diff = tf.where(tf.greater(pos_neg_diff, 0), pos_neg_diff, tf.zeros([pos_size * neg_size, 1]))
        pos_neg_diff = tf.pow(pos_neg_diff, cfg.power_p)

        loss_approx_auc = tf.reduce_mean(pos_neg_diff)
    return loss_approx_auc


# 获取上一次保存的模型
def get_last_state(logdir, num_batch):
    ckpt = tf.train.get_checkpoint_state(logdir)
    if ckpt and ckpt.model_checkpoint_path:
        # Restores from checkpoint
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        global_step = int(ckpt_name.split('-')[-1])
        last_epoch = global_step // num_batch
        last_step = global_step % num_batch
    else:
        global_step = 0
        last_epoch = 0
        last_step = 0
    return ckpt, global_step, last_epoch, last_step


if __name__ == "__main__":
    tf.app.run()
