import tensorflow as tf
import pandas as pd
import numpy as np
import os
import utils
# 进度条工具
from tqdm import tqdm
# 超参配置文件
from config import cfg

import data_input


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
        # 从文件流读取数据
        handle = tf.placeholder(tf.string, shape=[])
        aid, uid, features, labels, training_iter, validation_iter = data_input.get_data(summary, handle, True, cfg.batch)
        # 构造网络结构
        is_train = tf.placeholder(dtype=tf.bool)
        logits, outputs = build_arch(features, cfg.hidden, summary, is_train)

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
                if cfg.valid:
                    training_handle = sess.run(training_iter.string_handle())
                    validation_handle = sess.run(validation_iter.string_handle())

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

                tr_feed = {is_train: True}
                tr_sum_feed = {is_train: False}
                if cfg.valid:
                    tr_feed[handle] = training_handle
                    tr_sum_feed[handle] = training_handle
                    val_sum_feed = {is_train: False, handle: validation_handle}

                for e in range(last_epoch, cfg.epoch):
                    print('Training for epoch ' + str(e + 1) + '/' + str(cfg.epoch) + ':')

                    bar = tqdm(range(last_step, train_batch), initial=last_step, total=train_batch, ncols=150, leave=False,
                               unit='b')

                    for _ in bar:
                        if global_step % cfg.summary == 0:
                            tr_aid, tr_uid, tr_labels, tr_loss, tr_pre, summary_str = sess.run(
                                [aid, uid, labels, loss, outputs, merged_summary],
                                feed_dict=tr_sum_feed)
                            train_writer.add_summary(summary_str, global_step)
                            tr_auc = utils.cal_auc_by_aid_per_batch(tr_aid, tr_labels, tr_pre)

                            if cfg.valid:
                                # sess.run(validation_iter.initializer)
                                val_aid, val_uid, val_labels, val_loss, val_pre, summary_str = sess.run(
                                    [aid, uid, labels, loss, outputs, merged_summary],
                                    feed_dict=val_sum_feed)
                                valid_writer.add_summary(summary_str, global_step)
                                val_auc = utils.cal_auc_by_aid_per_batch(val_aid, val_labels, val_pre)
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
                            sess.run(opt, feed_dict=tr_feed)

                        global_step += 1
                        if global_step % cfg.checkpoint == 0:
                            saver.save(sess, cfg.logdir + '/model.ckpt', global_step=global_step)

                    saver.save(sess, cfg.logdir + '/model.ckpt', global_step=global_step)

                    last_step = 0
                    bar.close()

                    # 在验证集上计算loss和auc
                    # print('do valid ...')
                    # valid_bar = tqdm(range(train_valid_batch), total=train_valid_batch, ncols=100,
                    #                  leave=False,
                    #                  unit='b')
                    # result = list()
                    # for _ in valid_bar:
                    #     valid_labels, valid_pre, valid_loss = sess.run([labels, outputs, loss],
                    #                                                    feed_dict={state: 'valid', is_train: False})
                    #     result = np.append(result, valid_pre)
                    #
                    # valid_bar.close()
                    #
                    # print('reading train_valid.csv ...')
                    # valid_data = pd.read_csv('data/train_valid.csv')
                    # valid_data['score'] = np.array(result)
                    # print('computing auc ...')
                    # utils.cal_auc_by_aid(valid_data[['aid', 'uid', 'score']], False)

                train_writer.close()
                valid_writer.close()
                auc_saver.close()


def evaluate(graph):
    with graph.as_default():
        if cfg.valid:
            # 获取上一次保存的状态
            batch_num = TRAIN_VALID_NUM // cfg.batch
            ckpt, _, _, _ = get_last_state(cfg.logdir, batch_num)
            # 从文件流读取数据
            aid, uid, features, labels, _, _ = data_input.get_data([], None, False, cfg.batch, submission=False)

        else:  # 使用测试集生成submission
            # 获取上一次保存的状态
            batch_num = TEST_NUM // cfg.batch  # 测试集大小//batch大小
            ckpt, _, _, _ = get_last_state(cfg.logdir, batch_num)
            # 从文件流读取数据
            aid, uid, features, labels, _, _ = data_input.get_data([], None, False, cfg.batch, submission=True)

        if ckpt is None or batch_num == 0:
            print('No ckpt found!')
            return

        result = np.array([], dtype=np.float32)

        # 构造网络结构
        logits, outputs = build_arch(features, cfg.hidden, [], False)

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
                _outputs = sess.run(outputs)
                result = np.append(result, _outputs)

            bar.close()
            print('scores length: ', len(result))  # 2265989
            if cfg.valid:
                print('reading train_valid.csv ...')
                valid_data = pd.read_csv('data/train_valid.csv')
                valid_data['score'] = np.array(result)
                print('computing auc ...')
                utils.cal_auc_by_aid(valid_data[['aid', 'uid', 'label', 'score']])
            else:
                print('reading test_ad_user_all.csv ...')
                test_data = pd.read_csv('data/test_ad_user_all.csv')
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


def build_arch(inputs, hide_layer, summary, is_training):
    with tf.name_scope('arch'):
        layer = inputs
        for i, num in enumerate(hide_layer):
            layer = tf.layers.dense(layer, num, activation=None, use_bias=True, name='layer' + str(i+1))
            layer = tf.layers.batch_normalization(layer, training=is_training, name='bn_layer' + str(i+1))
            layer = tf.nn.relu(layer, name='act_layer' + str(i+1))
            layer = tf.layers.dropout(layer, cfg.drop_out, is_training)
            summary.append(tf.summary.histogram('layer' + str(i+1), layer))

        logits = tf.layers.dense(layer, 1, activation=None, use_bias=True, name='logits')  # logits是用于损失函数的输入，无需activation
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
