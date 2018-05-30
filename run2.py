import os

import numpy as np
import pandas as pd
import tensorflow as tf
# 进度条工具
from tqdm import tqdm

import data_input2
import utils
# 超参配置文件
from config import cfg


def main(_):
    tf.reset_default_graph()
    graph = tf.Graph()

    if cfg.train:
        train(graph)
    else:
        evaluate(graph)


def train(graph):
    # 获取上一次保存的状态
    ckpt, last_epoch, last_stage, local_step, global_step = get_last_state(cfg.logdir)

    summary = []
    with graph.as_default():
        with tf.name_scope('Input'):
            indices_i = tf.placeholder(dtype=tf.int64, shape=[None, 2])
            values_i = tf.placeholder(dtype=tf.float32, shape=[None])
            dense_shape_i = tf.placeholder(dtype=tf.int64, shape=[2])
            feature = tf.SparseTensor(indices=indices_i, values=values_i, dense_shape=dense_shape_i)

            labels = tf.placeholder(dtype=tf.int32, shape=[None, 1], name='labels')
            is_train = tf.placeholder(dtype=tf.bool, shape=[])

        # 构造网络结构
        logits, outputs = build_arch(feature, cfg.hidden, summary, is_train)

        # 构造损失函数
        loss = build_loss(labels, logits, summary)
        merged_summary = tf.summary.merge(summary)

        # 构造学习器
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):  # 为了batch normalization能正常运行
            opt = tf.train.AdamOptimizer(cfg.lr).minimize(loss)

        init_op = tf.group([tf.global_variables_initializer(), tf.local_variables_initializer()])
        saver = tf.train.Saver(max_to_keep=10)
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
                val_x, val_y, val_size = data_input2.get_valid_data()
                for stage in range(last_stage, 4):
                    print('stage', stage, ':')
                    tr_x, tr_y, tr_size = data_input2.get_train_data(stage)
                    train_batch = tr_size // cfg.batch
                    bar = tqdm(range(local_step, train_batch+1), initial=last_epoch, total=train_batch, ncols=150, leave=False,
                               unit='b')
                    tr_idx = 0
                    val_idx = 0
                    for i in bar:
                        if i == train_batch:
                            tr_x_batch = tr_x[tr_idx:].tocoo()
                            tr_y_batch = tr_y[tr_idx:]
                            tr_idx = tr_size-1
                        else:
                            tr_x_batch = tr_x[tr_idx: tr_idx+cfg.batch].tocoo()
                            tr_y_batch = tr_y[tr_idx: tr_idx+cfg.batch]
                            tr_idx += cfg.batch
                        tr_indics = np.mat([tr_x_batch.row, tr_x_batch.col]).transpose()
                        tr_values = tr_x_batch.data
                        tr_dense_shapes = tr_x_batch.shape

                        if i % cfg.summary == 0:
                            tr_loss, tr_pre, summary_str = sess.run(
                                [loss, outputs, merged_summary],
                                feed_dict={indices_i: tr_indics,
                                           values_i: tr_values,
                                           dense_shape_i: tr_dense_shapes,
                                           labels: tr_y_batch,
                                           is_train: False})
                            train_writer.add_summary(summary_str, global_step)
                            if val_idx + cfg.batch >= val_size:
                                val_idx = 0
                            val_x_batch = val_x[val_idx: val_idx + cfg.batch].tocoo()
                            val_y_batch = val_y[val_idx: val_idx + cfg.batch]
                            val_idx += cfg.batch
                            val_indics = np.mat([val_x_batch.row, val_x_batch.col]).transpose()
                            val_values = val_x_batch.data
                            val_dense_shapes = val_x_batch.shape
                            val_loss, val_pre, summary_str = sess.run(
                                [loss, outputs, merged_summary],
                                feed_dict={indices_i: val_indics,
                                           values_i: val_values,
                                           dense_shape_i: val_dense_shapes,
                                           labels: val_y_batch,
                                           is_train: False})
                            valid_writer.add_summary(summary_str, global_step)
                            bar.set_description('t_l:{:5.3f},v_l:{:5.3f}'.format(tr_loss, val_loss))
                        else:
                            sess.run(opt, feed_dict={indices_i: tr_indics,
                                                     values_i: tr_values,
                                                     dense_shape_i: tr_dense_shapes,
                                                     labels: tr_y_batch,
                                                     is_train: True})
                        global_step += 1
                        if global_step % cfg.checkpoint == 0:
                            saver.save(sess,
                                       cfg.logdir + '/model.ckpt-%02d-%02d-%05d-%05d' % (e, stage, i, global_step))
                    bar.close()
                saver.save(sess, cfg.logdir + '/model.ckpt-%02d-%02d-%05d-%05d' % (e+1, 4, 0, global_step))
            train_writer.close()
            valid_writer.close()


def evaluate(graph):
    if cfg.valid:
        # 获取上一次保存的状态
        ckpt, last_epoch, last_stage, local_step, global_step = get_last_state(cfg.logdir)
        # 读取数据
        X, y, data_size = data_input2.get_valid_data()

    else:  # 使用测试集生成submission
        # 获取上一次保存的状态
        ckpt, _, _, _, _ = get_last_state(cfg.logdir)
        # 读取数据
        X, data_size = data_input2.get_test_data(cfg.test_set)

    batch_num = data_size // cfg.batch
    if ckpt is None:
        print('No ckpt found!')
        return

    result = np.zeros([data_size, 1])
    with graph.as_default():

        indices_i = tf.placeholder(dtype=tf.int64, shape=[None, 2])
        values_i = tf.placeholder(dtype=tf.float32, shape=[None])
        dense_shape_i = tf.placeholder(dtype=tf.int64, shape=[2])
        feature = tf.SparseTensor(indices=indices_i, values=values_i, dense_shape=dense_shape_i)

        # 构造网络结构
        logits, outputs = build_arch(feature, cfg.hidden, [], False)

        init_op = tf.group([tf.global_variables_initializer(), tf.local_variables_initializer()])
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(init_op)

            print('load model: ', ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)

            print('computing result ...')
            bar = tqdm(range(0, batch_num + 1), total=batch_num, ncols=100, leave=False,
                       unit='b')
            test_idx = 0
            for i in bar:
                if i == batch_num:
                    x_batch = X[test_idx:].tocoo()
                else:
                    x_batch = X[test_idx: test_idx + cfg.batch].tocoo()
                indics = np.mat([x_batch.row, x_batch.col]).transpose()
                values = x_batch.data
                dense_shapes = x_batch.shape

                _outputs = sess.run(outputs, feed_dict={indices_i: indics,
                                                        values_i: values,
                                                        dense_shape_i: dense_shapes})
                if i == batch_num:
                    result[test_idx:] = _outputs
                    test_idx = data_size - 1
                else:
                    result[test_idx: test_idx + cfg.batch] = _outputs
                    test_idx += cfg.batch
                del _outputs

            bar.close()
            print('result shape:', result)
            print('scores length: ', len(result))  # 2265989
            if cfg.valid:
                print('computing auc ...')
                _, _, auc = utils.cal_auc(y, result)
                print('auc:', auc)
            else:
                print('reading res.csv ...')
                if cfg.test_set == 1:
                    test_data = pd.read_csv('data/output/test/res1.csv')
                else:
                    test_data = pd.read_csv('data/output/test/res2.csv')
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


def build_arch(feature, hide_layer, summary, is_training):
    with tf.name_scope('arch'):
        if cfg.arch == 1 or cfg.arch == 3:
            if cfg.feature == 1:  # lgb精简特征
                v = tf.Variable(tf.truncated_normal(shape=[data_input2.LGB_LEN, cfg.embed], stddev=0.01),
                                dtype=tf.float32, name='v')
            else:  # hash特征
                v = tf.Variable(tf.truncated_normal(shape=[data_input2.HASH_LEN, cfg.embed], stddev=0.01),
                                dtype=tf.float32, name='v')

            with tf.variable_scope('FM'):
                b = tf.get_variable('bias', shape=[1], initializer=tf.zeros_initializer())
                if cfg.feature == 1:
                    w = tf.get_variable('w', shape=[data_input2.LGB_LEN, 1],
                                        initializer=tf.truncated_normal_initializer(stddev=0.01))
                else:
                    w = tf.get_variable('w', shape=[data_input2.HASH_LEN, 1],
                                        initializer=tf.truncated_normal_initializer(stddev=0.01))

                linear_terms = tf.sparse_tensor_dense_matmul(feature, w)
                linear_terms = tf.add(linear_terms, b, name='linear')
                summary.append(tf.summary.histogram('linear_terms', linear_terms))

                part1 = tf.square(tf.sparse_tensor_dense_matmul(feature, v))
                part2 = tf.sparse_tensor_dense_matmul(tf.square(feature), tf.square(v))

                interaction_terms = tf.multiply(0.5,
                                                tf.reduce_mean(tf.subtract(part1, part2), 1, keep_dims=True), name='interaction')
                summary.append(tf.summary.histogram('interaction_terms', interaction_terms))

                y_fm = tf.add(linear_terms, interaction_terms, name='fm_out')
                summary.append(tf.summary.histogram('fm_outputs', y_fm))

        if cfg.arch == 2 or cfg.arch == 3:
            with tf.variable_scope('DNN', reuse=False):
                # embedding layer
                if cfg.feature == 1:  # lgb精简特征
                    dnn_v = tf.Variable(
                        tf.truncated_normal(shape=[data_input2.LGB_LEN, cfg.embed], mean=0, stddev=0.01),
                        dtype='float32')
                    dnn_input = tf.sparse_tensor_dense_matmul(feature, dnn_v)
                else:  # hash特征
                    dnn_v = tf.Variable(
                        tf.truncated_normal(shape=[data_input2.HASH_LEN, cfg.embed], mean=0, stddev=0.01),
                        dtype='float32')
                    dnn_input = tf.sparse_tensor_dense_matmul(feature, dnn_v)

                layer = dnn_input
                for i, num in enumerate(hide_layer):
                    layer = tf.layers.dense(layer, num, activation=None, use_bias=True, name='layer' + str(i+1))
                    layer = tf.layers.batch_normalization(layer, training=is_training, name='bn_layer' + str(i+1))
                    layer = tf.nn.relu(layer, name='act_layer' + str(i+1))
                    layer = tf.layers.dropout(layer, cfg.drop_out, is_training)
                    # summary.append(tf.summary.histogram('layer' + str(i+1), layer))

                y_dnn = tf.layers.dense(layer, 1, activation=None, use_bias=True, name='dnn_out')
            summary.append(tf.summary.histogram('dnn_outputs', y_dnn))

        if cfg.arch == 1:
            logits = y_fm
        elif cfg.arch == 2:
            logits = y_dnn
        else:
            logits = y_fm + y_dnn

        summary.append(tf.summary.histogram('logits', logits))
        outputs = tf.sigmoid(logits, name='final_outputs')
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
def get_last_state(logdir):
    ckpt = tf.train.get_checkpoint_state(logdir)
    if ckpt and ckpt.model_checkpoint_path:
        # Restores from checkpoint
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        steps = ckpt_name.split('-')
        epoch = int(steps[-4])
        stage = int(steps[-3])
        step = int(steps[-2])
        global_step = int(steps[-1])
    else:
        epoch = 0
        stage = 0
        step = 0
        global_step = 0
    return ckpt, epoch, stage, step, global_step


if __name__ == "__main__":
    tf.app.run()
