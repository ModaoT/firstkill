import os

import numpy as np
import pandas as pd
import tensorflow as tf
# 进度条工具
from tqdm import tqdm
import utils
from scipy import sparse

import feather



flags = tf.app.flags


############################
#    hyper parameters      #
############################
flags.DEFINE_string('mode', 'tv', 'test:仅执行一次操作；all：一个完整的4stack；tv：train+valid')

flags.DEFINE_integer('arch',  1, '网络架构：1，仅fm；2，cnn；3，deepFM')
flags.DEFINE_float('lr', 0.0001, '设置学习率')
flags.DEFINE_float('pos_weight', 1, '惩罚FN')
flags.DEFINE_integer('data_source', 1, '数据源：1，cnt 78维+2162 lgb；')
flags.DEFINE_integer('fold', 1, 'k折训练的阶段：1，在234训练，在1验证；2，在134训练，在2验证；3，在124训练，在3验证；4在123训练，在4验证')

# flags.DEFINE_float('drop_out', 0.5, 'drop out比率')
flags.DEFINE_integer('batch', 1024, '设置批大小')
flags.DEFINE_integer('epoch', 1, '设置训练的轮数')
# flags.DEFINE_integer('checkpoint', 1000, '每隔多少个批次保存一次模型')
flags.DEFINE_integer('summary', 250, '每隔多少个批次记录一次日志')
flags.DEFINE_boolean('train', False, '选择是训练还是推理')
flags.DEFINE_boolean('valid', False, '是否在训练中做交叉验证')  # 在train为False的前提下，valid如果为False，则生成submission，valid为True，在验证集上评估auc
flags.DEFINE_string('logdir', 'df_cnt_lgb_sparse_', '日志保存路径')



flags.DEFINE_list('hidden', [512,512,256,256], '设置隐藏层结构')
flags.DEFINE_integer('fm_embed', 512, 'fm部分的隐向量单元数')

"""
K_fold交叉验证即：
1，用234训练，在1上验证（通过改变fold参数，logdir也要改名，以区分不同模型）
2，用134训练，在2上验证
3，用124训练，在3上验证
4，用123训练，在4上验证
5，4个模型分别在test上预测，取平均值作为test结果
6，在1234上验证中计算出的结果，可以作为新的训练集的特征，以备后期模型融合
"""


cfg = tf.app.flags.FLAGS


if cfg.data_source == 1:
    DATA = '_cnt78_2162.npz'


def read_data(dir, cls):
    y_file = None
    y = None
    if dir == 'test':
        directory = 'data/test/'
    elif dir == 'tr1':
        directory = 'data/tr1/'
    elif dir == 'tr2':
        directory = 'data/tr2/'
    elif dir == 'tr3':
        directory = 'data/tr3/'
    elif dir == 'tr4':
        directory = 'data/tr4/'
    else:
        return None

    if cls == 'tr':
        # X_file = dir + '_9_2162.npz'
        X1_file = dir + DATA
        y_file = dir + '_y.csv'
    elif cls == 'test1':
        X1_file = dir+'1'+ DATA
    elif cls == 'test2':
        X1_file = dir+'2'+ DATA
    else:
        return None

    print('x1:',directory + X1_file)
    X1 = sparse.load_npz(directory + X1_file)
    if y_file is not None:
        y = pd.read_csv(directory + y_file).values

    return X1, y


def get_test_data(test_set):
    print('reading test data...')
    print('test set：', test_set)
    if test_set == 1:
        test_x_1, _ = read_data('test', 'test1')
    elif test_set == 2:
        test_x_1, _ = read_data('test', 'test2')
    else:
        return None
    return test_x_1


def get_valid_data(fold):
    print('reading valid data from fold', fold, '...')
    if fold == 1:
        val_x1, val_y = read_data('tr1', 'tr')
    elif fold == 2:
        val_x1, val_y = read_data('tr2', 'tr')
    elif fold == 3:
        val_x1, val_y = read_data('tr3', 'tr')
    elif fold == 4:
        val_x1, val_y = read_data('tr4', 'tr')
    else:
        return None
    print('x1 shape:', val_x1.shape)
    return val_x1, val_y, val_y.shape[0]


def get_train_data(fold, stage=0):
    print('reading train data from fold', fold, '...')
    if fold == 1:
        tr = ['tr2', 'tr3', 'tr4']
    elif fold == 2:
        tr = ['tr1', 'tr3', 'tr4']
    elif fold == 3:
        tr = ['tr1', 'tr2', 'tr4']
    elif fold == 4:
        tr = ['tr1', 'tr2', 'tr3']
    else:
        return None
    tr_x1, tr_y = read_data(tr[stage], 'tr')
    print('x1 shape:', tr_x1.shape)
    return tr_x1, tr_y, tr_y.shape[0]


def main(_):
    if cfg.mode == 'test':
        tf.reset_default_graph()
        graph = tf.Graph()
        logdir = cfg.logdir + str(cfg.fold)
        if cfg.train:
            train(graph, logdir, cfg.fold, cfg.epoch)
        else:
            evaluate(graph, cfg.valid, logdir, cfg.fold)
    elif cfg.mode == 'all':
        for i in range(4):
            # 训练
            tf.reset_default_graph()
            graph = tf.Graph()
            logdir = cfg.logdir + str(i+1)
            fold = i+1
            train(graph, logdir, fold, cfg.epoch)
            # 验证
            tf.reset_default_graph()
            graph = tf.Graph()
            evaluate(graph, True, logdir, fold)
            # 输出结果
            tf.reset_default_graph()
            graph = tf.Graph()
            evaluate(graph, False, logdir, fold)
    elif cfg.mode == 'tv':
        # 训练
        tf.reset_default_graph()
        graph = tf.Graph()
        logdir = cfg.logdir + str(cfg.fold)
        train(graph, logdir, cfg.fold, cfg.epoch)
        # 验证
        tf.reset_default_graph()
        graph = tf.Graph()
        evaluate(graph, True, logdir, cfg.fold)


def build_graph(train, hidden):
    with tf.name_scope('Input'):
        indices_i = tf.placeholder(dtype=tf.int64, shape=[None, 2])
        values_i = tf.placeholder(dtype=tf.float32, shape=[None])
        dense_shape_i = tf.placeholder(dtype=tf.int64, shape=[2])
        feature = tf.SparseTensor(indices=indices_i, values=values_i, dense_shape=dense_shape_i)

        if train:
            labels = tf.placeholder(dtype=tf.int32, shape=[None, 1], name='labels')
            is_train = tf.placeholder(dtype=tf.bool, shape=[])

    # 构造网络结构
    summary = []
    logits, outputs = build_arch(feature, hidden, summary, is_train if train else False)

    if train:
        # 构造损失函数
        loss = build_loss(labels, logits, summary)
        merged_summary = tf.summary.merge(summary)

        # 构造学习器
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):  # 为了batch normalization能正常运行
            opt = tf.train.AdamOptimizer(cfg.lr).minimize(loss)

    init_op = tf.group([tf.global_variables_initializer(), tf.local_variables_initializer()])
    saver = tf.train.Saver(max_to_keep=5)
    if train:
        return indices_i, values_i, dense_shape_i, labels, is_train, loss, outputs, merged_summary, opt, init_op, saver
    else:
        return indices_i, values_i, dense_shape_i, init_op, saver, outputs


def train(graph, logdir, fold, epoch):
    # 获取上一次保存的状态
    ckpt, last_epoch, last_stage, local_step, global_step = get_last_state(logdir)

    with graph.as_default():
        cnt_i, labels, is_train, loss, outputs, merged_summary, opt, init_op, saver = build_graph(True, cfg.hidden)
        with tf.Session() as sess:
            sess.run(init_op)
            train_writer = tf.summary.FileWriter(logdir + '/train', sess.graph)
            valid_writer = tf.summary.FileWriter(logdir + '/valid')
            if ckpt and ckpt.model_checkpoint_path:
                # 加载上次保存的模型
                print('load model: ', ckpt.model_checkpoint_path)
                saver.restore(sess, ckpt.model_checkpoint_path)
            # 计算图结构分析
            param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(
                tf.get_default_graph(),
                tfprof_options=tf.contrib.tfprof.model_analyzer.TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
            print('total_params: %d\n' % param_stats.total_parameters)

            for e in range(last_epoch, epoch):
                print('Training for epoch ' + str(e + 1) + '/' + str(epoch) + ':')
                if cfg.valid:
                    val_x1, val_y, val_size = get_valid_data(fold)
                for stage in range(last_stage, 3):
                    print()
                    print('stage', stage, ':')
                    tr_x1, tr_y, tr_size = get_train_data(fold, stage)
                    train_batch = tr_size // cfg.batch
                    bar = tqdm(range(local_step, train_batch+1), initial=last_epoch, total=train_batch, ncols=150, leave=False,
                               unit='b')
                    tr_idx = 0
                    val_idx = 0
                    for i in bar:
                        if i == train_batch:
                            tr_x1_batch = tr_x1[tr_idx:] / 255.
                            tr_y_batch = tr_y[tr_idx:]
                            tr_idx = tr_size-1
                        else:
                            tr_x1_batch = tr_x1[tr_idx: tr_idx+cfg.batch] / 255.
                            tr_y_batch = tr_y[tr_idx: tr_idx+cfg.batch]
                            tr_idx += cfg.batch

                        if i % cfg.summary == 0:
                            tr_loss, tr_pre, summary_str = sess.run(
                                [loss, outputs, merged_summary],
                                feed_dict={cnt_i: tr_x1_batch,
                                           labels: tr_y_batch,
                                           is_train: False})
                            train_writer.add_summary(summary_str, global_step)
                            bar.set_description('t_l:{:5.3f}'.format(tr_loss))
                            if cfg.valid:
                                if val_idx + cfg.batch >= val_size:
                                    val_idx = 0
                                val_x1_batch = val_x1[val_idx: val_idx + cfg.batch] / 255.
                                val_y_batch = val_y[val_idx: val_idx + cfg.batch]
                                val_idx += cfg.batch
                                val_loss, val_pre, summary_str = sess.run(
                                    [loss, outputs, merged_summary],
                                    feed_dict={cnt_i: val_x1_batch,
                                               labels: val_y_batch,
                                               is_train: False})
                                valid_writer.add_summary(summary_str, global_step)
                                bar.set_description('t_l:{:5.3f},v_l:{:5.3f}'.format(tr_loss, val_loss))
                        else:
                            sess.run(opt, feed_dict={cnt_i: tr_x1_batch,
                                                     labels: tr_y_batch,
                                                     is_train: True})
                        global_step += 1
                        # if global_step % cfg.checkpoint == 0:
                        #     saver.save(sess,
                        #                logdir + '/model.ckpt-%02d-%02d-%05d-%05d' % (e, stage, i, global_step))
                    bar.close()
                    saver.save(sess,
                               logdir + '/model.ckpt-%02d-%02d-%05d-%05d' % (e, stage+1, 0, global_step))
                    del tr_x1, tr_y
                    print()
                    print('stage', stage, 'finished!')
                if cfg.valid:
                    del val_x1, val_y
                saver.save(sess, logdir + '/model.ckpt-%02d-%02d-%05d-%05d' % (e+1, 0, 0, global_step))
            train_writer.close()
            valid_writer.close()


def evaluate(graph, valid, logdir, fold):
    # 获取上一次保存的状态
    ckpt, _, _, _, _ = get_last_state(logdir)
    if ckpt is None:
        print('No ckpt found!')
        return

    if valid:        # 读取数据
        X1_valid, y, data_size_valid = get_valid_data(fold)
        batch_num_valid = data_size_valid // cfg.batch

        result_valid = np.zeros([data_size_valid, 1])

    else:  # 使用测试集生成submission
        # 读取数据
        # data_size_test1 = 11729073
        data_size_test2 = 11727304
        # X1_test1, X2_test1 = get_test_data(1)
        X1_test2 = get_test_data(2)
        # batch_num_test1 = data_size_test1 // cfg.batch
        batch_num_test2 = data_size_test2 // cfg.batch

        # result_test1 = np.zeros([data_size_test1, 1])
        result_test2 = np.zeros([data_size_test2, 1])

    with graph.as_default():
        cnt_i, init_op, saver, outputs = build_graph(False, cfg.hidden)
        with tf.Session() as sess:
            sess.run(init_op)
            print('load model: ', ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
            print()
            if valid:
                print('computing valid result ...')
                result_valid = predict(sess, batch_num_valid, result_valid, data_size_valid, X1_valid, outputs, cnt_i)
                print()
                print('result_valid shape:', result_valid.shape)
                print('computing auc ...')
                _, _, auc = utils.cal_auc(y, result_valid)
                print('auc:', auc)
                print('saving result to {}/tr{}_predict_{:.6f}.csv ...'.format(logdir, fold, auc))
                prob = pd.DataFrame({'prob': result_valid[:, 0].reshape([data_size_valid, ])})
                prob.prob = prob.prob.apply(lambda x: float('%.6f' % x))
                prob.to_csv('{}/tr{}_predict_{:.6f}.csv'.format(logdir, fold, auc), index=False)
            else:
                # print('computing test result1 ...')
                # result_test1 = predict(sess, batch_num_test1, result_test1, data_size_test1, (X1_test1, X2_test1), outputs,
                #                        indices_i, values_i, dense_shape_i, cnt_i)
                # print()
                # print('result_test1 shape:', result_test1.shape)
                # del X1_test1, X2_test1
                print('computing test result2 ...')
                result_test2 = predict(sess, batch_num_test2, result_test2, data_size_test2, X1_test2, outputs, cnt_i)
                print()
                print('result_test2 shape:', result_test2.shape)

                del X1_test2
                # print('reading res1.csv ...')
                # test_data1 = pd.read_csv('data/test/res1.csv')
                # test_data1['score'] = np.array(result_test1)
                # test_data1['score'] = test_data1['score'].apply(lambda x: float('%.6f' % x))
                # print('writing results into {}/test1_predict{}.csv'.format(logdir, fold))
                # test_data1[['aid', 'uid', 'score']].to_csv('{}/test1_predict{}.csv'.format(logdir, fold),
                #                                            columns=['aid', 'uid', 'score'],
                #                                            index=False)

                print('reading res2.csv ...')
                test_data2 = pd.read_csv('data/test/res2.csv')
                test_data2['score'] = np.array(result_test2)
                test_data2['score'] = test_data2['score'].apply(lambda x: float('%.6f' % x))
                print('writing results into {}/test2_predict{}.csv'.format(logdir, fold))
                test_data2[['aid', 'uid', 'score']].to_csv('{}/test2_predict{}.csv'.format(logdir, fold),
                                                           columns=['aid', 'uid', 'score'],
                                                           index=False)

        print('finish')


def predict(sess, batch_num, result, data_size, X, outputs, ctr_i):
    bar = tqdm(range(0, batch_num + 1), total=batch_num, ncols=100, leave=False,
                unit='b')
    test_idx = 0
    for i in bar:
        if i == batch_num:
            x1_batch = X[test_idx:] / 255.
        else:
            x1_batch = X[test_idx: test_idx + cfg.batch] / 255.

        _outputs = sess.run(outputs, feed_dict={ctr_i: x1_batch})
        if i == batch_num:
            result[test_idx:] = _outputs
            test_idx = data_size - 1
        else:
            result[test_idx: test_idx + cfg.batch] = _outputs
            test_idx += cfg.batch
        del _outputs
    bar.close()
    return result


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
    if cfg.arch == 1 or cfg.arch == 3:
        v = tf.Variable(tf.truncated_normal(shape=[cfg.feature_len, cfg.fm_embed], stddev=0.1),
                        dtype=tf.float32, name='v')

        with tf.variable_scope('FM'):
            b = tf.get_variable('bias', shape=[1], initializer=tf.zeros_initializer())
            w = tf.get_variable('w', shape=[cfg.feature_len, 1],
                                initializer=tf.truncated_normal_initializer(stddev=0.1))

            linear_terms = tf.sparse_tensor_dense_matmul(feature, w)
            linear_terms = tf.add(linear_terms, b, name='linear')
            summary.append(tf.summary.histogram('linear_terms', linear_terms))

            part1 = tf.square(tf.sparse_tensor_dense_matmul(feature, v))
            part2 = tf.sparse_tensor_dense_matmul(tf.square(feature), tf.square(v))

            interaction_terms = tf.multiply(0.5,
                                            tf.reduce_mean(tf.subtract(part1, part2), 1, keepdims=True),
                                            name='interaction')
            summary.append(tf.summary.histogram('interaction_terms', interaction_terms))

            y_fm = tf.add(linear_terms, interaction_terms, name='fm_out')
            # y_fm = tf.concat([linear_terms, interaction_terms], axis=1, name='fm_out')
            summary.append(tf.summary.histogram('fm_outputs', y_fm))

    if cfg.arch == 2 or cfg.arch == 3:
        with tf.variable_scope('DNN', reuse=False):
            # embedding layer
            dnn_v = tf.Variable(
                tf.truncated_normal(shape=[cfg.feature_len, cfg.embed], mean=0, stddev=0.01),
                dtype='float32')
            dnn_input = tf.sparse_tensor_dense_matmul(feature, dnn_v)

            layer = dnn_input
            for i, num in enumerate(hide_layer):
                layer = tf.layers.dense(layer, num, activation=None, use_bias=True, name='layer' + str(i + 1))
                layer = tf.layers.batch_normalization(layer, training=is_training, name='bn_layer' + str(i + 1))
                layer = tf.nn.relu(layer, name='act_layer' + str(i + 1))
                # layer = tf.layers.dropout(layer, cfg.drop_out, is_training)
                summary.append(tf.summary.histogram('layer' + str(i+1), layer))

            if cfg.arch == 2:
                y_dnn = tf.layers.dense(layer, 1, activation=None, use_bias=True, name='dnn_out')
            else:
                y_dnn = layer
        summary.append(tf.summary.histogram('dnn_outputs', y_dnn))

    if cfg.arch == 1:
        logits = y_fm
    elif cfg.arch == 2:
        logits = y_dnn
    else:
        # logits = y_fm + y_dnn
        logits = tf.concat([y_fm, y_dnn], axis=1)
        # logits = tf.layers.batch_normalization(logits, training=is_training)
        logits = tf.layers.dense(logits, 1, activation=None, use_bias=True, name='logits')

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



