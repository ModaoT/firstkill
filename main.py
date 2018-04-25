import tensorflow as tf
import pandas as pd
import os
# 进度条工具
from tqdm import tqdm
# 超参配置文件
from config import cfg

import data_input


TRAIN_NUM = 8798814
TEST_NUM = 2265989


def main(_):
    tf.reset_default_graph()
    graph = tf.Graph()
    summary = []
    with graph.as_default():
        # 从文件流读取数据
        features, labels = data_input.get_data(summary, cfg.train, cfg.batch)
        # 构造网络结构
        logits, outputs = build_arch(features, cfg.hidden, summary)
        # 构造损失函数
        loss = build_loss(labels, logits, summary)
        # 获取上一次保存的状态
        train_batch = TRAIN_NUM // cfg.batch  # 训练集大小//batch大小
        test_batch = TEST_NUM // cfg.batch  # 测试集大小//batch大小
        ckpt, global_step, last_epoch, last_step = get_last_state(cfg.logdir, train_batch)

        if cfg.train:
            # 直接调用tensorflow的metric.auc计算近似的AUC
            auc, update_op = tf.metrics.auc(labels, outputs)
            summary.append(tf.summary.scalar('train_auc', auc))

            merged_summary = tf.summary.merge(summary)
            # 构造学习器
            opt = tf.train.AdamOptimizer(cfg.lr).minimize(loss)

            init_op = tf.group([tf.global_variables_initializer(), tf.local_variables_initializer()])
            saver = tf.train.Saver()
            with tf.Session() as sess:
                sess.run(init_op)

                train_writer = tf.summary.FileWriter(cfg.logdir + '/train', sess.graph)

                if ckpt and ckpt.model_checkpoint_path:
                    # 加载上次保存的模型
                    saver.restore(sess, ckpt.model_checkpoint_path)
                # 计算图结构分析
                param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(
                    tf.get_default_graph(),
                    tfprof_options=tf.contrib.tfprof.model_analyzer.TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
                print('total_params: %d\n' % param_stats.total_parameters)

                for e in range(last_epoch, cfg.epoch):
                    print('Training for epoch ' + str(e + 1) + '/' + str(cfg.epoch) + ':')

                    bar = tqdm(range(last_step, train_batch), initial=last_step, total=train_batch, ncols=100, leave=False,
                               unit='b')
                    for _ in bar:
                        if global_step % cfg.summary == 0:
                            # train
                            # _, train_score, summary_str = sess.run(
                            #        [opt, score, summary])
                            train_loss, _, train_auc, summary_str = sess.run(
                                [loss, opt, auc, merged_summary])
                            train_writer.add_summary(summary_str, global_step)
                            bar.set_description('loss:{}, auc:{}'.format(train_loss, train_auc))
                        else:
                            sess.run(opt)

                        global_step += 1
                        if global_step % cfg.checkpoint == 0:
                            saver.save(sess, cfg.logdir + '/model.ckpt', global_step=global_step)

                    bar.close()

                train_writer.close()

        else:
            result = pd.Series([])

            init_op = tf.group([tf.global_variables_initializer(), tf.local_variables_initializer()])
            saver = tf.train.Saver()
            with tf.Session() as sess:
                sess.run(init_op)

                saver.restore(sess, ckpt.model_checkpoint_path)

                bar = tqdm(range(0, test_batch), total=test_batch, ncols=100, leave=False,
                           unit='b')
                for _ in bar:
                    outputs = sess.run([outputs])
                    result.append(outputs)

                bar.close()

            test_data = pd.read_csv('data/test_ad_user_all.csv')
            test_data['score'] = result
            test_data[['aid', 'uid', 'score']].to_csv('data/submission.csv', columns=['aid', 'uid', 'score'], index=False)


def train():
    pass


def evaluate():
    pass


def build_arch(inputs, hide_layer, summary):
    with tf.name_scope('arch'):
        layer = inputs
        for i, num in enumerate(hide_layer):
            layer = tf.layers.dense(layer, num, activation=tf.nn.relu)
            summary.append(tf.summary.histogram('layer' + str(i+1), layer))

        logits = tf.layers.dense(layer, 1, activation=None)  # logits是用于损失函数的输入，无需activation
        summary.append(tf.summary.histogram('logits', logits))
        outputs = tf.sigmoid(logits)
        summary.append(tf.summary.histogram('outputs', outputs))
    return logits, outputs


def build_loss(labels, logits, summary):
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(labels, tf.float32), logits=logits))
    summary.append(tf.summary.scalar('loss', loss))
    return loss

#batch中包含正负两类样本时，使用AUC_loss，否则使用最大似然交叉熵
def choose_loss(outputs, labels, gammar, power_p):
    # 设计损失函数
    def build_loss():
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=outputs))
        return loss
    # In[ ]:
    # approximated AUC loss function
    def build_AUC_loss():
        #获取正负样本的索引位置
        labels_temp = labels[:, 0]
        pos_idx = tf.where(tf.equal(labels_temp, 1.0))[:, 0]
        neg_idx = tf.where(tf.equal(labels_temp, .0))[:, 0]

        predictions = tf.nn.softmax(outputs)
        # 获取正负样本位置对应的预测值
        pos_predict = tf.gather(predictions, pos_idx)
        pos_size = tf.shape(pos_predict)[0]
        neg_predict = tf.gather(predictions, neg_idx)
        neg_size = tf.shape(neg_predict)[0]
        # 按照论文'Optimizing Classifier Performance
        # via an Approximation to the Wilcoxon-Mann-Whitney Statistic'中的公式(7)计算loss_function
        pos_neg_diff = tf.reshape(
            -(tf.matmul(pos_predict, tf.ones([1, neg_size])) -
              tf.matmul(tf.ones([pos_size, 1]), tf.reshape(neg_predict, [1, neg_size])) - gammar),
            [-1, 1]
        )
        pos_neg_diff = tf.where(tf.greater(pos_neg_diff, 0), pos_neg_diff, tf.zeros([pos_size * neg_size, 1]))
        pos_neg_diff = tf.pow(pos_neg_diff, power_p)

        loss_approx_auc = tf.reduce_mean(pos_neg_diff)
        return loss_approx_auc
    #此处根据是否包含负样本，tf.cond选择使用的loss函数
    used_loss_function = tf.cond(tf.greater(tf.reduce_sum(labels),1), build_AUC_loss, build_loss)
    return used_loss_function


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
