import tensorflow as tf
import os
# 进度条工具
from tqdm import tqdm
# 超参配置文件
from config import cfg

import data_input


def main(_):
    tf.reset_default_graph()
    graph = tf.Graph()
    summary = []
    with graph.as_default():
        # 从文件流读取数据
        feature, label = data_input.get_data(cfg.training, cfg.batch, summary)
        # 构造网络结构
        logits, outputs = build_arch(feature, cfg.hidden, summary)

        if cfg.training:
            tf.logging.info(' Start training...')
            train(model)
            tf.logging.info('Training done')
        else:
            evaluation(model)


def train():
    tf.reset_default_graph()
    X_train, y_train = read_data('data/train_user_ad_base.csv', batch_size=batch_size, num_epochs=epoch)
    print(X_train.shape)
    print(y_train.shape)
    outputs = build_arch(X_train)
    # loss = choose_loss(outputs, y_train, gammar, power_p)
    loss = build_cross_entropy_loss(outputs, y_train)
    score = tf.reduce_mean(loss)  # build_score(outputs, y_train)
    # 直接调用tensorflow的metric.auc计算近似的AUC
    corss_score, corss_val_opt = tf.metrics.auc(y_train, (outputs))
    opt = tf.train.AdamOptimizer(lr).minimize(loss)

    # ### 训练

    # In[ ]:

    num_batch = 100000 // batch_size  # 训练集大小//batch大小
    ckpt, global_step, last_epoch, last_step = get_last_state(logdir, num_batch)
    init_op = tf.global_variables_initializer()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init_op)
        tf.local_variables_initializer().run()  # 对loacl变量初始化后，文件读取不会报错
        tf.train.start_queue_runners(sess=sess)  # train的文件队列开始填充
        train_writer = tf.summary.FileWriter(logdir + '/train', sess.graph)

        if ckpt and ckpt.model_checkpoint_path:
            # 加载上次保存的模型
            saver.restore(sess, ckpt.model_checkpoint_path)
        # 计算图结构分析
        param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(
            tf.get_default_graph(),
            tfprof_options=tf.contrib.tfprof.model_analyzer.TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
        print('total_params: %d\n' % param_stats.total_parameters)

        for e in range(last_epoch, epoch):
            print('Training for epoch ' + str(e + 1) + '/' + str(epoch) + ':')

            bar = tqdm(range(last_step, num_batch), initial=last_step, total=num_batch, ncols=100, leave=False,
                       unit='b')
            for _ in bar:
                if global_step % save_summaries_steps == 0:
                    # train
                    # _, train_score, summary_str = sess.run(
                    #        [opt, score, summary])
                    _, train_score = sess.run(
                        [opt, score])
                    # train_writer.add_summary(summary_str, global_step)
                    bar.set_description('tr_acc:{}'.format(train_score))

                    # 交叉验证：从训练集中读出batch_size大小的数据，进行交叉验证
                    validation_score, _ = sess.run(
                        [corss_score, corss_val_opt])
                    print('cross_validation_score:', validation_score, '\t')
                else:
                    sess.run(opt)

                global_step += 1
                if global_step % save_checkpoint_steps == 0:
                    saver.save(sess, logdir + '/model.ckpt', global_step=global_step)

        train_writer.close()
        print('Starting prediction')


def evaluate():


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
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels, logits))
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
