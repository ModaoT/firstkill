# coding: utf-8

# In[1]:
'''
采用tensorflow Dataset类读入数据；
实现AUC loss函数
调用tf.metric.auc计算validation的score
可以直接在"train_user_ad_base.csv"上训练
问题：
1.文件剩余行数小于batch_size时可能出问题，因为每次是batchsize的整数倍读出。
2.训练结果只有0.5左右。。。代码需要review。
'''

# 用于测试集的划分
from sklearn.model_selection import train_test_split
import tensorflow as tf
import os
# 进度条工具
from tqdm import tqdm


# In[ ]:


# 继承tensorflow Dataset类，使用TextLineReader构建iterator从csv文件读取数据
'''
读数据参考
https://zhuanlan.zhihu.com/p/27238630
https://medium.com/google-cloud/how-to-do-time-series-prediction-using-rnns-and-tensorflow-and-cloud-ml-engine-2ad2eeb189e8
https://blog.csdn.net/lujiandong1/article/details/53376802
'''
def read_data(filename,  batch_size, num_epochs, reading_format = [[.0] for x in range(0, 20)], mode = tf.contrib.learn.ModeKeys.TRAIN):
    input_file_names = tf.train.match_filenames_once(filename)
    #产生文件队列
    filename_queue = tf.train.string_input_producer(
        input_file_names, num_epochs=num_epochs, shuffle=True)

    #对于训练集的读取
    if mode == tf.contrib.learn.ModeKeys.TRAIN:
        reader =  tf.TextLineReader(skip_header_lines=1)
        _, value = reader.read_up_to(filename_queue, num_records=batch_size)
        #此处还要处理，当文件行数不够一个batch_size的情况，尚未处理
        value_column = tf.expand_dims(value, -1)

        # 用decode_csv解读出来的是list类型，需要后面concat
        # record_defaults: A list of Tensor objects with specific types.
        # Acceptable types are float32, float64, int32, int64, string.
        # One tensor per column of the input record, with either a scalar default value
        # for that column or empty if the column is required.
        # default = [[.0] for x in range(0, 20)]. 20是csv的列数
        all_data = tf.decode_csv(value_column, record_defaults=reading_format)
        inputs = all_data[3:]  # first seq_length values
        label = all_data[2:3]
        # 从list转换为tensor
        inputs = tf.concat(inputs, axis=1)
        label = tf.concat(label, axis=1)
        # 对于-1的label转换为0，便于后面调用tf.metric，以及构建最大似然loss函数
        label = tf.where(tf.greater(label, .0),label,tf.zeros([batch_size,1]))
        return inputs, label
    else:
        # 以下需要根据test集合更改，尚未更改
        reader =  tf.TextLineReader(skip_header_lines=1)
        _, value = reader.read_up_to(filename_queue, num_records=batch_size)

        value_column = tf.expand_dims(value, -1)

        # all_data is a list of tensors
        all_data = tf.decode_csv(value_column, record_defaults=reading_format)
        inputs = all_data[3:]  # first seq_length values
        label = all_data[2:3]
        # from list of tensors to tensor with one more dimension
        inputs = tf.concat(inputs, axis=1)
        label = tf.concat(label, axis=1)
        return inputs, label


# ## 通过得到的特征向量与标签，用tensorflow搭建神经网络进行训练

# In[ ]:


# 设计神经网络结构
def build_arch(inputs):
    layer1 = tf.layers.dense(inputs, 128, activation=tf.nn.relu)
    layer2 = tf.layers.dense(layer1, 256, activation=tf.nn.relu)
    layer3 = tf.layers.dense(layer2, 56, activation=tf.nn.relu)
    outputs = tf.layers.dense(layer3, 1, activation=tf.nn.relu)
    
    return outputs


# In[ ]:
def build_cross_entropy_loss(outputs, labels):
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=outputs))
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
        posi_idx = tf.where(tf.equal(labels_temp, 1.0))[:, 0]
        neg_idx = tf.where(tf.equal(labels_temp, .0))[:, 0]

        prdictions = tf.nn.softmax(outputs)
        # 获取正负样本位置对应的预测值
        posi_predict = tf.gather(prdictions, posi_idx)
        posi_size = tf.shape(posi_predict)[0]
        neg_predict = tf.gather(prdictions, neg_idx)
        neg_size = tf.shape(neg_predict)[0]
        # 按照论文'Optimizing Classifier Performance
        # via an Approximation to the Wilcoxon-Mann-Whitney Statistic'中的公式(7)计算loss_function
        posi_neg_diff = tf.reshape(
            -(tf.matmul(posi_predict, tf.ones([1, neg_size])) -
              tf.matmul(tf.ones([posi_size, 1]), tf.reshape(neg_predict, [1, neg_size])) - gammar),
            [-1, 1]
        )
        posi_neg_diff = tf.where(tf.greater(posi_neg_diff, 0), posi_neg_diff, tf.zeros([posi_size * neg_size, 1]))
        posi_neg_diff = tf.pow(posi_neg_diff, power_p)

        loss_approx_auc = tf.reduce_mean(posi_neg_diff)
        return loss_approx_auc
    #此处根据是否包含负样本，tf.cond选择使用的loss函数
    used_loss_function = tf.cond(tf.greater(tf.reduce_sum(labels),1), build_AUC_loss, build_loss)
    return used_loss_function



# In[ ]:


# 设计分数统计函数
def build_score(outputs, labels):
    score = 0
    return score


# In[ ]:


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


# ### 构建计算图

# In[ ]:


# 调试超参
lr = 0.05
batch_size = 1000
epoch = 60
logdir = 'log/'
save_checkpoint_steps = 100
save_summaries_steps = 100
#AUC parameter
gammar = 0.3
power_p = 2


# In[1]:


tf.reset_default_graph()
X_train, y_train = read_data('data/train_user_ad_base.csv', batch_size=batch_size, num_epochs=epoch)
print(X_train.shape)
print(y_train.shape)
outputs = build_arch(X_train)
#loss = choose_loss(outputs, y_train, gammar, power_p)
loss = build_cross_entropy_loss(outputs, y_train)
score = tf.reduce_mean(loss)#build_score(outputs, y_train)
#直接调用tensorflow的metric.auc计算近似的AUC
corss_score, corss_val_opt = tf.metrics.auc(y_train,(outputs))
opt = tf.train.AdamOptimizer(lr).minimize(loss)


# ### 训练

# In[ ]:


num_batch = 100000//batch_size #训练集大小//batch大小
ckpt, global_step, last_epoch, last_step = get_last_state(logdir, num_batch)
init_op = tf.global_variables_initializer()

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init_op)
    tf.local_variables_initializer().run()#对loacl变量初始化后，文件读取不会报错
    tf.train.start_queue_runners(sess=sess) #train的文件队列开始填充
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
        print('Training for epoch ' + str(e+1) + '/' + str(epoch) + ':')
        
        bar = tqdm(range(last_step, num_batch), initial=last_step, total=num_batch, ncols=100, leave=False,
                       unit='b')
        for _ in bar:
            if global_step % save_summaries_steps == 0:
                # train
                #_, train_score, summary_str = sess.run(
                #        [opt, score, summary])
                _, train_score = sess.run(
                    [opt, score])
                #train_writer.add_summary(summary_str, global_step)
                bar.set_description('tr_acc:{}'.format(train_score))

                #交叉验证：从训练集中读出batch_size大小的数据，进行交叉验证
                validation_score, _ = sess.run(
                    [corss_score, corss_val_opt ])
                print('cross_validation_score:',validation_score,'\t')
            else:
                sess.run(opt)

            global_step += 1
            if global_step % save_checkpoint_steps == 0:
                saver.save(sess, logdir + '/model.ckpt', global_step=global_step)

    train_writer.close()
    print('Starting prediction')


