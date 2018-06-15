import tensorflow as tf

flags = tf.app.flags


############################
#    hyper parameters      #
############################
flags.DEFINE_string('mode', 'all', 'test:仅执行一次操作；all：一个完整的4stack；tv：train+valid')
flags.DEFINE_integer('arch',  2, '网络架构：1，仅fm；2，仅dnn；3，deepFM')  # deepFM效果最好，dnn次之，fm最差

flags.DEFINE_float('lr', 0.0001, '设置学习率')
flags.DEFINE_float('pos_weight', 1, '惩罚FN')
flags.DEFINE_integer('data_source', 6, '数据源：1，lgb精简特征2162维；2，lgb精简特征4236维；3，lgb精简特征6789维；'
                                       '4，lgb精简特征9098维；5，lgb精简特征13475维；6，lgb精简特征9098维+ctr；'
                                       '7，lgb精简特征13475维+ctr')
flags.DEFINE_integer('fold', 1, 'k折训练的阶段：1，在234训练，在1验证；2，在134训练，在2验证；3，在124训练，在3验证；4在123训练，在4验证')

# flags.DEFINE_float('drop_out', 0.5, 'drop out比率')
flags.DEFINE_integer('batch', 1024, '设置批大小')
flags.DEFINE_integer('epoch', 1, '设置训练的轮数')
flags.DEFINE_integer('checkpoint', 1000, '每隔多少个批次保存一次模型')
flags.DEFINE_integer('summary', 250, '每隔多少个批次记录一次日志')
flags.DEFINE_boolean('train', False, '选择是训练还是推理')
flags.DEFINE_boolean('valid', False, '是否在训练中做交叉验证')  # 在train为False的前提下，valid如果为False，则生成submission，valid为True，在验证集上评估auc
flags.DEFINE_string('logdir', 'fm_dnn_9098_ctr_15_', '日志保存路径')

last_feature = 15
flags.DEFINE_integer('ctr_feature', 25, '用于知识蒸馏的神经元数量')
flags.DEFINE_integer('last_feature', last_feature, '用于知识蒸馏的神经元数量')
flags.DEFINE_list('hidden', [512, 512, 512, 512, last_feature], '设置隐藏层结构')
flags.DEFINE_list('fm_embed', [512], 'fm部分的隐向量单元数')
flags.DEFINE_integer('embed', 512, '各嵌入层的单元数')

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

# Uncomment this line to run in debug mode
# tf.logging.set_verbosity(tf.logging.INFO)
