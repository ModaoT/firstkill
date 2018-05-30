import tensorflow as tf

flags = tf.app.flags


############################
#    hyper parameters      #
############################

flags.DEFINE_float('lr', 0.0001, '设置学习率')
flags.DEFINE_float('pos_weight', 1, '惩罚FN')
flags.DEFINE_integer('feature', 2, '选择数据类型：1，经lgb筛选过的特征；2，list特征映射为hash等长向量')  # 从调试结果看，hash效果不好
flags.DEFINE_integer('arch', 3, '网络架构：1，仅fm；2，仅dnn；3，deepFM')
flags.DEFINE_integer('fold', 1, 'k折训练的阶段：1，在234训练，在1验证；2，在134训练，在2验证；3，在124训练，在3验证；4在123训练，在4验证')
flags.DEFINE_integer('test_set', 1, '使用哪个测试集：1，A赛段；2，B赛段')

# flags.DEFINE_float('drop_out', 0.1, 'drop out比率')
flags.DEFINE_integer('batch', 2048, '设置批大小')
flags.DEFINE_integer('epoch', 1, '设置训练的轮数')
flags.DEFINE_integer('checkpoint', 200, '每隔多少个批次保存一次模型')
flags.DEFINE_integer('summary', 50, '每隔多少个批次记录一次日志')
flags.DEFINE_boolean('train', False, '选择是训练还是推理')
flags.DEFINE_boolean('valid', False, '是否在训练中做交叉验证')  # 在train为False的前提下，valid如果为False，则生成submission，valid为True，在验证集上评估auc
flags.DEFINE_string('logdir', 'logdir_hash_3', '日志保存路径')
flags.DEFINE_string('auc_path', 'results/auc.csv', '损失和auc数据')
flags.DEFINE_list('hidden', [128, 64, 32], '设置隐藏层结构')
flags.DEFINE_integer('embed', 128, '各嵌入层的单元数')

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
