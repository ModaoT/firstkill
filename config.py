import tensorflow as tf

flags = tf.app.flags


############################
#    hyper parameters      #
############################

flags.DEFINE_float('lr', 0.0001, '设置学习率')
flags.DEFINE_float('pos_weight', 1, '惩罚FN')
flags.DEFINE_integer('feature', 1, '选择数据类型：1，经lgb筛选过的特征；2，list特征映射为hash等长向量')
flags.DEFINE_integer('arch', 1, '网络架构：1，仅fm；2，仅dnn；3，deepFM')
flags.DEFINE_integer('fold', 1, 'k折训练的阶段：1，在234训练，在1验证；2，在134训练，在2验证；3，在124训练，在3验证；4在123训练，在4验证')
flags.DEFINE_integer('test_set', 1, '使用哪个测试集：1，A赛段；2，B赛段')

flags.DEFINE_float('drop_out', 0.1, 'drop out比率')
flags.DEFINE_integer('batch', 2048, '设置批大小')
flags.DEFINE_integer('epoch', 2, '设置训练的轮数')
flags.DEFINE_integer('checkpoint', 200, '每隔多少个批次保存一次模型')
flags.DEFINE_integer('summary', 50, '每隔多少个批次记录一次日志')
flags.DEFINE_boolean('train', True, '选择是训练还是推理')
flags.DEFINE_boolean('valid', True, '是否在训练中做交叉验证')
flags.DEFINE_string('logdir', 'logdir', '日志保存路径')
flags.DEFINE_string('auc_path', 'results/auc.csv', '损失和auc数据')
flags.DEFINE_list('hidden', [128, 64, 32], '设置隐藏层结构')
flags.DEFINE_integer('embed', 128, '各嵌入层的单元数')


cfg = tf.app.flags.FLAGS

# Uncomment this line to run in debug mode
# tf.logging.set_verbosity(tf.logging.INFO)
