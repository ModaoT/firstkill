import tensorflow as tf

flags = tf.app.flags


############################
#    hyper parameters      #
############################

flags.DEFINE_float('lr', 0.01, '设置学习率')
flags.DEFINE_float('pos_weight', 1, '惩罚FN')
flags.DEFINE_float('drop_out', 0.5, 'drop out比率')
flags.DEFINE_integer('batch', 256, '设置批大小')
flags.DEFINE_integer('epoch', 1, '设置训练的轮数')
flags.DEFINE_integer('checkpoint', 200, '每隔多少个批次保存一次模型')
flags.DEFINE_integer('summary', 50, '每隔多少个批次记录一次日志')
flags.DEFINE_boolean('train', True, '选择是训练还是推理')
flags.DEFINE_boolean('valid', True, '是否在训练中做交叉验证')
flags.DEFINE_string('logdir', 'logdir', '日志保存路径')
flags.DEFINE_string('auc_path', 'results/auc.csv', '损失和auc数据')
flags.DEFINE_list('hidden', [256, 256, 128], '设置隐藏层结构')
flags.DEFINE_integer('embed', 500, '各嵌入层的单元数')


cfg = tf.app.flags.FLAGS

# Uncomment this line to run in debug mode
# tf.logging.set_verbosity(tf.logging.INFO)
