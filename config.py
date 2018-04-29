import tensorflow as tf

flags = tf.app.flags


############################
#    hyper parameters      #
############################

flags.DEFINE_float('lr', 0.01, '设置学习率')
flags.DEFINE_float('pos_weight', 1, '惩罚FN')
flags.DEFINE_float('drop_out', 0.5, 'drop out比率')
flags.DEFINE_integer('batch', 1024, '设置批大小')
flags.DEFINE_integer('epoch', 5, '设置训练的轮数')
flags.DEFINE_integer('checkpoint', 200, '每隔多少个批次保存一次模型')
flags.DEFINE_integer('summary', 50, '每隔多少个批次记录一次日志')
flags.DEFINE_boolean('train', True, '选择是训练还是推理')
flags.DEFINE_boolean('valid', True, '是否在训练中做交叉验证')
flags.DEFINE_string('logdir', 'logdir', '日志保存路径')
flags.DEFINE_string('auc_path', 'results/auc.csv', '损失和auc数据')
flags.DEFINE_list('hidden', [128, 64, 64, 64, 32, 32, 32, 16, 16, 8, 8], '设置隐藏层结构')
flags.DEFINE_list('embed', [24, 24, 24, 24], '各嵌入层的单元数')

flags.DEFINE_float('gammar', 0.3, 'AUC参数')
flags.DEFINE_integer('power_p', 2, 'AUC参数')


cfg = tf.app.flags.FLAGS

# Uncomment this line to run in debug mode
# tf.logging.set_verbosity(tf.logging.INFO)
