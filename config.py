import tensorflow as tf

flags = tf.app.flags


############################
#    hyper parameters      #
############################

flags.DEFINE_float('lr', 0.1, '设置学习率')
flags.DEFINE_integer('batch', 128, '设置批大小')
flags.DEFINE_integer('epoch', 10, '设置训练的轮数')
flags.DEFINE_integer('checkpoint', 200, '每隔多少个批次保存一次模型')
flags.DEFINE_integer('summary', 100, '每隔多少个批次记录一次日志')
flags.DEFINE_boolean('train', True, '选择是训练还是推理')
flags.DEFINE_string('logdir', 'logdir', '日志保存路径')
flags.DEFINE_list('hidden', [128, 256, 56], '设置隐藏层结构')

flags.DEFINE_float('gammar', 0.3, 'AUC参数')
flags.DEFINE_integer('power_p', 2, 'AUC参数')

cfg = tf.app.flags.FLAGS

# Uncomment this line to run in debug mode
# tf.logging.set_verbosity(tf.logging.INFO)
