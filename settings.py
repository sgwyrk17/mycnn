# encoding: utf-8

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('classes_num', 2, 'size of images')

flags.DEFINE_integer('input_size', 28, 'size of images')
flags.DEFINE_integer('color_dim', 3, 'dimention of color channel')
flags.DEFINE_integer('filter_size', 5, 'size of filter')
flags.DEFINE_integer('num_filters1', 32, 'filters1')
flags.DEFINE_integer('num_filters2', 64, 'filters1')

flags.DEFINE_integer('learning_rate', 0.0001, 'learning_late')
flags.DEFINE_integer('batch_size', 10, 'size of mini-batch')
flags.DEFINE_integer('epoch', 100, 'epoch')