#coding:utf-8

import cv2
import numpy as np
import tensorflow as tf
import tensorflow.python.platform
import os

import dataset

# settings
import settings
FLAGS = settings.FLAGS

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool(x):
	return tf.nn.max_pool(x, ksize=[1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

x = tf.placeholder(tf.float32, [None, FLAGS.input_size*FLAGS.input_size*FLAGS.color_dim])
x_image = tf.reshape(x, [-1, FLAGS.input_size, FLAGS.input_size, FLAGS.color_dim])

#first conv
with tf.name_scope('conv1') as scope:
	W_conv1 = tf.Variable(tf.truncated_normal([FLAGS.filter_size, FLAGS.filter_size, FLAGS.color_dim, FLAGS.num_filters1], stddev = 0.1))
	b_conv1 = tf.Variable(tf.constant(0.1, shape = [FLAGS.num_filters1]))
	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

#first pool
with tf.name_scope('pool1') as scope:
	h_pool1 = max_pool(h_conv1)

#second conv
with tf.name_scope('conv2') as scope:
	W_conv2 = tf.Variable(tf.truncated_normal([FLAGS.filter_size, FLAGS.filter_size, FLAGS.num_filters1, FLAGS.num_filters2], stddev = 0.1))
	b_conv2 = tf.Variable(tf.constant(0.1, shape = [FLAGS.num_filters2]))
	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) +b_conv2)

#second pool
with tf.name_scope('pool2') as scope:
	h_pool2 = max_pool(h_conv2)

#full-connected1
with tf.name_scope('fc1'):
	pool_out = FLAGS.input_size / 4
	num_units1 = pool_out*pool_out*FLAGS.num_filters2
	num_units2 = 1024
	h_pool2_flat = tf.reshape(h_pool2, [-1, num_units1])
	w_fc1 = tf.Variable(tf.truncated_normal([num_units1, num_units2]))
	b_fc1 = tf.Variable(tf.constant(0.1, shape = [num_units2]))
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)
	keep_prob = tf.placeholder(tf.float32)

	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
	# print np.array(h_fc1_drop).shape

#full-connected2
with tf.name_scope('fc0'):
	w0 = tf.Variable(tf.zeros([num_units2, FLAGS.classes_num]))
	b0 = tf.Variable(tf.zeros([FLAGS.classes_num]))

#calculate p
with tf.name_scope('softmax') as scope:
	p = tf.nn.softmax(tf.matmul(h_fc1_drop, w0) + b0)

#loss(t : labels)
t = tf.placeholder(tf.float32, [None, 2])
cross_entropy = -tf.reduce_sum(t * tf.log(p))
tf.scalar_summary("cross_entropy", cross_entropy)

#train_step 
train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(p, 1), tf.argmax(t, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.scalar_summary("accuracy", accuracy)

#from dataset
train_image, train_label = dataset.train_load_data()
test_image, test_label = dataset.test_load_data()
	
#session start
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())
saver = tf.train.Saver()
#for Tensorboard
summary_op = tf.merge_all_summaries()
summary_writer = tf.train.SummaryWriter('mycnn_tensorboard', graph_def = sess.graph_def)

for step in range(FLAGS.epoch):
	for i in range(len(train_image)/FLAGS.batch_size):
		batch = FLAGS.batch_size * i
		train_image_batch = []
		train_label_batch = []
		sess.run(train_step, feed_dict={
			x: train_image[batch : batch+FLAGS.batch_size], 
			t: train_label[batch : batch+FLAGS.batch_size], 
			keep_prob: 0.5})

	train_accuracy = accuracy.eval(feed_dict={x:train_image, t: train_label, keep_prob: 1.0})
	print("step %d, training accuracy %g"%(step, train_accuracy))
	#for tensorboard
	summary_str = sess.run(summary_op, feed_dict={
   	            x: train_image,
   	            t: train_label,
   	            keep_prob: 1.0})
	summary_writer.add_summary(summary_str, step)

print "test accuracy %g"%sess.run(accuracy, feed_dict={
	x: test_image,
	t: test_label,
	keep_prob: 1.0})

save_path = saver.save(sess, "model.ckpt")		