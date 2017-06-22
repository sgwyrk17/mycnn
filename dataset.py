#coding : utf-8

import os
from glob import glob
import cv2
import numpy as np

# settings
import settings
FLAGS = settings.FLAGS

#load dataset, and labeling
def train_load_data():
	# print os.path.abspath(__file__)

	train_img_dirs = ['cat', 'dog']
	train_image = []
	train_label = []
	for i, d in enumerate(train_img_dirs):
		files = os.listdir('./data/' + d)
		for file in files:
			path = './data/' + d + '/' + file
			img = cv2.imread(path)
			img = cv2.resize(img, (FLAGS.input_size, FLAGS.input_size))
			img = img.flatten().astype(np.float32)/255.0
			train_image.append(img)

			tmp = np.zeros(FLAGS.classes_num)
			tmp[i] = 1
			train_label.append(tmp)
	train_image = np.asarray(train_image)
	# print train_image.shape
	train_label = np.asarray(train_label)
	return (train_image, train_label)

def test_load_data():
	print os.path.abspath(__file__)

	test_img_dirs = ['test_cat', 'test_dog']
	test_image = []
	test_label = []
	for i, d in enumerate(test_img_dirs):
		files = os.listdir('./data/' + d)
		for file in files:
			path = './data/' + d + '/' + file
			img = cv2.imread(path)
			img = cv2.resize(img, (FLAGS.input_size, FLAGS.input_size))
			img = img.flatten().astype(np.float32)/255.0
			test_image.append(img)

			tmp = np.zeros(FLAGS.classes_num)
			tmp[i] = 1
			test_label.append(tmp)
	test_image = np.asarray(test_image)
	test_label = np.asarray(test_label)
	return (test_image, test_label)

# if __name__ == '__main__':
# 	load_data()

