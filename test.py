#coding:utf-8
#pickup gray-scale image

import scipy.misc
import numpy as np
import os

count = 0
gray = (64, 64)
for i in range(100):
	path = os.path.join("./data/dog", "dog_%d.jpg") % (count)
	if gray == scipy.misc.imread(path).astype(np.float).shape:
		print "dog_{0}".format(count)
	#print "tigercat_{0} : {1}".format(count, str(scipy.misc.imread(path).astype(np.float).shape))
	count += 1