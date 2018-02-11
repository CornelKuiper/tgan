import tensorflow as tf
import numpy as np


class Data_reader(object):
	def __init__(self, data, labels=None):
		self.data = np.load(data)
		self.use_label=False
		if labels != None:
			self.use_label=True
			self.labels = np.load(labels)
			assert self.labels.shape[0]==self.data.shape[0]

	def next_batch(self, batch_size):
		if self.use_label:
			batch = [self.data[:batch_size], self.labels[:batch_size]]
			self.data = np.roll(self.data, batch_size, axis=0)
			self.labels = np.roll(self.labels, batch_size, axis=0)
			return batch
		else:
			batch = self.data[:batch_size]
			self.data = np.roll(self.data, batch_size, axis=0)
			return batch

class Data_reader_(object):
	def __init__(self, data, batch_size, padding=False, lmin=4,lmax=36, min_batch_size=1):
		self.data = np.load(data)
		self.batch_size = batch_size
		self.padding=padding
		self.min_batch_size = min_batch_size
		if self.padding:
			data_size = sum([i.shape for i in data[lmin:lmax+1]])
			data_ = np.zeros([data_size, lmax, 200])
			idx = 0
			for i, data_i in enumerate(self.data[lmin:lmax+1], start=lmin):
				size = data_i.shape[0]
				data_[idx:idx+size,:i] = data_i
				idx += size
			self.data = data_
			del data_zero
		else:
			data_ = []
			for i, bucket in enumerate(self.data[lmin:lmax+1], start=lmin):
				for j in range(0, bucket.shape[0],self.batch_size):
					bucket_j = bucket[j:j+self.batch_size]
					data_.append(bucket_j)
					if bucket_j.shape[0] != self.batch_size:
						print("final bucket_{} batch of size:\t{}".format(i, bucket_j.shape[0]))
			self.data = np.asarray(data_)
			del data_

		self.data = np.random.permutation(self.data)
		self.data_size = self.data.shape[0]
		if padding:
			print("NR OF BATCHES: {}".format(self.data_size/self.batch_size))
		else:
			print("NR OF BATCHES: {}".format(self.data_size))

		self.padding = padding
		self.count = 0

	def next_batch(self):
		if self.padding:
			if self.count >= self.data_size:
				self.count = 0
				self.data = np.random.permutation(self.data)
			batch = self.data[:self.batch_size]
			self.data = np.roll(self.data, self.batch_size, axis=0)
			self.count += self.batch_size
		else:
			if self.count >= self.data_size:
				self.count = 0
				self.data = np.random.permutation(self.data)
			batch = self.data[self.count]
			self.count += 1
		while batch.shape[1] < self.min_batch_size:
			batch = self.next_batch()

		return batch

def weight_variable(shape):
	weight = tf.get_variable("weight", shape, initializer=tf.truncated_normal_initializer(stddev=0.01))
	return weight

def bias_variable(shape):
	bias = tf.get_variable("bias", shape, initializer=tf.truncated_normal_initializer(stddev=0.01))
	return bias

def max_pool_2x2(x, k=2, s=2, padding='SAME'):
	return tf.nn.max_pool(x, [1, k, k, 1], [1, s, s, 1], padding)

def conv2d(x, k, co, s=1, bn = True, padding = 'SAME'):
	if  isinstance(k, list):
		W = weight_variable([k[0], k[1], shape(x)[-1], co])
	else:
		W = weight_variable([k, k, shape(x)[-1], co])
	B = bias_variable([co])
	if not isinstance(s, list):
		s = [1,s,s,1]
	if bn:
		L = batch_norm(tf.nn.conv2d(x, W, s, padding) + B)
	else:
		L = tf.nn.conv2d(x, W, s, padding) + B
	return L

def conv2d_dilated(x, k, co, rate=1, bn = True, padding = 'SAME'):
	if  isinstance(k, list):
		W = weight_variable([k[0], k[1], shape(x)[-1], co])
	else:
		W = weight_variable([k, k, shape(x)[-1], co])
	B = bias_variable([co])
	L = tf.nn.atrous_conv2d(x, W, rate=rate, padding=padding) + B
	if bn:
		L = batch_norm(L)
	return L

def fc(x, co, bn = True):
	W = weight_variable([shape(x)[-1], co])
	B = bias_variable([co])
		
	if bn:
		L = batch_norm(tf.matmul(x, W) + B)
	else:
		L = tf.matmul(x, W) + B
	return L

def deconv2d(x, W, shape, s=1, padding = 'SAME'):
	deconv = tf.nn.conv2d_transpose(value=x, filter=W, output_shape=shape, strides=[1,s,s,1], padding='SAME')
	return tf.reshape(deconv, shape)

def lrelu(x, alpha=0.2):
	L = tf.maximum(x, alpha*x, "lrelu")
	return L

def batch_norm(x, epsilon=1e-5):
	return tf.layers.batch_normalization(inputs=x, epsilon=epsilon)

def shape(tensor):
	return tensor.get_shape().as_list()

def concat_y(x, y):
	with tf.name_scope("concat_y"):
		yb = tf.tile(tf.reshape(y, [-1, 1, 1, shape(y)[-1]]),[1, tf.shape(x)[1], tf.shape(x)[2], 1])
		xy = tf.concat([x, yb], 3)
	return xy	