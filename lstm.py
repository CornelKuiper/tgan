#!/usr/bin/env python
import os
from tqdm import trange
import numpy as np
import tensorflow as tf
from ops import *


# from tensorflow.examples.tutorials.mnist import input_data
class Model(object):
	def __init__(self):
		self.learning_rate = 0.0005
		self.batch_size = 20

		self.start = 0
		self.end = 1000
		self.checkpoint = 200
		self.testpoint = 50
		self.time_steps = 20
		# self.mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

		self.__build()

	def __discriminator(self, x):
		x = tf.expand_dims(x, 3)
		
		with tf.variable_scope('conv1-3'):
			L1_3 = tf.nn.relu(conv2d(x, co=300, k=[3,300], s=[1,1,1,1], padding='VALID', bn=False))
			L1_3_time = L1_3.get_shape()[1]
			L1_3 = tf.nn.max_pool(L1_3, [1,-1,1,1], [1,1,1,1], 'VALID')
			print(L1_3)

		with tf.variable_scope('conv1-4'):
			L1_4 = tf.nn.relu(conv2d(x, co=300, k=[4,300], s=[1,1,1,1], padding='VALID', bn=False))
			L1_4_time = L1_4.get_shape()[1]
			L1_4 = tf.nn.max_pool(L1_4, [1,L1_4_time,1,1], [1,1,1,1], 'VALID')

		with tf.variable_scope('conv1-5'):
			L1_5 = tf.nn.relu(conv2d(x, co=300, k=[5,300], s=[1,1,1,1], padding='VALID', bn=False))
			L1_5_time = L1_5.get_shape()[1]
			L1_5 = tf.nn.max_pool(L1_5, [1,L1_5_time,1,1], [1,1,1,1], 'VALID')

		L1 = tf.concat([L1_3, L1_4, L1_5],3)
		L1 = tf.reshape(L1, [-1, 900])


		with tf.variable_scope('FC1'):
			L2 = fc(L1, 200, bn=False)
		with tf.variable_scope('FC2'):
			L3 = fc(L2, 1, bn=False)
		return L3, tf.nn.sigmoid(L3)

	def __generator(self, z):
		L1 = tf.nn.rnn_cell.LSTMCell(300, initializer=tf.truncated_normal_initializer(stddev=0.01))
		multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell([L1])
		initial_state = multi_rnn_cell.zero_state(tf.shape(z)[0], tf.float32)

		#rnn_out = [batch_size, max_time, data]
		rnn_out, state = tf.nn.dynamic_rnn(multi_rnn_cell, z, initial_state=initial_state, time_major=False)
		print(rnn_out)
		return rnn_out

	def __build(self):
		self.x = tf.placeholder(tf.float32, shape=[None, self.time_steps, 300])
		self.y_ = tf.placeholder(tf.float32, shape=[None, self.time_steps, 300])
		self.z = tf.placeholder(tf.float32, shape=[None, self.time_steps, 100])

		with tf.variable_scope("generator") as G:
			g_out = self.__generator(self.z)

		with tf.variable_scope("discriminator") as D:
			D_real, D_real_sig = self.__discriminator(self.y_)
			D.reuse_variables()
			D_false, D_false_sig = self.__discriminator(g_out)

		with tf.name_scope("loss"):
			D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real, labels=(tf.ones_like(D_real)-0.1)))
			D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_false, labels=tf.zeros_like(D_false)))
			self.D_loss = D_loss_real + D_loss_fake

			self.G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_false, labels=tf.ones_like(D_false)))

		G_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"generator")
		D_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"discriminator")

		with tf.name_scope("optimizer"):
			self.D_solver = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5).minimize(self.D_loss, var_list = D_var)
			self.G_solver = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5).minimize(self.G_loss, var_list = G_var)
		
		#summaries
		Distribution_True = tf.summary.histogram("distribution/true", D_real_sig)
		Distribution_False = tf.summary.histogram("distribution/false", D_false_sig)
		Distribution_Total = tf.summary.histogram("distribution/false", tf.concat([D_real, D_false], 0))
		self.Distribution_summary = tf.summary.merge([Distribution_True, Distribution_False, Distribution_Total])


		self.session = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=6))
		self.saver = tf.train.Saver(max_to_keep=10)
		print("MODEL BUILD")

	def init(self, path='log'):
		self.session.run(tf.global_variables_initializer())
		if not os.path.exists(path):
			os.makedirs(path)
		self.directory = "{}/{}".format(path, len(os.listdir(path)))
		self.writer = tf.summary.FileWriter(self.directory, self.session.graph)
		print("DIRECTORY:\t{}".format(self.directory))
		print("MODEL VARIABLES INITIALIZED")

	def restore(self, path, iteration):
		self.directory = path
		self.writer = tf.summary.FileWriter(self.directory, self.session.graph)
		self.saver.restore(self.session, "{}/model/{}/model.ckpt".format(self.directory, iteration))
		print("MODEL VARIABLES RESTORED")


	# def run(self, input_x, input_y):
	# 	out_recog, out_gender = self.session.run([self.recognition, self.gender], feed_dict={self.x: input_x, self.y: input_y})
	# 	return out_recog, out_gender

	def __checkpoint(self, iteration):
		print("SAVING MODEL")
		chkpt_name = '{}/model/{}'.format(self.directory, iteration)
		if not os.path.exists(chkpt_name):
			os.makedirs(chkpt_name)
		self.saver.save(self.session, '{}/model.ckpt'.format(chkpt_name))

	def train(self):

		for i in trange(self.start, self.end):
			if i%self.checkpoint==0:
				self.__checkpoint(i)

			if i == 0:
				self.writer.add_summary(self.session.run(self.x_images, feed_dict={self.x: batch_x}),0)
			D_cost_total = 0
			G_cost_total = 0
			for ix in trange(0, batches):
				batch_x = batch_x.reshape([-1,4,300])
				batch_y = np.repeat(batch_y, 4, axis=0).reshape([-1,4,10])
				batch_z_length = np.randint(5, 45)
				batch_z = np.random.uniform(-1., 1., size=[batch_y.shape[0], batch_z_length, 100])
				_, D_cost, _2, G_cost, Dist_summary= self.session.run([self.D_solver, self.D_loss, self.G_solver, self.G_loss, self.Distribution_summary], feed_dict={self.x: batch_x, self.y_: batch_y, self.z: batch_z})
					
				self.writer.add_summary(Dist_summary, i)
				D_cost_total+=D_cost
				G_cost_total+=G_cost

			D_cost_total = tf.Summary(value=[tf.Summary.Value(tag="Discriminator", simple_value=D_cost_total)])
			self.writer.add_summary(D_cost_total, i)

			G_cost_total = tf.Summary(value=[tf.Summary.Value(tag="Generator", simple_value=G_cost_total)])
			self.writer.add_summary(G_cost_total, i)

			if i%self.testpoint==0:
				z_set = np.random.uniform(-1., 1., size=[10, 4, 100])
				y_test = np.zeros((10, 10))
				np.fill_diagonal(y_test, 1)
				y_test = np.repeat(y_test, 4, axis=0).reshape([-1,4,10])
				summary = self.session.run(self.G_images, feed_dict={self.y_: y_test, self.z: z_set})
				self.writer.add_summary(summary, i)

if __name__ == '__main__' :
	model = Model()
	# model.init()
	# model.train()



