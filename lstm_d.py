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
		self.time_steps = 45
		self.vec_size = 300
		# self.mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

		self.__build()

	def __discriminator(self, x):
		x = tf.expand_dims(x, 3)
		
		with tf.variable_scope('conv1-3'):
			L1_3 = tf.nn.relu(conv2d(x, co=300, k=[3,self.vec_size], s=[1,1,1,1], padding='VALID', bn=False))
			L1_3 = tf.reduce_max(L1_3, 1)

		with tf.variable_scope('conv1-4'):
			L1_4 = tf.nn.relu(conv2d(x, co=300, k=[4,self.vec_size], s=[1,1,1,1], padding='VALID', bn=False))
			L1_4 = tf.reduce_max(L1_4, 1)

		with tf.variable_scope('conv1-5'):
			L1_5 = tf.nn.relu(conv2d(x, co=300, k=[5,self.vec_size], s=[1,1,1,1], padding='VALID', bn=False))
			L1_5 = tf.reduce_max(L1_5, 1)

		L1 = tf.concat([L1_3, L1_4, L1_5],1)
		L1 = tf.reshape(L1, [-1, 900])


		with tf.variable_scope('FC1'):
			L2 = fc(L1, 200, bn=False)
		with tf.variable_scope('FC2'):
			L3 = fc(L2, 1, bn=False)
		return L3, tf.nn.sigmoid(L3)


	def __build(self):
		self.x = tf.placeholder(tf.float32, shape=[None, self.time_steps, self.vec_size])
		self.y_ = tf.placeholder(tf.float32, shape=[None, 1])

		with tf.variable_scope("discriminator") as D:
			D_real, D_real_sig = self.__discriminator(self.x)

		with tf.name_scope("loss"):
			self.D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real, labels=self.y_))

		with tf.name_scope("accuracy"):
			self.D_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.greater(D_real_sig, 0.5), tf.float32), self.y_), tf.float32))
			self.D_accuracy_total = tf.reduce_sum(tf.cast(tf.equal(tf.cast(tf.greater(D_real_sig, 0.5), tf.float32), self.y_), tf.float32))

		with tf.name_scope("optimizer"):
			self.D_solver = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5).minimize(self.D_loss)

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
		embedding = np.load()
		labels = np.load()
		data_size = labels.shape[0]
		for i in trange(self.start, self.end):
			if i%self.checkpoint==0:
				self.__checkpoint(i)

			D_cost_total = 0
			D_acc_total = 0
			for ix in trange(0, data_size, self.batch_size):
				batch_x = embedding[ix:ix+batch_size]
				batch_y = labels[ix:ix+batch_size]
				_, D_cost, D_acc= self.session.run([self.D_solver, self.D_loss, self.D_accuracy_total], feed_dict={self.x: batch_x, self.y_: batch_y})
				D_acc_total+=D_acc
				D_cost_total+=D_cost
				
			D_acc_total/=data_size
			D_cost_total = tf.Summary(value=[tf.Summary.Value(tag="Discriminator/loss", simple_value=D_cost_total)])
			self.writer.add_summary(D_cost_total, i)
			D_cost_total = tf.Summary(value=[tf.Summary.Value(tag="Discriminator/accuracy", simple_value=D_acc_total)])
			self.writer.add_summary(D_cost_total, i)


if __name__ == '__main__' :
	model = Model()
	model.init()
	# model.train()



