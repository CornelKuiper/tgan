#!/usr/bin/env python
import os
from tqdm import trange
import numpy as np
import tensorflow as tf
from ops import *
import gensim


class Model(object):
	def __init__(self):
		self.learning_rate = 0.00005
		self.batch_size = 20

		self.start = 0
		self.end = 1200
		self.checkpoint = 300
		self.testpoint = 150
		self.time_steps = 45
		self.vector_size = 200

		# self.embedding = gensim.models.KeyedVectors.load_word2vec_format('data/glove_word2vec.txt')
		# self.embedding = self.embedding.wv

		self.__build()
	def __discriminator(self, x):
		x = tf.expand_dims(x, 3)
		
		with tf.variable_scope('conv1-3'):
			L1_3 = tf.nn.relu(conv2d(x, co=300, k=[3,self.vector_size], s=[1,1,1,1], padding='VALID', bn=False))
			L1_3 = tf.reduce_max(L1_3, 1)

		with tf.variable_scope('conv1-4'):
			L1_4 = tf.nn.relu(conv2d(x, co=300, k=[4,self.vector_size], s=[1,1,1,1], padding='VALID', bn=False))
			L1_4 = tf.reduce_max(L1_4, 1)

		with tf.variable_scope('conv1-5'):
			L1_5 = tf.nn.relu(conv2d(x, co=300, k=[5,self.vector_size], s=[1,1,1,1], padding='VALID', bn=False))
			L1_5 = tf.reduce_max(L1_5, 1)

		L1 = tf.concat([L1_3, L1_4, L1_5],1)
		L1 = tf.reshape(L1, [-1, 900])


		with tf.variable_scope('FC1'):
			L2 = fc(L1, 200, bn=False)
		with tf.variable_scope('FC2'):
			L3 = fc(L2, 1, bn=False)
		return L3, tf.nn.sigmoid(L3)

	def __generator(self, z, t):

		z_shape = tf.shape(z)
		#adding extra dim which might help generating correct ending of sequence
		with tf.name_scope("time_step"):
			step_size = tf.minimum(z_shape[1], 5)
			steps = tf.range(tf.cast(step_size, dtype=tf.float32)-1, -1, -1, dtype=tf.float32)*0.45
			steps = tf.tanh(steps)
			step_ = tf.ones([z_shape[1]-step_size])
			steps = tf.concat([step_, steps],0)
			steps = tf.reshape(steps, [1, -1, 1])

			steps = tf.tile(steps, [z_shape[0], 1, 1])
			z = tf.concat([z, steps], 2)

		L1 = tf.nn.rnn_cell.GRUCell(512)
		L2 = tf.nn.rnn_cell.GRUCell(1024)
		L3 = tf.nn.rnn_cell.GRUCell(self.vector_size)
		multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell([L1, L2, L3])
		initial_state = multi_rnn_cell.zero_state(z_shape[0], tf.float32)

		rnn_out, state = tf.nn.dynamic_rnn(multi_rnn_cell, z, initial_state=initial_state, time_major=False)
		return rnn_out


	def mmd(self, x, y, sigma = 1):
		sigma = 1
		x = tf.layers.flatten(x)
		y = tf.layers.flatten(y)

		xx = tf.matmul(x,x,transpose_b=True)
		yy = tf.matmul(y,y,transpose_b=True)
		xy = tf.matmul(x,y,transpose_b=True)
		#[batch_size, batch_size]

		rx = tf.expand_dims(tf.diag_part(xx),0)
		ry = tf.expand_dims(tf.diag_part(yy),0)

		K = tf.exp(- sigma * (tf.transpose(rx) + rx - 2*xx))
		L = tf.exp(- sigma * (tf.transpose(ry) + ry - 2*yy))
		P = tf.exp(- sigma * (tf.transpose(rx) + ry - 2*xy))
		batch_size = tf.to_float(tf.shape(x)[0])
		beta = tf.div(1.,(batch_size*(batch_size-1)))
		gamma = tf.div(2.,(batch_size*batch_size))

		mmdistance = beta * (tf.reduce_sum(K)+tf.reduce_sum(L)) - gamma * tf.reduce_sum(P)
		return mmdistance

	def __build(self):
		self.x = tf.placeholder(tf.float32, shape=[None, self.time_steps, self.vector_size])
		self.z = tf.placeholder(tf.float32, shape=[None, self.time_steps, 100])

		self.y_ = tf.placeholder(tf.string)


		with tf.variable_scope("generator") as G:
			self.g_out = self.__generator(self.z)
		self.mmd_loss = self.mmd(self.g_out, self.x)

		with tf.variable_scope("discriminator") as D:
			D_real, D_real_sig = self.__discriminator(self.x)
			D.reuse_variables()
			D_false, D_false_sig = self.__discriminator(self.g_out)

		with tf.name_scope("loss"):
			D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real, labels=tf.random_normal(tf.shape(D_real), mean=1, stddev=0.1)))
			D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_false, labels=tf.random_normal(tf.shape(D_false), mean=0.1, stddev=0.1)))
			self.D_loss = (D_loss_real + D_loss_fake)/2

			self.G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_false, labels=tf.ones_like(D_false)))

		G_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"generator")
		D_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"discriminator")

		with tf.name_scope("optimizer"):
			self.D_solver = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5).minimize(self.D_loss, var_list = D_var)
			self.G_solver = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5).minimize(-self.D_loss, var_list = G_var)
		
		#summaries
		Distribution_True = tf.summary.histogram("distribution/true", D_real_sig)
		Distribution_False = tf.summary.histogram("distribution/false", D_false_sig)
		Distribution_Total = tf.summary.histogram("distribution/both", tf.concat([D_real, D_false], 0))
		self.Distribution_summary = tf.summary.merge([Distribution_True, Distribution_False, Distribution_Total])

		self.text_summary = tf.summary.text("sentences", self.y_)


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


	def run(self, input_z):
		out = self.session.run(self.g_out, feed_dict={self.z: input_z})
		return out

	def __checkpoint(self, iteration):
		print("SAVING MODEL")
		chkpt_name = '{}/model/{}'.format(self.directory, iteration)
		if not os.path.exists(chkpt_name):
			os.makedirs(chkpt_name)
		self.saver.save(self.session, '{}/model.ckpt'.format(chkpt_name))

	def train(self):
		data_reader = Data_reader(data="data/glove_trump_norm.npy")
		D_cost = 0
		G_cost = 0
		test_z = np.random.normal(0, 1, size=[3, self.time_steps, 100])
		for i in trange(self.start, self.end):
			if i%self.checkpoint==0:
				self.__checkpoint(i)
			D_cost_total = 0
			G_cost_total = 0
			for ix in trange(0, 10):
				batch_x = data_reader.next_batch(self.batch_size)
				batch_z = np.random.normal(0, 1, size=[batch_x.shape[0], self.time_steps, 100])

				_, __, D_cost, G_cost, Dist_summary= self.session.run([self.D_solver, self.G_solver, self.D_loss, self.G_loss, self.Distribution_summary], feed_dict={self.x: batch_x, self.z: batch_z})

				self.writer.add_summary(Dist_summary, i)
				D_cost_total+=D_cost
				G_cost_total+=G_cost

			D_cost_total = tf.Summary(value=[tf.Summary.Value(tag="Discriminator", simple_value=D_cost_total)])
			self.writer.add_summary(D_cost_total, i)

			G_cost_total = tf.Summary(value=[tf.Summary.Value(tag="Generator", simple_value=G_cost_total)])
			self.writer.add_summary(G_cost_total, i)

			# if i%self.testpoint==0:
			# 	phrases = []
			# 	output = self.session.run(self.g_out, feed_dict={self.z: test_z})
			# 	for words in output:
			# 		sentence = []
			# 		for word in words:
			# 			sentence.append(self.embedding.most_similar([word], topn=1)[0][0])
			# 		sentence = ' '.join(sentence)
			# 		phrases.append(sentence)
			# 	rand = np.random.randint(0,self.batch_size-2)
			# 	for words in batch_x[rand:rand+2]:
			# 		sentence = []
			# 		for word in words:
			# 			sentence.append(self.embedding.most_similar_cosmul([word], topn=1)[0][0])
			# 		sentence = ' '.join(sentence)
			# 		phrases.append(sentence)

			# 	text_summ = self.session.run(self.text_summary, feed_dict={self.y_: phrases})
			# 	self.writer.add_summary(text_summ, i)


if __name__ == '__main__' :
	model = Model()
	model.init()
	model.train()



