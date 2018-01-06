#!/usr/bin/env python
import os
from tqdm import trange
import numpy as np
import tensorflow as tf
from ops import *
import gensim


class Model(object):
	def __init__(self):
		self.learning_rate = 0.0001
		self.batch_size = 20

		self.start = 1
		self.end = 10001
		self.checkpoint = 300
		self.testpoint = 5
		self.time_steps = 45
		self.vector_size = 200

		# print("LOADING EMBEDDING...", end=" ")
		# self.embedding = gensim.models.KeyedVectors.load_word2vec_format('data/glove_word2vec.txt')
		# self.embedding = self.embedding.wv
		# print("DONE")

		self.__build()

	def __discriminator(self, x):
		x = tf.expand_dims(x, 3)
		
		with tf.variable_scope('conv1-3'):
			L1_3 = lrelu(conv2d(x, co=300, k=[2,self.vector_size], s=[1,1,1,1], padding='VALID', bn=False))
			L1_3 = tf.reduce_max(L1_3, 1)

		with tf.variable_scope('conv1-4'):
			L1_4 = lrelu(conv2d(x, co=300, k=[3,self.vector_size], s=[1,1,1,1], padding='VALID', bn=False))
			L1_4 = tf.reduce_max(L1_4, 1)

		with tf.variable_scope('conv1-5'):
			L1_5 = lrelu(conv2d(x, co=300, k=[4,self.vector_size], s=[1,1,1,1], padding='VALID', bn=False))
			L1_5 = tf.reduce_max(L1_5, 1)

		L1 = tf.concat([L1_3, L1_4, L1_5],1)
		L1 = tf.reshape(L1, [-1, 900])

		with tf.variable_scope('FC1'):
			L2 = lrelu(fc(L1, 200, bn=False))
		with tf.variable_scope('FC2'):
			L3 = fc(L2, 1, bn=False)
		return L3


	def __generator(self, z):

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

		L1 = tf.contrib.rnn.GRUCell(2048)
		L2 = tf.contrib.rnn.LayerNormBasicLSTMCell(1024)
		L3 = tf.contrib.rnn.LayerNormBasicLSTMCell(512)
		L4 = tf.contrib.rnn.GRUCell(self.vector_size)

		# L1 = tf.nn.rnn_cell.GRUCell(512)
		# L2 = tf.nn.rnn_cell.GRUCell(1024)
		# L3 = tf.nn.rnn_cell.GRUCell(self.vector_size)

		multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell([L1, L2, L3, L4])
		initial_state = multi_rnn_cell.zero_state(z_shape[0], tf.float32)

		rnn_out, state = tf.nn.dynamic_rnn(multi_rnn_cell, z, initial_state=initial_state, time_major=False)
		return rnn_out


	def __build(self):
		self.x = tf.placeholder(tf.float32, shape=[None, None, self.vector_size])
		self.z = tf.placeholder(tf.float32, shape=[None, None, 100])

		self.y_ = tf.placeholder(tf.string)


		with tf.variable_scope("generator") as G:
			self.g_out = self.__generator(self.z)

		with tf.variable_scope("discriminator") as D:
			#add gaussian noise
			# noisy_x = self.x + tf.random_normal(tf.shape(self.x), 0.0, 0.005)

			D_real = self.__discriminator(self.x)
			D.reuse_variables()
			D_false = self.__discriminator(self.g_out)

			#gradient penalty for wasserstein gp
			epsilon = tf.random_uniform([tf.shape(self.x)[0],1,1], 0.0, 1.0)
			x_hat = self.x + epsilon*(self.g_out-self.x)

			D_false_w = self.__discriminator(x_hat)
			gradients = tf.gradients(D_false_w, [x_hat])[0]
			slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=1))
			self.gradient_penalty = 10*tf.reduce_mean(tf.square(slopes-1.0))


		# cost functions
		self.G_loss = -tf.reduce_mean(D_false)
		self.D_loss = tf.reduce_mean(D_false) - tf.reduce_mean(D_real) + self.gradient_penalty

		G_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"generator")
		D_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"discriminator")

		with tf.name_scope("optimizer"):
			self.D_solver = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5, beta2=0.9).minimize(self.D_loss, var_list = D_var)
			self.G_solver = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5, beta2=0.9).minimize(self.G_loss, var_list = G_var)

		#summaries
		Distribution_True = tf.summary.histogram("distribution/true", D_real)
		Distribution_False = tf.summary.histogram("distribution/false", D_false)
		Distribution_Total = tf.summary.histogram("distribution/both", tf.concat([D_real, D_false], 0))
		self.Distribution_summary = tf.summary.merge([Distribution_True, Distribution_False, Distribution_Total])

		G_loss_summ = tf.summary.scalar("G_loss", self.G_loss)
		D_loss_summ = tf.summary.scalar("D_loss", self.D_loss)
		Grad_penalty_summ = tf.summary.scalar("gradient_penalty", self.gradient_penalty)

		self.Cost_summary = tf.summary.merge([G_loss_summ, D_loss_summ, Grad_penalty_summ])

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
		data_reader = Data_reader_(data="data/trump_embedding_dynamic.npy", batch_size=20)
		test_z = np.random.normal(0, 1, size=[5, 6, 100])
		for i in trange(self.start, self.end):
			if i%self.checkpoint==0:
				self.__checkpoint(i)

			D_cost_total = 0
			G_cost_total = 0

			#train discriminator first for n_critic(5) times
			for ix in trange(0, 5):
				batch_x = data_reader.next_batch()
				batch_z = np.random.normal(0, 1, size=[batch_x.shape[0], batch_x.shape[1], 100])

				_ = self.session.run(self.D_solver, feed_dict={self.x: batch_x, self.z: batch_z})

			batch_x = data_reader.next_batch()
			batch_z = np.random.normal(0, 1, size=[batch_x.shape[0], batch_x.shape[1], 100])

			_, Loss_summary, Dist_summary= self.session.run([self.G_solver, self.Cost_summary, self.Distribution_summary], feed_dict={self.x: batch_x, self.z: batch_z})
			
			self.writer.add_summary(Loss_summary, i)
			self.writer.add_summary(Dist_summary, i)


			# if i%self.testpoint==0:
			# 	phrases = []
			# 	output = self.session.run(self.g_out, feed_dict={self.z: test_z})
			# 	for words in output:
			# 		sentence = []
			# 		for word in words:
			# 			top_words = self.embedding.most_similar([word], topn=1)
			# 			top_word = top_words[0][0]
			# 			sentence.append(top_word)
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



