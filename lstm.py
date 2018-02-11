#!/usr/bin/env python
import os
from tqdm import trange
import numpy as np
import tensorflow as tf
from ops import *
import gensim
import threading

class Model(object):
	def __init__(self):
		self.learning_rate = 0.0001
		self.batch_size = 20

		self.start = 1
		self.iterations = 10000
		self.checkpoint = 500
		self.testpoint = 100
		self.vector_size = 200
		self.z_size = 100

		self.loaded_embedding = False

		def load_embedding():
			print("LOADING EMBEDDING...")
			self.embedding = gensim.models.KeyedVectors.load_word2vec_format('data/glove_word2vec.txt')
			self.embedding = self.embedding.wv
			print("LOADED EMBEDDING!")
			self.loaded_embedding = True

		t = threading.Thread(target=load_embedding)
		t.start()


		self.__build()

	"""
	def __discriminator(self, x):
		x_shape = tf.shape(x)
		x = tf.expand_dims(x, 2)


		#adding extra dim which might help generating correct ending of sequence
		with tf.name_scope("time_step"):
			step_size = tf.minimum(x_shape[1], 5)
			steps = tf.range(tf.cast(step_size, dtype=tf.float32)-1, -1, -1, dtype=tf.float32)*0.45
			steps = tf.tanh(steps)
			step_ = tf.ones([x_shape[1]-step_size])
			steps = tf.concat([step_, steps],0)
			steps = tf.reshape(steps, [1,-1,1,1])

			steps = tf.tile(steps, [x_shape[0],1,1,1])
			x = tf.concat([x, steps], 3)


		kernel_size = 2
		rates = [kernel_size**i for i in range(2)]
		padding_needed = sum(rates[:-1])

		with tf.variable_scope('wave-0'):
			l1 = tf.nn.leaky_relu(conv2d_dilated(x, k=[kernel_size,1], co=2048, rate=[rates[0],1], bn=False, padding='VALID'))
			l1_min_size = rates[-1]-(kernel_size-1)
			pad_size = tf.maximum(l1_min_size - tf.shape(l1)[1], 0)
			pad = tf.zeros([x_shape[0], pad_size, 1, 2048])
			l1 = tf.concat([l1, pad], 1)
			# l1 = tf.Print(l1, [tf.shape(l1)], message="l1= ", summarize=6)
		with tf.variable_scope('wave-1'):
			l2 = tf.nn.leaky_relu(conv2d_dilated(l1, k=[kernel_size,1], co=1024, rate=[rates[1],1], bn=False, padding='VALID'))
			# l2 = tf.Print(l2, [tf.shape(l2)], message="l2= ", summarize=6)

			l2 = tf.reduce_mean(l2, 1)



		l2 = tf.reshape(l2, [-1,1024])

		with tf.variable_scope('FC1'):
			l3 = tf.nn.leaky_relu(fc(l2, 512, bn=False))
		with tf.variable_scope('FC2'):
			l4 = fc(l3, 1, bn=False)
		return l4	

	"""
	def __discriminator(self, x):
		x = tf.expand_dims(x, 2)
		
		with tf.variable_scope('conv1-2'):
			L1_3 = tf.nn.leaky_relu(conv2d(x, co=300, k=[2,1], s=1, padding='VALID', bn=False))
			L1_3 = tf.reduce_mean(L1_3, 1)

		with tf.variable_scope('conv1-3'):
			L1_4 = tf.nn.leaky_relu(conv2d(x, co=300, k=[3,1], s=1, padding='VALID', bn=False))
			L1_4 = tf.reduce_mean(L1_4, 1)

		with tf.variable_scope('conv1-4'):
			L1_5 = tf.nn.leaky_relu(conv2d(x, co=300, k=[4,1], s=1, padding='VALID', bn=False))
			L1_5 = tf.reduce_mean(L1_5, 1)

		L1 = tf.concat([L1_3, L1_4, L1_5],1)
		L1 = tf.reshape(L1, [-1, 900])

		with tf.variable_scope('FC1'):
			L2 = tf.nn.leaky_relu(fc(L1, 512, bn=False))
		with tf.variable_scope('FC2'):
			L3 = fc(L2, 1, bn=False)
		return L3
	"""


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

		L1 = tf.contrib.rnn.LayerNormBasicLSTMCell(2048)
		L2 = tf.contrib.rnn.LayerNormBasicLSTMCell(1024)
		L3 = tf.contrib.rnn.LayerNormBasicLSTMCell(200)

		multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell([L1, L2, L3])
		initial_state = multi_rnn_cell.zero_state(z_shape[0], tf.float32)

		rnn_out, state = tf.nn.dynamic_rnn(multi_rnn_cell, z, initial_state=initial_state, time_major=False)
		rnn_out *= 2
		return rnn_out
	"""
	def __generator(self, z):
		
		kernel_size = 2
		rates = [kernel_size**i for i in range(3)]

		padding_needed = sum(rates)
		z = tf.expand_dims(z, 2)
		# z = tf.pad(z, [[0,0],[padding_needed,0],[0,0],[0,0]])

		pad_noise = tf.random_normal([tf.shape(z)[0],padding_needed,1,100], 0.0, 1.0)
		z = tf.concat([pad_noise, z], 1)
		z_shape = tf.shape(z)

		#adding extra dim which might help generating correct ending of sequence
		with tf.name_scope("time_step"):
			step_size = tf.minimum(z_shape[1], 5)
			steps = tf.range(tf.cast(step_size, dtype=tf.float32)-1, -1, -1, dtype=tf.float32)*0.45
			steps = tf.tanh(steps)
			step_ = tf.ones([z_shape[1]-step_size])
			steps = tf.concat([step_, steps],0)
			steps = tf.reshape(steps, [1,-1,1,1])

			steps = tf.tile(steps, [z_shape[0],1,1,1])
			z = tf.concat([z, steps], 3)

		with tf.variable_scope('wave-0'):
			l1 = tf.nn.relu(conv2d_dilated(z, k=[kernel_size,1], co=2048, rate=[rates[0],1], bn=False, padding='VALID'))
		with tf.variable_scope('wave-1'):
			l2 = tf.nn.relu(conv2d_dilated(l1, k=[kernel_size,1], co=1024, rate=[rates[1],1], bn=True, padding='VALID'))
		with tf.variable_scope('wave-2'):
			l3 = 2*tf.nn.tanh(conv2d_dilated(l2, k=[kernel_size,1], co=200, rate=[rates[2],1], bn=False, padding='VALID'))
			l3 = tf.squeeze(l3, 2)

		return l3

	def __build(self):
		self.x = tf.placeholder(tf.float32, shape=[None, None, self.vector_size])
		self.z = tf.placeholder(tf.float32, shape=[None, None, self.z_size])

		self.y_ = tf.placeholder(tf.string)


		with tf.variable_scope("generator") as G:
			self.g_out = self.__generator(self.z)

		with tf.variable_scope("discriminator") as D:
			#add gaussian noise
			# noisy_x = self.x + tf.random_uniform(tf.shape(self.x), -0.005, 0.005)

			D_real = self.__discriminator(self.x)
			D.reuse_variables()
			D_false = self.__discriminator(self.g_out)

		with tf.name_scope("loss"):
			D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real, labels=tf.random_normal(tf.shape(D_real), mean=1, stddev=0.1)))
			D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_false, labels=tf.random_normal(tf.shape(D_false), mean=0.1, stddev=0.1)))
			self.D_loss = (D_loss_real + D_loss_fake)/2

			self.G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_false, labels=tf.ones_like(D_false)))

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

		self.Cost_summary = tf.summary.merge([G_loss_summ, D_loss_summ])

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
		self.start = iteration+1
		self.writer = tf.summary.FileWriter(self.directory, self.session.graph)
		self.saver.restore(self.session, "{}/model/{}/model.ckpt".format(self.directory, iteration))
		print("MODEL VARIABLES RESTORED")


	def run(self, input_z):
		phrases = []

		output = self.session.run(self.g_out, feed_dict={self.z: input_z})
		for words in output:
			sentence = []
			for word in words:
				top_words = self.embedding.most_similar([word], topn=1)
				top_word = top_words[0][0]
				sentence.append(top_word)
			sentence = ' '.join(sentence)
			phrases.append(sentence)
		return phrases

	def __checkpoint(self, iteration):
		print("SAVING MODEL")
		chkpt_name = '{}/model/{}'.format(self.directory, iteration)
		if not os.path.exists(chkpt_name):
			os.makedirs(chkpt_name)
		self.saver.save(self.session, '{}/model.ckpt'.format(chkpt_name))

	def train(self):
		data_reader = Data_reader_(data="data/trump_embedding_dynamic.npy", batch_size=20)
		for i in trange(self.start, self.start+self.iterations):
			if i%self.checkpoint==0:
				self.__checkpoint(i)


			batch_x = data_reader.next_batch()
			batch_z = np.random.normal(0, 1, size=[batch_x.shape[0], batch_x.shape[1], self.z_size])

			_, __, Loss_summary, Dist_summary= self.session.run([self.D_solver, self.G_solver, self.Cost_summary, self.Distribution_summary], feed_dict={self.x: batch_x, self.z: batch_z})
			
			self.writer.add_summary(Loss_summary, i)
			self.writer.add_summary(Dist_summary, i)


			if i%self.testpoint==0 and self.loaded_embedding:
				phrases = []
				test_z_ = [np.random.normal(0, 1, size=[5, 2**idx, self.z_size]) for idx in range(2,5)]
				for test_z in test_z_:
					output = self.session.run(self.g_out, feed_dict={self.z: test_z})
					for words in output:
						sentence = []
						for word in words:
							top_words = self.embedding.most_similar([word], topn=1)
							top_word = top_words[0][0]
							sentence.append(top_word)
						sentence = ' '.join(sentence)
						phrases.append(sentence)

				text_summ = self.session.run(self.text_summary, feed_dict={self.y_: phrases})
				self.writer.add_summary(text_summ, i)


if __name__ == '__main__' :
	model = Model()
	# model.init()
	model.restore(path='log/10', iteration=2500)
	model.train()



