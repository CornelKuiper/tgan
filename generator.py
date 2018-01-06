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
		self.iterations = 10000
		self.checkpoint = 300
		self.testpoint = 100
		self.time_steps = 45
		self.vector_size = 200

		# self.embedding = gensim.models.KeyedVectors.load_word2vec_format('data/glove_word2vec.txt')
		# self.embedding = self.embedding.wv

		self.__build()


	"""
	def __generator(self, x, z):
		# L1 = tf.nn.rnn_cell.LSTMCell(self.vector_size)
		# L1 = tf.contrib.rnn.LayerNormBasicLSTMCell(self.vector_size)
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
			input = tf.concat([x[:,:-1], z, steps], 2)


		L1 = tf.nn.rnn_cell.GRUCell(512)
		L2 = tf.nn.rnn_cell.GRUCell(1024)
		L3 = tf.nn.rnn_cell.GRUCell(self.vector_size)
		multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell([L1, L2, L3])
		initial_state = multi_rnn_cell.zero_state(tf.shape(input)[0], tf.float32)

		#rnn_out = [batch_size, max_time, data]
		rnn_out, state = tf.nn.dynamic_rnn(multi_rnn_cell, input, initial_state=initial_state, time_major=False)

		return rnn_out
	"""

	def __generator(self, x, z):
		
		z_shape = tf.shape(z)

		#adding extra dim which might help generating correct ending of sequence
		with tf.name_scope("time_step"):
			step_size = tf.minimum(z_shape[1], 5)
			steps = tf.range(-tf.cast(step_size, dtype=tf.float32)+1, 1, 1, dtype=tf.float32)*0.45
			steps = tf.tanh(steps)+1
			step_ = tf.zeros([z_shape[1]-step_size])
			steps = tf.concat([step_, steps],0)
			steps = tf.reshape(steps, [1, -1, 1])

			steps = tf.tile(steps, [z_shape[0], 1, 1])


		z = tf.concat([x[:,0:1], steps[:,0:1]], 2)
		kernel_size = 2
		rates = [kernel_size**i for i in range(3)]

		padding_needed = sum(rates)
		z = tf.expand_dims(z, 2)
		z = tf.pad(z, [[0,0],[padding_needed,0],[0,0],[0,0]])
		print(z)

		def cond(z, steps, i, c):
			return tf.less(i, tf.shape(x)[1])

		def body(z, steps, i, c):

			with tf.variable_scope('wave-0'):
				l1 = tf.nn.relu(conv2d_dilated(z, k=[kernel_size,1], co=1024, rate=[rates[0],1], bn=False, padding='VALID'))

			with tf.variable_scope('wave-1'):
				l2 = tf.nn.relu(conv2d_dilated(l1, k=[kernel_size,1], co=512, rate=[rates[1],1], bn=False, padding='VALID'))

			with tf.variable_scope('wave-2'):
				l3 = tf.nn.tanh(conv2d_dilated(l2, k=[kernel_size,1], co=self.vector_size, rate=[rates[2],1], bn=False, padding='VALID'))
			# l3 = tf.Print(l3, [tf.shape(l3), tf.shape(x)], message="DEBUG: = ", summarize=6)
			l3_z = tf.concat([l3, steps[:,i-1]],3)
			z = tf.concat([z[:,1:], l3_z], 1)

			c = c.write(i, tf.reshape(l3, [tf.shape(l3)[0],self.vector_size]))
			return z, steps, i+1, c

		c = tf.TensorArray(tf.float32, tf.shape(x)[1])
		step_shape = tf.shape(steps)
		steps = tf.reshape(steps, [step_shape[0],step_shape[1],1,1,1])

		c = c.write(0, x[:,0])
		z, steps, i, c = tf.while_loop(cond, body, [z, steps, 1, c])

		c = c.stack()
		c = tf.transpose(c, [1,0,2])


		return c


	def __build(self):
		self.x = tf.placeholder(tf.float32, shape=[None, None, self.vector_size], name='INPUT')
		self.z = tf.placeholder(tf.float32, shape=[None, None, 100], name='RANDOM')

		self.y_ = tf.placeholder(tf.string)


		with tf.variable_scope("generator") as G:
			self.g_out = self.__generator(self.x, self.z)

		with tf.name_scope("optimizer"):
			self.G_loss = tf.reduce_mean(tf.square(self.g_out - self.x))

			G_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"generator")
			self.G_solver = tf.train.AdamOptimizer(self.learning_rate,).minimize(self.G_loss, var_list = G_var)

		self.text_summary = tf.summary.text("sentences", self.y_)
		self.Loss_summary = tf.summary.scalar("gradient_penalty", self.G_loss)


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
		self.start = iteration
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
		data_reader = Data_reader_(data="data/trump_embedding_dynamic.npy", batch_size=25)
		D_cost = 0
		G_cost = 0
		for i in trange(self.start, self.start+self.iterations):
		# 	if i%self.checkpoint==0:
		# 		self.__checkpoint(i)

			batch_x = data_reader.next_batch()
			batch_z = np.random.normal(0, 1, size=[batch_x.shape[0], batch_x.shape[1]-1, 100])


			# print(self.session.run(self.test, feed_dict={self.x: batch_x, self.z: batch_z}).shape)

			_, summary= self.session.run([self.G_solver, self.Loss_summary], feed_dict={self.x: batch_x, self.z: batch_z})

			self.writer.add_summary(summary, i)

			# if i%self.testpoint==0:
			# 	phrases = []
			# 	output = self.session.run(self.g_out, feed_dict={self.x: batch_x[0:3], self.z: batch_z[0:3]})
			# 	for j, words in enumerate(output):
			# 		sentence = []
			# 		sentence.append(self.embedding.most_similar([batch_x[j, 0]], topn=1)[0][0])
			# 		for word in words:
			# 			sentence.append(self.embedding.most_similar([word], topn=1)[0][0])
			# 		sentence = ' '.join(sentence)
			# 		phrases.append(sentence)
			# 	output = self.session.run(self.g_out, feed_dict={self.x: batch_x[0:3], self.z: batch_z[0:3]})
			# 	for words in batch_x[:3]:
			# 		sentence = []
			# 		for word in words:
			# 			sentence.append(self.embedding.most_similar_cosmul([word], topn=1)[0][0])
			# 		sentence = ' '.join(sentence)
			# 		phrases.append(sentence)

			# 	text_summ = self.session.run(self.text_summary, feed_dict={self.y_: phrases})
			# 	self.writer.add_summary(text_summ, i)


if __name__ == '__main__' :
	model = Model()
	model.init(path='log/generator')
	# model.restore(path='log/generator/6', iteration=9900)
	model.train()



