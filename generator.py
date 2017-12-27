#!/usr/bin/env python
import os
from tqdm import trange
import numpy as np
import tensorflow as tf
from ops import *
import gensim


class Model(object):
	def __init__(self):
		self.learning_rate = 0.0005
		self.batch_size = 20

		self.start = 0
		self.end = 1200
		self.checkpoint = 300
		self.testpoint = 100
		self.time_steps = 45
		self.vector_size = 200

		self.embedding = gensim.models.KeyedVectors.load_word2vec_format('data/glove_word2vec.txt')
		self.embedding = self.embedding.wv

		self.__build()



	def __generator(self, x, z, t):
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
		rnn_out = tf.cond(t, lambda: self.train_rnn(multi_rnn_cell, input), lambda: self.test_rnn(multi_rnn_cell, x, z))

		return rnn_out

	def train_rnn(self, multi_rnn_cell, input):
		initial_state = multi_rnn_cell.zero_state(tf.shape(input)[0], tf.float32)

		#rnn_out = [batch_size, max_time, data]
		rnn_out, state = tf.nn.dynamic_rnn(multi_rnn_cell, input, initial_state=initial_state, time_major=False)
		return rnn_out

	def test_rnn(self, multi_rnn_cell, x, z):
		initial_state = multi_rnn_cell.zero_state(tf.shape(x)[0], tf.float32)

		def condition(x, state, counter):
			return tf.less(counter, tf.shape(z)[1])

		def loop_body(x, state, counter):
			y = tf.concat([x[:,counter:counter+1], z[:,counter:counter+1]], 2)
			rnn_out, state = tf.nn.dynamic_rnn(multi_rnn_cell, y, initial_state=state, time_major=False)
			x = tf.concat([x, rnn_out],1)
			return x, state, tf.add(counter, 1)

		rnn_out, state, counter = tf.while_loop(condition, loop_body, [x[:,0:1], initial_state, 0])
		return rnn_out


	def __build(self):
		self.x = tf.placeholder(tf.float32, shape=[None, None, self.vector_size], name='INPUT')
		self.z = tf.placeholder(tf.float32, shape=[None, None, 50], name='RANDOM')
		self.t = tf.placeholder_with_default(True, shape=None, name='TRAIN')

		self.y_ = tf.placeholder(tf.string)


		with tf.variable_scope("generator") as G:
			self.g_out = self.__generator(self.x, self.z, self.t)

		with tf.name_scope("optimizer"):
			self.G_loss = tf.reduce_mean(tf.square(self.g_out - self.x[:,1:]))

			G_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"generator")
			self.G_solver = tf.train.AdamOptimizer(self.learning_rate,).minimize(self.G_loss, var_list = G_var)

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
		D_cost = 0
		G_cost = 0
		for i in trange(self.start, self.end):
			if i%self.checkpoint==0:
				self.__checkpoint(i)
			D_cost_total = 0
			G_cost_total = 0
			for ix in trange(0, 10):
				batch_x = data_reader.next_batch()
				batch_z = np.random.normal(0, 1, size=[batch_x.shape[0], batch_x.shape[1]-1, 50])
				output = self.session.run(self.g_out, feed_dict={self.x: batch_x[0:3], self.z: batch_z[0:3], self.t:False})
				_, G_cost= self.session.run([self.G_solver, self.G_loss], feed_dict={self.x: batch_x, self.z: batch_z})

				G_cost_total+=G_cost

			G_cost_total = tf.Summary(value=[tf.Summary.Value(tag="Generator", simple_value=G_cost_total)])
			self.writer.add_summary(G_cost_total, i)

			if i%self.testpoint==0:
				phrases = []
				output = self.session.run(self.g_out, feed_dict={self.x: batch_x[0:3], self.z: batch_z[0:3], self.t:False})
				for j, words in enumerate(output):
					sentence = []
					sentence.append(self.embedding.most_similar([batch_x[j, 0]], topn=1)[0][0])
					for word in words:
						sentence.append(self.embedding.most_similar([word], topn=1)[0][0])
					sentence = ' '.join(sentence)
					phrases.append(sentence)
				output = self.session.run(self.g_out, feed_dict={self.x: batch_x[0:3], self.z: batch_z[0:3]})
				for j, words in enumerate(output):
					sentence = []
					sentence.append(self.embedding.most_similar([batch_x[j, 0]], topn=1)[0][0])
					for word in words:
						sentence.append(self.embedding.most_similar([word], topn=1)[0][0])
					sentence = ' '.join(sentence)
					phrases.append(sentence)
				for words in batch_x[:3]:
					sentence = []
					for word in words:
						sentence.append(self.embedding.most_similar_cosmul([word], topn=1)[0][0])
					sentence = ' '.join(sentence)
					phrases.append(sentence)

				text_summ = self.session.run(self.text_summary, feed_dict={self.y_: phrases})
				self.writer.add_summary(text_summ, i)


if __name__ == '__main__' :
	model = Model()
	model.init()
	model.train()



