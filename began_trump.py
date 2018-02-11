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
		self.testpoint = 20
		self.vector_size = 200
		self.z_size = 128

		self.loaded_embedding = False
		
		def load_embedding():
			print("LOADING EMBEDDING...")
			self.embedding = gensim.models.KeyedVectors.load_word2vec_format('data/glove_word2vec.txt')
			self.embedding = self.embedding.wv
			print("LOADED EMBEDDING!")
			self.loaded_embedding = True

		t = threading.Thread(target=load_embedding)
		t.start()

		print("BUILDING MODEL...")
		self.__build()



	def __discriminator(self, x):
		with tf.variable_scope("encoder"):
			x_shape = tf.shape(x)
			#adding extra dim which might help generating correct ending of sequence
			with tf.name_scope("time_step"):
				step_size = tf.minimum(x_shape[1], 5)
				steps = tf.range(tf.cast(step_size, dtype=tf.float32)-1, -1, -1, dtype=tf.float32)*0.45
				steps = tf.tanh(steps)
				step_ = tf.ones([x_shape[1]-step_size])
				steps = tf.concat([step_, steps],0)
				steps = tf.reshape(steps, [1, -1, 1])

				steps = tf.tile(steps, [x_shape[0], 1, 1])
				z = tf.concat([x, steps], 2)

			L1 = tf.nn.rnn_cell.LayerNormBasicLSTMCell(512)
			L2 = tf.nn.rnn_cell.LayerNormBasicLSTMCell(1024)
			L3 = tf.nn.rnn_cell.LayerNormBasicLSTMCell(128)

			multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell([L1, L2, L3])
			initial_state = multi_rnn_cell.zero_state(x_shape[0], tf.float32)

			rnn_out, state = tf.nn.dynamic_rnn(multi_rnn_cell, z, initial_state=initial_state, time_major=False)

		with tf.variable_scope("decoder"):
			rnn_out = self.__generator(rnn_out)

		return rnn_out


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

		L1 = tf.nn.rnn_cell.LayerNormBasicLSTMCell(512)
		L2 = tf.nn.rnn_cell.LayerNormBasicLSTMCell(1024)
		L3 = tf.nn.rnn_cell.LayerNormBasicLSTMCell(200)

		multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell([L1, L2, L3])
		initial_state = multi_rnn_cell.zero_state(z_shape[0], tf.float32)

		rnn_out, state = tf.nn.dynamic_rnn(multi_rnn_cell, z, initial_state=initial_state, time_major=False)

		return rnn_out


	def __build(self):
		self.x = tf.placeholder(tf.float32, shape=[None, None, self.vector_size])
		self.z = tf.placeholder(tf.float32, shape=[None, None, self.z_size])
		self.k_t = tf.Variable(0., trainable=False, name='k_t')
		self.gamma = 1
		self.lambda_k = 0.001

		self.y_ = tf.placeholder(tf.string)


		with tf.variable_scope("generator") as G:
			self.g_out = self.__generator(self.z)

		with tf.variable_scope("discriminator") as D:

			self.D_real = self.__discriminator(self.x)
			D.reuse_variables()
			self.D_false = self.__discriminator(self.g_out)

		d_loss_fake = tf.reduce_mean(tf.square(self.D_false-self.g_out))
		d_loss_real = tf.reduce_mean(tf.square(self.D_real-self.x))

		# cost functions
		g_loss = d_loss_fake
		d_loss = d_loss_real - self.k_t * d_loss_fake


		balance = self.k_t + self.lambda_k * (self.gamma * d_loss_real - d_loss_fake)
		measure = d_loss_real + tf.abs(self.gamma * d_loss_real - d_loss_fake)

		G_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"generator")
		D_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"discriminator")

		with tf.name_scope("optimizer"):
			self.D_solver = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5).minimize(d_loss, var_list = D_var)
			self.G_solver = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5).minimize(g_loss, var_list = G_var)


		with tf.control_dependencies([self.D_solver, self.G_solver]):
			self.k_update = tf.assign(self.k_t, tf.clip_by_value(balance, 0, 1))

		#summaries
		G_loss_summ = tf.summary.scalar("G_loss", g_loss)
		D_loss_summ = tf.summary.scalar("D_loss", d_loss)
		D_loss_fake_summ = tf.summary.scalar("D_loss/fake", d_loss_fake)
		D_loss_real_summ = tf.summary.scalar("D_loss/real", d_loss_real)


		balance_summ = tf.summary.scalar("misc/balance", balance)
		measure_summ = tf.summary.scalar("misc/measure", measure)

		self.Cost_summary = tf.summary.merge([G_loss_summ, D_loss_summ, D_loss_fake_summ, D_loss_real_summ, balance_summ, measure_summ])

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

			_, __, Loss_summary= self.session.run([self.D_solver, self.G_solver, self.Cost_summary], feed_dict={self.x: batch_x, self.z: batch_z})
			
			self.writer.add_summary(Loss_summary, i)


			if i%self.testpoint==0 and self.loaded_embedding:
				phrases = []
				output_g, reconstructed_g, reconstructed_r = self.session.run([self.g_out, self.D_false, self.D_real], feed_dict={self.x: batch_x[:5], self.z: batch_z[:5]})
				for output in [output_g, reconstructed_g, batch_x[:5], reconstructed_r]:
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
	model.init()
	# model.restore(path='log/8', iteration=500)
	model.train()



