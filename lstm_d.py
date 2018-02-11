#!/usr/bin/env python
import os
from tqdm import trange
import numpy as np
import tensorflow as tf
from ops import *


# from tensorflow.examples.tutorials.mnist import input_data
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

		self.__build()

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
			l3 = fc(l2, 512, bn=False)
		with tf.variable_scope('FC2'):
			l4 = fc(l3, 1, bn=False)
		return l4, tf.nn.sigmoid(l4)


	def __build(self):
		self.x = tf.placeholder(tf.float32, shape=[None, None, self.vector_size])
		self.y = tf.placeholder(tf.float32, shape=[None, None, self.vector_size])


		with tf.variable_scope("discriminator") as D:

			D_real, D_real_sig = self.__discriminator(self.x)
			D.reuse_variables()
			D_false, D_false_sig = self.__discriminator(self.y)


		D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real, labels=(tf.ones_like(D_real)-0.1)))
		D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_false, labels=tf.zeros_like(D_false)))

		self.D_loss = D_loss_real + D_loss_fake

		D_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"discriminator")

		with tf.name_scope("optimizer"):
			self.D_solver = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5, beta2=0.9).minimize(self.D_loss, var_list = D_var)

		accuracy = tf.reduce_mean(tf.concat([tf.cast(tf.greater(D_real_sig, 0.5), tf.float32),tf.cast(tf.less(D_false_sig, 0.5), tf.float32)],0))
		accuracy_summ = tf.summary.scalar("accuracy", accuracy)

		#summaries
		Distribution_True = tf.summary.histogram("distribution/true", D_real)
		Distribution_False = tf.summary.histogram("distribution/false", D_false)
		Distribution_Total = tf.summary.histogram("distribution/both", tf.concat([D_real, D_false], 0))
		self.Distribution_summary = tf.summary.merge([Distribution_True, Distribution_False, Distribution_Total])

		D_loss_summ = tf.summary.scalar("D_loss", self.D_loss)

		self.Cost_summary = tf.summary.merge([D_loss_summ, accuracy_summ])

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
		data_reader_t = Data_reader_(data="data/trump_embedding_dynamic.npy", batch_size=20, min_batch_size=20)
		data_reader_g = Data_reader_(data="data/geo_embedding_dynamic.npy", batch_size=20, min_batch_size=20)
		for i in trange(self.start, self.start+self.iterations):
			if i%self.checkpoint==0:
				self.__checkpoint(i)

			batch_x = data_reader_t.next_batch()
			batch_y = data_reader_g.next_batch()
			while batch_y.shape[1] != batch_x.shape[1]:
				batch_y = data_reader_g.next_batch()

			_, Loss_summary, Dist_summary = self.session.run([self.D_solver, self.Cost_summary, self.Distribution_summary], feed_dict={self.x: batch_x, self.y: batch_y})

			self.writer.add_summary(Loss_summary, i)
			self.writer.add_summary(Dist_summary, i)

			



if __name__ == '__main__' :
	model = Model()
	model.init()
	# model.restore(path='log/8', iteration=500)
	model.train()



