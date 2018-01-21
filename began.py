#!/usr/bin/env python
import os
from tqdm import trange
import numpy as np
import tensorflow as tf
from ops import *
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


class Model(object):
	def __init__(self):
		self.learning_rate = 0.00005
		self.batch_size = 20

		self.start = 1
		self.end = 1000
		self.checkpoint = 300
		self.testpoint = 5


		self.__build()

	def __discriminator(self, x):
		with tf.variable_scope("encoder"):
			with tf.variable_scope('l1'):
				L1 =  conv2d(x, co=64, k=5, bn=False)#28,28,1

			with tf.variable_scope('l2'):
				L2 = tf.nn.relu(conv2d(L1, co=128, k=5)) #28
				# L2 = tf.image.resize_nearest_neighbor(L2, [14,14])
				L2 = tf.layers.max_pooling2d(L2, pool_size=2, strides=2, padding='SAME')

			with tf.variable_scope('l3'):
				L3 = tf.nn.relu(conv2d(L2, co=256, k=5)) #28

			with tf.variable_scope('l4'):
				L4 = tf.nn.relu(conv2d(L3, co=512, k=5)) #14
				# L4 = tf.image.resize_nearest_neighbor(L4, [4,4])
				L4 = tf.layers.max_pooling2d(L4, pool_size=2, strides=2, padding='SAME')

			with tf.variable_scope('l5'):
				L5 = tf.nn.relu(conv2d(L4, co=1028, k=5)) #14

				L5 = tf.reshape(L5, [-1, 7*7*1028])


			with tf.variable_scope('l6'):
				L6 = fc(L5, 100, bn=False)

		with tf.variable_scope("decoder"):
			out = self.__generator(L6)

		return out

	def __generator(self, z):

		with tf.variable_scope('FC1'):
			L1 = fc(z, 16*1028, bn=False)
			L1 = tf.reshape(L1, [-1, 4, 4, 1028])

		with tf.variable_scope('l2'):
			L2 = tf.nn.relu(conv2d(L1, co=512, k=5)) #7
			L2 = tf.image.resize_nearest_neighbor(L2, [14,14])

		with tf.variable_scope('l3'):
			L3 = tf.nn.relu(conv2d(L2, co=256, k=5)) #14

		with tf.variable_scope('l4'):
			L4 = tf.nn.relu(conv2d(L3, co=128, k=5)) #14
			L4 = tf.image.resize_nearest_neighbor(L4, [28,28])

		with tf.variable_scope('l5'):
			L5 = tf.nn.relu(conv2d(L4, co=64, k=5)) #28

		with tf.variable_scope('l6'):
			L6 = conv2d(L5, co=1, k=5, bn=False)#28,28,1

		return L6



	def __build(self):
		self.x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
		self.z = tf.placeholder(tf.float32, shape=[None, 100])

		self.k_t = tf.Variable(0., trainable=False, name='k_t')
		self.gamma = 0.75
		self.lambda_k = 0.001


		with tf.variable_scope("generator") as G:
			self.g_out = self.__generator(self.z)

		with tf.variable_scope("discriminator") as D:

			D_real = self.__discriminator(self.x)
			D.reuse_variables()
			D_false = self.__discriminator(self.g_out)


		d_loss_fake = tf.reduce_mean(tf.abs(D_false-self.g_out))
		d_loss_real = tf.reduce_mean(tf.abs(D_real-self.x))

		# cost functions
		g_loss = d_loss_fake
		d_loss = d_loss_real - self.k_t * d_loss_fake


		balance = self.k_t + self.lambda_k * (self.gamma * d_loss_real - g_loss)
		measure = d_loss_real + tf.abs(self.gamma * d_loss_real - g_loss)

		G_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"generator")
		D_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"discriminator")

		with tf.name_scope("optimizer"):
			self.D_solver = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5, beta2=0.9).minimize(d_loss, var_list = D_var)
			self.G_solver = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5, beta2=0.9).minimize(g_loss, var_list = G_var)


		with tf.control_dependencies([self.D_solver, self.G_solver]):
			self.k_update = tf.assign(self.k_t, tf.clip_by_value(balance, 0, 1))

		#summaries
		x_images = tf.summary.image('True_images', self.x, 8)
		R_images = tf.summary.image('Reconstructed_images', tf.clip_by_value(D_real,0, 1), 8)
		G_images = tf.summary.image('Generated_images', tf.clip_by_value(D_false,0, 1), 8)

		self.Image_summary = tf.summary.merge([x_images, R_images, G_images])


		G_loss_summ = tf.summary.scalar("G_loss", g_loss)
		D_loss_summ = tf.summary.scalar("D_loss", d_loss)
		D_loss_fake_summ = tf.summary.scalar("D_loss/fake", d_loss_fake)
		D_loss_real_summ = tf.summary.scalar("D_loss/real", d_loss_real)

		balance_summ = tf.summary.scalar("misc/balance", balance)
		measure_summ = tf.summary.scalar("misc/measure", measure)

		self.Cost_summary = tf.summary.merge([G_loss_summ, D_loss_summ, D_loss_fake_summ, D_loss_real_summ, balance_summ, measure_summ])


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

		test_z = np.random.normal(0, 1, size=[3, 6, 100])
		for i in trange(self.start, self.end):
			if i%self.checkpoint==0:
				self.__checkpoint(i)

			D_cost_total = 0
			G_cost_total = 0

			batch_x, batch_y = mnist.train.next_batch(100)
			batch_x = batch_x.reshape([-1,28,28,1])
			batch_z = np.random.normal(0, 1, size=[batch_x.shape[0], 100])

			_, __, ___, Loss_summary= self.session.run([self.D_solver, self.G_solver, self.k_update, self.Cost_summary], feed_dict={self.x: batch_x, self.z: batch_z})
			
			self.writer.add_summary(Loss_summary, i)

			if i%self.testpoint==0:
				img_summ = self.session.run(self.Image_summary, feed_dict={self.x: batch_x, self.z: batch_z})
				self.writer.add_summary(img_summ, i)


if __name__ == '__main__' :
	model = Model()
	model.init()
	model.train()



