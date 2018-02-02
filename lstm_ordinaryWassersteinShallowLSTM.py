#!/usr/bin/env python
import os
from tqdm import trange
import numpy as np
import tensorflow as tf
from ops import *
import gensim
from config import OrdinaryWasserSteinShallow


class Model(object):
    def __init__(self):
        param = OrdinaryWasserSteinShallow.param

        self.learning_rate  = param["learning_rate"]
        self.batch_size     = param["batch_size"]

        self.start          = param["start"]
        self.end            = param["end"]
        self.checkpoint     = param["checkpoint"]
        self._embedding     = param["_embedding"]
        self.testpoint      = param["testpoint"]
        self.time_steps     = param["time_steps"]
        self.vector_size    = param["vector_size"]

        #wasserstein weight clipping
        self.w_clipping     = param["w_clipping"]
        self.n_critic       = param["n_critic"]

        #dropout
        self.drop_D_inter   = param["drop_D_inter"]     #in between convolutional discriminator layers.
        self.drop_input     = param["drop_input"]       #dropout on input data
        self.drop_output    = param["drop_output"]      #dropout on output
        self.drop_rnn       = param["drop_rnn"]         #dropout in recurrent connections

        self.__build()

    def embedding(self):
        if self._embedding is None:
            print("LOADING EMBEDDING...", end=" ")
            self._embedding = gensim.models.KeyedVectors.load_word2vec_format('data/glove_word2vec.txt')
            self._embedding = self._embedding.wv
            print("DONE")
        return self._embedding


    def __discriminator(self, x):
        x_shape = tf.shape(x)

        #adding extra dim to suggest sentence length
        with tf.name_scope("time_step_D"):
            step_size = tf.minimum(x_shape[1], 5)
            steps = tf.range(tf.cast(step_size, dtype=tf.float32)-1, -1, -1, dtype=tf.float32)*0.45
            steps = tf.tanh(steps)
            step_ = tf.ones([x_shape[1]-step_size])
            steps = tf.concat([step_, steps],0)
            steps = tf.reshape(steps, [1, -1, 1])

            steps = tf.tile(steps, [x_shape[0], 1, 1])
            x = tf.concat([x, steps], 2)

        # #batch normalization is necessary to prevent vanishing gradient, according to WGAN paper.
        L1 = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LayerNormBasicLSTMCell(1024),
                                        input_keep_prob = 1 - self.drop_input, output_keep_prob = 1 - self.drop_output, state_keep_prob = 1 - self.drop_rnn)
        L2 = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LayerNormBasicLSTMCell(128),#single output vs fully connected layer instead?
                                        input_keep_prob = 1 - self.drop_input, output_keep_prob = 1 - self.drop_output, state_keep_prob = 1 - self.drop_rnn)

        #bind rnn layers
        multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell([L1, L2])
        initial_state = multi_rnn_cell.zero_state(x_shape[0], tf.float32)
        
        #create the dynamic rnn
        rnn_out, state = tf.nn.dynamic_rnn(multi_rnn_cell, x, initial_state=initial_state, time_major=False)

        #state contains final state of all layers. Grab the final outputs of last layers
        final_state = state[1][1]   #(lstmtupleL1, lstmtupleL2) where lstmtupleL2 = (hidden state, outputs)

        #fully connected classifier on top
        classified = fc(final_state, 1, bn=False)
        return classified


    def __generator(self, z):
        z_shape = tf.shape(z)
        #adding extra dim which might help generating correct ending of sequenc
        with tf.name_scope("time_step"):
            step_size = tf.minimum(z_shape[1], 5)
            steps = tf.range(tf.cast(step_size, dtype=tf.float32)-1, -1, -1, dtype=tf.float32)*0.45
            steps = tf.tanh(steps)
            step_ = tf.ones([z_shape[1]-step_size])
            steps = tf.concat([step_, steps],0)
            steps = tf.reshape(steps, [1, -1, 1])

            steps = tf.tile(steps, [z_shape[0], 1, 1])
            z = tf.concat([z, steps], 2)

        #batch normalization is necessary to prevent vanishing gradient, according to WGAN paper.
        L1 = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LayerNormBasicLSTMCell(2048),
                                        input_keep_prob = 1 - self.drop_input, output_keep_prob = 1 - self.drop_output, state_keep_prob = 1 - self.drop_rnn)
        L2 = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LayerNormBasicLSTMCell(self.vector_size),
                                        input_keep_prob = 1 - self.drop_input, output_keep_prob = 1 - self.drop_output, state_keep_prob = 1 - self.drop_rnn)

        multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell([L1, L2])
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

        # cost functions
        self.G_loss = -tf.reduce_mean(D_false)
        self.D_loss = tf.reduce_mean(D_false) - tf.reduce_mean(D_real)# + self.gradient_penalty #comment gradient penalty for ordinary gan. Also add weight clipping

        G_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"generator")
        D_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"discriminator")

        with tf.name_scope("optimizer"):
            #use rmsprop for ordinary wasserstein as opposed to adam.
            self.D_solver = tf.train.RMSPropOptimizer(self.learning_rate, decay=0.9, momentum=0.0).minimize(self.D_loss, var_list = D_var)
            self.G_solver = tf.train.RMSPropOptimizer(self.learning_rate, decay=0.9, momentum=0.0).minimize(self.G_loss, var_list = G_var)
            # self.D_solver = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5, beta2=0.9).minimize(self.D_loss, var_list = D_var)
            # self.G_solver = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5, beta2=0.9).minimize(self.G_loss, var_list = G_var)

        #collect a clipping operation for each weight variable to be run in session
        self.D_weight_clipping = [var.assign(tf.clip_by_value(var, -self.w_clipping, self.w_clipping)) for var in D_var]

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

            #train discriminator first for n_critic times
            for ix in trange(0, self.n_critic):
                batch_x = data_reader.next_batch()
                batch_z = np.random.normal(0, 1, size=[batch_x.shape[0], batch_x.shape[1], 100])

                _ = self.session.run([self.D_solver, self.D_weight_clipping], feed_dict={self.x: batch_x, self.z: batch_z})

            batch_x = data_reader.next_batch()
            batch_z = np.random.normal(0, 1, size=[batch_x.shape[0], batch_x.shape[1], 100])

            _, Loss_summary, Dist_summary= self.session.run([self.G_solver, self.Cost_summary, self.Distribution_summary], feed_dict={self.x: batch_x, self.z: batch_z})

            self.writer.add_summary(Loss_summary, i)
            self.writer.add_summary(Dist_summary, i)


            if i%self.testpoint==0:
                phrases = []
                output = self.session.run(self.g_out, feed_dict={self.z: test_z})
                for words in output:
                    sentence = []
                    for word in words:
                        top_words = self.embedding().most_similar([word], topn=1)
                        top_word = top_words[0][0]
                        sentence.append(top_word)
                    sentence = ' '.join(sentence)
                    phrases.append(sentence)
                rand = np.random.randint(0,self.batch_size-2)
                for words in batch_x[rand:rand+2]:
                    sentence = []
                    for word in words:
                        sentence.append(self.embedding().most_similar_cosmul([word], topn=1)[0][0])
                    sentence = ' '.join(sentence)
                    phrases.append(sentence)

                text_summ = self.session.run(self.text_summary, feed_dict={self.y_: phrases})
                self.writer.add_summary(text_summ, i)


if __name__ == '__main__' :
    model = Model()
    model.init()
    model.train()
