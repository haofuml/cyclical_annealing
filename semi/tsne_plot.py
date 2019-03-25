import os
import sys
import cPickle
import datetime

import numpy as np
from math import floor
from model_new import *
from utils import prepare_data_for_cnn, get_minibatches_idx, restore_from_save, _clip_gradients_seperate_norm

import tensorflow as tf
from tensorflow.contrib import layers
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


GPUID = 0
os.environ['CUDA_VISIBLE_DEVICES'] = str(GPUID)

class Options(object):
    def __init__(self):
        self.fix_emb = False
        self.reuse_w = False
        self.reuse_cnn = False
        self.reuse_discrimination = False  # reuse cnn for discrimination
        self.restore = True
        self.tanh = False  # activation fun for the top layer of cnn, otherwise relu
        self.model = 'cnn_rnn' #'cnn_deconv'  # 'cnn_rnn', 'rnn_rnn' , default: cnn_deconv

        self.permutation = 0
        self.substitution = 's'  # Deletion(d), Insertion(a), Substitution(s) and Permutation(p)

        self.W_emb = None
        self.cnn_W = None
        self.cnn_b = None
        self.maxlen = 17
        self.n_words = None
        self.filter_shape = 5
        self.filter_size = 300
        self.multiplier = 2
        self.embed_size = 256

        self.layer = 3
        self.stride = [2, 2, 2]   # for two layer cnn/deconv , use self.stride[0]
        self.batch_size = 100
        self.n_gan = 900  # encoder output dim, self.filter_size * 3
        self.L = 100
        self.z_dim = 256
        self.relu_w = False

        self.plot_type = 'cyc' # 'ae', 'vae', 'cyc'
        self.use_z = False
        self.save_path = "./save/%s_pre" % self.plot_type

        # batch norm & dropout
        self.batch_norm = False
        self.dropout = False
        self.dropout_ratio = 0.5

        self.sent_len = self.maxlen + 2*(self.filter_shape-1)
        self.sent_len2 = np.int32(floor((self.sent_len - self.filter_shape)/self.stride[0]) + 1)
        self.sent_len3 = np.int32(floor((self.sent_len2 - self.filter_shape)/self.stride[1]) + 1)
        self.sent_len4 = np.int32(floor((self.sent_len3 - self.filter_shape)/self.stride[2]) + 1)
        print ('Use model %s' % self.model)
        print ('Use %d conv/deconv layers' % self.layer)

    def __iter__(self):
        for attr, value in self.__dict__.iteritems():
            yield attr, value

def sample_z(mu, logvar):
    eps = tf.random_normal(shape=tf.shape(mu))
    return mu + eps * tf.exp(logvar/2.)

def ae(x, opt, is_train=None, opt_t=None):
    # print x.get_shape()  # batch L
    if not opt_t: opt_t = opt
    x_emb, W_norm = embedding(x, opt)  # batch L emb
    x_emb = tf.expand_dims(x_emb, 3)  # batch L emb 1

    res = {}
    # cnn encoder
    H_enc, res = conv_encoder(x_emb, is_train, opt, res)

    # infer latent variable z from H_enc
    biasInit = tf.constant_initializer(0.001, dtype=tf.float32)
    z = layers.linear(H_enc, num_outputs=opt.z_dim, biases_initializer=biasInit, scope='z')

    return z

def vae(x, opt, is_train=None, opt_t=None):
    # print x.get_shape()  # batch L
    if not opt_t: opt_t = opt
    x_emb, W_norm = embedding(x, opt)  # batch L emb
    x_emb = tf.expand_dims(x_emb, 3)  # batch L emb 1

    res = {}
    # cnn encoder
    H_enc, res = conv_encoder(x_emb, is_train, opt, res)

    # infer latent variable z from H_enc
    biasInit = tf.constant_initializer(0.001, dtype=tf.float32)
    mu = layers.linear(H_enc, num_outputs=opt.z_dim, biases_initializer=biasInit, scope='mu')
    logvar = layers.linear(H_enc, num_outputs=opt.z_dim, biases_initializer=biasInit, scope='logvar')

    z = sample_z(mu, logvar)

    return mu, z


def run_model(opt, X):
    try:
        params = np.load('./param_g.npz')
        if params['Wemb'].shape == (opt.n_words, opt.embed_size):
            print('Use saved embedding.')
            opt.W_emb = params['Wemb']
        else:
            print('Emb Dimension mismatch: param_g.npz:'+ str(params['Wemb'].shape) + ' opt: ' + str((opt.n_words, opt.embed_size)))
            opt.fix_emb = False
    except IOError:
        print('No embedding file found.')
        opt.fix_emb = False

    with tf.device('/gpu:0'):
        x_ = tf.placeholder(tf.int32, shape=[opt.batch_size, opt.sent_len])
        if opt.plot_type == 'ae':
            x_lat_ = ae(x_, opt)
        elif opt.plot_type == 'vae' or opt.plot_type == 'cyc':
            mu_, z_ = vae(x_, opt)
            x_lat_ = z_ if opt.use_z else mu_

    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.3
    np.set_printoptions(precision=3)
    np.set_printoptions(threshold=np.inf)
    saver = tf.train.Saver()

    with tf.Session(config = config) as sess:
        sess.run(tf.global_variables_initializer())
        if opt.restore:
            try:
                t_vars = tf.trainable_variables()
                #print([var.name[:-2] for var in t_vars])
                loader = restore_from_save(t_vars, sess, opt)

            except Exception as e:
                print(e)
                print("No saving session, using random initialization")
                sess.run(tf.global_variables_initializer())

        X_emb = np.zeros([len(X), opt.z_dim], dtype='float32')
        kf = get_minibatches_idx(len(X), opt.batch_size)
        t = 0
        for _, index in kf:
        	sents_b = [X[i] for i in index]
        	x_b = prepare_data_for_cnn(sents_b, opt)
        	x_lat = np.squeeze(sess.run(x_lat_, feed_dict={x_:x_b}))
        	X_emb[t*opt.batch_size : (t+1)*opt.batch_size] = x_lat
        	if (t+1) % 10 == 0:
        		print('%d / %d' %(t+1, len(kf)))
        	t += 1

    return X_emb


def main():
    loadpath = "./data/yelp_short_s10.p"
    x = cPickle.load(open(loadpath, "rb"))
    train, val, test = x[0], x[1], x[2]
    train_lab, val_lab, test_lab = x[3], x[4], x[5]
    wordtoix, ixtoword = x[6], x[7]

    batch_num = 50
    np.random.seed(123)
    opt = Options()
    opt.n_words = len(ixtoword)
    sample_idx = np.random.choice(len(test_lab), opt.batch_size*batch_num, replace=False)
    X = [test[ix] for ix in sample_idx]
    y = [test_lab[ix] for ix in sample_idx]
    y = np.array(y)

    print datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")
    X_emb = run_model(opt, X)
    X_emb_2d = TSNE(n_components=2, init='pca').fit_transform(X_emb)
    np.savez('./figs/tsne_%s.npz'%opt.plot_type, X_emb_2d, y)
    blue = y == 0
    red = y == 1
    fig = plt.figure(figsize=(5,5))
    plt.scatter(X_emb_2d[red, 0], X_emb_2d[red, 1], c="r", s=25, edgecolor='none', alpha=0.5)
    plt.scatter(X_emb_2d[blue, 0], X_emb_2d[blue, 1], c="b", s=25, edgecolor='none', alpha=0.5)
    plt.savefig('./figs/tsne_%s.pdf'%opt.plot_type, bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    main()
