import os
import sys
import datetime
import time
GPUID = 0
os.environ['CUDA_VISIBLE_DEVICES'] = str(GPUID)

import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib import layers
#from tensorflow.contrib import metrics
#from tensorflow.contrib.learn import monitors
from tensorflow.contrib import framework
from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow.python.platform import tf_logging as logging
#from tensorflow.contrib.learn.python.learn.metric_spec import MetricSpec
from tensorflow.contrib.tensorboard.plugins import projector
import cPickle
import numpy as np
import os
import scipy.io as sio
from math import floor
import pdb

from model_new import *
from utils import prepare_data_for_cnn, prepare_data_for_rnn, get_minibatches_idx, normalizing, restore_from_save, \
    prepare_for_bleu, cal_BLEU, sent2idx, _clip_gradients_seperate_norm
from denoise import *

profile = False
#import tempfile
#from tensorflow.examples.tutorials.mnist import input_data

logging.set_verbosity(logging.INFO)
#tf.logging.verbosity(1)
# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
#flags.DEFINE_string('train_dir', 'data', 'Directory to put the training data.')

class Options(object):
    def __init__(self):
        self.fix_emb = False
        self.reuse_w = False
        self.reuse_cnn = False
        self.reuse_discrimination = False  # reuse cnn for discrimination
        self.restore = False
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
        self.lr = 1e-4

        self.layer = 3
        self.stride = [2, 2, 2]   # for two layer cnn/deconv , use self.stride[0]
        self.batch_size = 32
        self.max_epochs = 100
        self.n_gan = 900 # encoder output dim, self.filter_size * 3
        self.n_hid = 256 # lstm cell dim
        self.z_dim = 256 # latent dim
        self.init_h_only = True
        self.L = 100
        self.bp_truncation = None

        self.optimizer = 'Adam' #tf.train.AdamOptimizer(beta1=0.9) #'Adam' # 'Momentum' , 'RMSProp'
        self.clip_grad = None  # None  #100  #  20#
        self.attentive_emb = False
        self.decay_rate = 0.99
        self.decay_ep = 2
        self.relu_w = False

        #self.save_path = "./save/" +str(self.n_gan) + "_dim_" + self.model + "_" + self.substitution + str(self.permutation)
        self.save_path = "./save/cyc_pre"
        self.log_path = "./log/cyc_pre"
        self.print_freq = 100
        self.save_freq_ep = -1
        self.save_last = True
        self.vae_anneal = True
        self.anneal_ep = 0 # use cyclical annealing
        self.max_beta = 1.0
        self.cycle_ep = 10

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

def vae(beta, x, x_org, is_train, opt, lr, opt_t=None):
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
    kl_loss = tf.reduce_mean(-0.5 * tf.reduce_sum(1 + logvar - tf.square(mu) - tf.exp(logvar), axis=-1))

    rec_loss, rec_sent_1, _ = lstm_decoder_embedding(z, x_org, W_norm, opt_t, is_train)
    _, rec_sent_2, _ = lstm_decoder_embedding(z, x_org, W_norm, opt_t, is_train, feed_previous=True, is_reuse=True)

    res['rec_sents_feed_y'] = rec_sent_1
    res['rec_sents'] = rec_sent_2

    # compute total loss
    loss = rec_loss + beta * kl_loss
    tf.summary.scalar('beta', beta)
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('kl_loss', kl_loss)
    tf.summary.scalar('rec_loss', rec_loss)
    summaries = [
                "learning_rate",
                "loss",
                # "gradients",
                # "gradient_norm",
                ]
    global_step = tf.Variable(0, trainable=False)

    train_op = layers.optimize_loss(
        loss,
        global_step=global_step,
        #aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N,
        #framework.get_global_step(),
        optimizer=opt.optimizer,
        clip_gradients=(lambda grad: _clip_gradients_seperate_norm(grad, opt.clip_grad)) if opt.clip_grad else None,
        learning_rate=lr,
        summaries=summaries
        )

    return res, loss, rec_loss, kl_loss, train_op


def run_model(opt, train, val, test, test_lab, wordtoix, ixtoword):
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

    min_val_loss = 1e50
    min_test_loss = 1e50
    best_epoch = -1
    epoch_t = len(train)//opt.batch_size
    cycle_t = epoch_t*opt.cycle_ep
    full_kl_step = cycle_t//2

    with tf.device('/gpu:0'):
        beta_ = tf.placeholder(tf.float32, shape=())
        x_ = tf.placeholder(tf.int32, shape=[opt.batch_size, opt.sent_len])
        x_org_ = tf.placeholder(tf.int32, shape=[opt.batch_size, opt.sent_len])
        lr_ = tf.placeholder(tf.float32, shape=(), name='lr')
        is_train_ = tf.placeholder(tf.bool, name='is_train_')
        res_, loss_, rec_loss_, kl_loss_, train_op_ = vae(beta_, x_, x_org_, is_train_, opt, lr_)
        merged = tf.summary.merge_all()

    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.3
    np.set_printoptions(precision=3)
    np.set_printoptions(threshold=np.inf)
    saver = tf.train.Saver()

    def run_epoch(sess, epoch, mode, print_freq=-1, display_sent=-1, train_writer=None):
        fetches_ = {
            'loss': loss_,
            'rec_loss': rec_loss_,
            'kl_loss': kl_loss_
        }

        if mode == 'train':
            x, is_train = train, 1
            fetches_['train_op'] = train_op_
            fetches_['summary'] = merged
        elif mode == 'val':
            assert(print_freq == -1)
            x, is_train = val, None
        elif mode == 'test':
            assert(print_freq == -1)
            x, is_train = test, None
        
        acc_loss, acc_rec, acc_kl, acc_n = 0.0, 0.0, 0.0, 0.0
        local_t = 0
        global_t = epoch*epoch_t # only used in train mode
        start_time = time.time()
        kf = get_minibatches_idx(len(x), opt.batch_size, shuffle=True)
        
        for _, index in kf:
            local_t += 1
            global_t_cyc = global_t % cycle_t
            lr_t = 0.5*opt.lr*(1+np.cos(float(global_t_cyc)/cycle_t*np.pi))
            global_t += 1
            if mode == 'train':
                if opt.vae_anneal:
                    beta_t = opt.max_beta * np.minimum((global_t_cyc+1.)/full_kl_step, 1.)
                else:
                    beta_t = opt.max_beta
            else:
                beta_t = opt.max_beta
            
            sents_b = [x[i] for i in index]
            sents_b_n = add_noise(sents_b, opt)
            x_b_org = prepare_data_for_rnn(sents_b, opt) # Batch L
            x_b = prepare_data_for_cnn(sents_b_n, opt) # Batch L
            feed_t = {beta_: beta_t, x_: x_b, x_org_: x_b_org, is_train_:is_train, lr_:lr_t}
            fetches = sess.run(fetches_, feed_dict=feed_t)
            
            batch_size = len(index)
            acc_n += batch_size
            acc_loss += fetches['loss']*batch_size
            acc_rec += fetches['rec_loss']*batch_size
            acc_kl += fetches['kl_loss']*batch_size
            if print_freq>0 and local_t%print_freq==0:
                print("%s Iter %d: loss %.4f, rec %.4f, kl %.4f, beta %.4f, lr %.4fe-4, time %.1fs" % 
                    (mode, local_t, acc_loss/acc_n, acc_rec/acc_n, acc_kl/acc_n, beta_t, lr_t*1e4, time.time()-start_time))
                sys.stdout.flush()
            if mode == 'train' and train_writer != None:
                train_writer.add_summary(fetches['summary'], global_t)
        
        if display_sent>0:
            index_d = np.random.choice(len(x), opt.batch_size, replace=False)
            sents_d = [x[i] for i in index_d]
            sents_d_n = add_noise(sents_d, opt)
            x_d_org = prepare_data_for_rnn(sents_d, opt) # Batch L
            x_d = prepare_data_for_cnn(sents_d_n, opt) # Batch L
            res = sess.run(res_, feed_dict={beta_: beta_t, x_: x_d, x_org_: x_d_org, is_train_:is_train})
            for i in range(display_sent):
                print("%s Org: "%mode + " ".join([ixtoword[ix] for ix in sents_d[i] if ix!=0 and ix!=2]))
                if mode == 'train':
                    print("%s Rec(feedy): "%mode + " ".join([ixtoword[ix] for ix in res['rec_sents_feed_y'][i] if ix!=0 and ix!=2]))
                print("%s Rec: "%mode + " ".join([ixtoword[ix] for ix in res['rec_sents'][i] if ix!=0 and ix!=2]))
        
        print("%s Epoch %d: loss %.4f, rec %.4f, kl %.4f, beta %.4f, time %.1fs" % 
            (mode, epoch, acc_loss/acc_n, acc_rec/acc_n, acc_kl/acc_n, beta_t, time.time()-start_time))
        return acc_loss/acc_n, acc_rec/acc_n, acc_kl/acc_n

    with tf.Session(config = config) as sess:
        # writer = tf.summary.FileWriter(opt.log_path + '/train', sess.graph)
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

        for epoch in range(opt.max_epochs):
            print("Starting epoch %d" % epoch)
            _, _, _ = run_epoch(sess, epoch, 'train', opt.print_freq, display_sent=1)
            val_loss, _, _ = run_epoch(sess, epoch, 'val', display_sent=1)
            test_loss, _, _ = run_epoch(sess, epoch, 'test', display_sent=1)

            if val_loss < min_val_loss:
                min_val_loss = val_loss
                best_epoch = epoch
                min_test_loss = test_loss
                saver.save(sess, opt.save_path)

            if opt.save_freq_ep>0 and (epoch+1)%opt.save_freq_ep == 0:
                saver.save(sess, opt.save_path, global_step=epoch)

            if opt.save_last:
                saver.save(sess, opt.save_path+'_last')

            print("Min Val Loss %.4f, Min Test Loss %.4f, Best Epoch %d\n" % (min_val_loss, min_test_loss, best_epoch))


def main():
    loadpath = "./data/yelp_short_s10.p"
    x = cPickle.load(open(loadpath, "rb"))
    train, val, test = x[0], x[1], x[2]
    train_lab, val_lab, test_lab = x[3], x[4], x[5]
    wordtoix, ixtoword = x[6], x[7]

    train_lab = np.array(train_lab, dtype='float32')
    val_lab = np.array(val_lab, dtype='float32')
    test_lab = np.array(test_lab, dtype='float32')

    opt = Options()
    opt.n_words = len(ixtoword)
    sys.stdout = open(opt.log_path + '.log.txt', 'w')

    print datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")
    print dict(opt)
    print('Total words: %d' % opt.n_words)
    
    run_model(opt, train, val, test, test_lab, wordtoix, ixtoword)

if __name__ == '__main__':
    main()
