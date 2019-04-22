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
    prepare_for_bleu, cal_BLEU, sent2idx, _clip_gradients_seperate_norm, set_global_seeds
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
        self.lr = 1e-4
        self.category = 1
        self.seed = 123

        self.part_data = True
        self.train_percent = 10  # 10%  1%

        self.layer = 3
        self.stride = [2, 2, 2]   # for two layer cnn/deconv , use self.stride[0]
        self.batch_size = 32
        self.max_epochs = 1000
        self.n_gan = 900 # encoder output dim, self.filter_size * 3
        self.n_hid = 256 # lstm cell dim
        self.z_dim = 256 # latent dim
        self.L = 100

        self.optimizer = 'Adam' #tf.train.AdamOptimizer(beta1=0.9) #'Adam' # 'Momentum' , 'RMSProp'
        self.clip_grad = None  # None  #100  #  20#
        self.attentive_emb = False
        self.decay_rate = 0.99
        self.decay_ep = 100
        self.relu_w = False

        #self.save_path = "./save/" +str(self.n_gan) + "_dim_" + self.model + "_" + self.substitution + str(self.permutation)
        self.save_path = "./save/ae_pre"
        self.log_path = "./log/ae_pre_cla"
        self.save_freq_ep = -1
        self.save_last = False

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

def ae(x, y, is_train, opt, epoch_t, opt_t=None):
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

    logits = discriminator_linear(z, opt, prefix='classify_', is_train=is_train)  # batch * 1
    prob = tf.nn.sigmoid(logits)

    correct_prediction = tf.equal(tf.round(prob), tf.round(y))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits))

    tf.summary.scalar('loss', loss)
    summaries = [
                "learning_rate",
                "loss",
                # "gradients",
                # "gradient_norm",
                ]
    global_step = tf.Variable(0, trainable=False)

    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if 'dis' in var.name]

    train_op = layers.optimize_loss(
        loss,
        global_step=global_step,
        variables=d_vars,
        #aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N,
        #framework.get_global_step(),
        optimizer=opt.optimizer,
        clip_gradients=(lambda grad: _clip_gradients_seperate_norm(grad, opt.clip_grad)) if opt.clip_grad else None,
        learning_rate_decay_fn=lambda lr, g: tf.train.exponential_decay(learning_rate=lr, global_step = g, 
            decay_rate=opt.decay_rate, decay_steps=int(epoch_t*opt.decay_ep)),
        learning_rate=opt.lr,
        summaries=summaries
        )

    return loss, train_op, accuracy


def run_model(opt, train, val, test, train_lab, val_lab, test_lab, wordtoix, ixtoword):
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

    min_val_loss = 1e30
    min_test_loss = 1e30
    max_val_accuracy = 0.
    max_test_accuracy = 0.
    best_epoch = -1
    epoch_t = len(train)//opt.batch_size

    with tf.device('/gpu:0'):
        x_ = tf.placeholder(tf.int32, shape=[opt.batch_size, opt.sent_len])
        y_ = tf.placeholder(tf.float32, shape=[opt.batch_size, 1])
        is_train_ = tf.placeholder(tf.bool, name='is_train_')
        loss_, train_op_, accuracy_ = ae(x_, y_, is_train_, opt, epoch_t)
        merged = tf.summary.merge_all()

    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.3
    np.set_printoptions(precision=3)
    np.set_printoptions(threshold=np.inf)
    saver = tf.train.Saver()

    def run_epoch(sess, epoch, mode, print_freq=-1, train_writer=None):
        fetches_ = {
            'loss': loss_,
            'accuracy': accuracy_
        }

        if mode == 'train':
            x, y, is_train = train, train_lab, 1
            fetches_['train_op'] = train_op_
            fetches_['summary'] = merged
        elif mode == 'val':
            assert(print_freq == -1)
            x, y, is_train = val, val_lab, None
        elif mode == 'test':
            assert(print_freq == -1)
            x, y, is_train = test, test_lab, None

        correct, acc_loss, acc_n = 0.0, 0.0, 0.0
        local_t = 0
        global_t = epoch*epoch_t # only used in train mode
        start_time = time.time()
        kf = get_minibatches_idx(len(x), opt.batch_size, shuffle=True)

        for _, index in kf:
            local_t += 1
            global_t += 1

            sents_b = [x[i] for i in index]
            sents_b_n = add_noise(sents_b, opt)
            y_b = [y[i] for i in index]
            y_b = np.array(y_b)
            y_b = y_b.reshape((len(y_b), 1))
            x_b = prepare_data_for_cnn(sents_b_n, opt) # Batch L
            feed_t = {x_: x_b, y_: y_b, is_train_: is_train}
            fetches = sess.run(fetches_, feed_dict=feed_t)

            batch_size = len(index)
            acc_n += batch_size
            acc_loss += fetches['loss']*batch_size
            correct += fetches['accuracy']*batch_size
            if print_freq>0 and local_t%print_freq==0:
                print("%s Iter %d: loss %.4f, acc %.4f, time %.1fs" %
                    (mode, local_t, acc_loss/acc_n, correct/acc_n, time.time()-start_time))
            if mode == 'train' and train_writer != None:
                train_writer.add_summary(fetches['summary'], global_t)

        print("%s Epoch %d: loss %.4f, acc %.4f, time %.1fs" %
            (mode, epoch, acc_loss/acc_n, correct/acc_n, time.time()-start_time))
        return acc_loss/acc_n, correct/acc_n

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
            _, _ = run_epoch(sess, epoch, 'train')
            val_loss, val_accuracy = run_epoch(sess, epoch, 'val')
            test_loss, test_accuracy = run_epoch(sess, epoch, 'test')

            # if val_loss < min_val_loss:
            #     min_val_loss = val_loss
            #     min_test_loss = test_loss
            #     best_epoch = epoch
            #     max_test_accuracy = test_accuracy
            #     saver.save(sess, opt.save_path+"_cla")
            if val_accuracy > max_val_accuracy:
                max_val_accuracy = val_accuracy
                min_test_loss = test_loss
                best_epoch = epoch
                max_test_accuracy = test_accuracy
                saver.save(sess, opt.save_path+"_cla")

            if opt.save_freq_ep>0 and (epoch+1)%opt.save_freq_ep == 0:
                saver.save(sess, opt.save_path+"_cla", global_step=epoch)

            if opt.save_last:
                saver.save(sess, opt.save_path+'_cla_last')

            # print("Min Val Loss %.4f, Min Test Loss %.4f, Max Test Acc %.4f, Best Epoch %d\n" %
            #       (min_val_loss, min_test_loss, max_test_accuracy, best_epoch))
            print("Max Val Acc %.4f, Min Test Loss %.4f, Max Test Acc %.4f, Best Epoch %d\n" %
                  (max_val_accuracy, min_test_loss, max_test_accuracy, best_epoch))

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
    set_global_seeds(opt.seed)
    opt.n_words = len(ixtoword)
    sys.stdout = open(opt.log_path + '.log.txt', 'w')

    print datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")
    print dict(opt)
    print('Total words: %d' % opt.n_words)
    
    if opt.part_data:
        # np.random.seed(123)
        train_ind = np.random.choice(len(train_lab), int(len(train_lab)*opt.train_percent/100), replace=False)
        train = [train[t] for t in train_ind]
        train_lab = [train_lab[t] for t in train_ind]

    run_model(opt, train, val, test, train_lab, val_lab, test_lab, wordtoix, ixtoword)

if __name__ == '__main__':
    main()
