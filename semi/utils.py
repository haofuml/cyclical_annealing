import numpy as np
# import theano
# from theano import config
import tensorflow as tf
from collections import OrderedDict
import nltk
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from tensorflow.python import pywrap_tensorflow
import pdb
import data_utils
import sys
from tensorflow.python.ops import clip_ops
from rougescore import rouge_n, rouge_1, rouge_2, rouge_l

def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * tf.abs(x)

def sent2idx(text, wordtoix, opt, is_cnn = True):
    
    sent = [wordtoix[x] for x in text.split()]
    
    return prepare_data_for_cnn([sent for i in range(opt.batch_size)], opt)
    


def prepare_data_for_cnn(seqs_x, opt): 
    maxlen=opt.maxlen
    filter_h=opt.filter_shape
    lengths_x = [len(s) for s in seqs_x]
    # print lengths_x
    if maxlen != None:
        new_seqs_x = []
        new_lengths_x = []
        for l_x, s_x in zip(lengths_x, seqs_x):
            if l_x < maxlen:
                new_seqs_x.append(s_x)
                new_lengths_x.append(l_x)
        lengths_x = new_lengths_x
        seqs_x = new_seqs_x
        
        if len(lengths_x) < 1  :
            return None, None
    
    pad = filter_h -1
    x = []   
    for rev in seqs_x:    
        xx = []
        for i in xrange(pad):
            xx.append(0)
        for idx in rev:
            xx.append(idx)
        while len(xx) < maxlen + 2*pad:
            xx.append(0)
        x.append(xx)
    x = np.array(x,dtype='int32')
    return x   
    
    
def prepare_data_for_rnn(seqs_x, opt, is_add_GO = True):
    
    maxlen=opt.maxlen
    lengths_x = [len(s) for s in seqs_x]
    # print lengths_x
    if maxlen != None:
        new_seqs_x = []
        new_lengths_x = []
        for l_x, s_x in zip(lengths_x, seqs_x):
            if l_x < maxlen:
                new_seqs_x.append(s_x)
                new_lengths_x.append(l_x)
        lengths_x = new_lengths_x
        seqs_x = new_seqs_x
        
        if len(lengths_x) < 1  :
            return None, None

    n_samples = len(seqs_x)
    maxlen_x = np.max(lengths_x)
    x = np.zeros(( n_samples, opt.sent_len)).astype('int32')
    for idx, s_x in enumerate(seqs_x):
        if is_add_GO:
            x[idx, 0] = 1 # GO symbol
            x[idx, 1:lengths_x[idx]+1] = s_x
        else:
            x[idx, :lengths_x[idx]] = s_x
    return x   
    


def restore_from_save(t_vars, sess, opt):
    save_keys = tensors_key_in_file(opt.save_path)
    #print(save_keys.keys()) 
    ss = set([var.name for var in t_vars])&set([s+":0" for s in save_keys.keys()])
    cc = {var.name:var for var in t_vars}
    ss_right_shape = set([s for s in ss if cc[s].get_shape() == save_keys[s[:-2]]])  # only restore variables with correct shape
    
    if opt.reuse_discrimination:
        ss2 = set([var.name[2:] for var in t_vars])&set([s+":0" for s in save_keys.keys()])
        cc2 = {var.name[2:][:-2]:var for var in t_vars if var.name[2:] in ss2 if var.get_shape() == save_keys[var.name[2:][:-2]]}
        for s_iter in ss_right_shape:
            cc2[s_iter[:-2]] = cc[s_iter]
        
        loader = tf.train.Saver(var_list=cc2)
        loader.restore(sess, opt.save_path)
        print("Loaded variables for discriminator:"+str(cc2.keys()))
    
    else:    
        # for var in t_vars:
        #     if var.name[:-2] in ss:
        #         tf.assign(t_vars, save_keys[var.name[:-2]])
        loader = tf.train.Saver(var_list= [var for var in t_vars if var.name in ss_right_shape])
        loader.restore(sess, opt.save_path)
        print("Loading variables from '%s'." % opt.save_path)
        print("Loaded variables:"+str(ss_right_shape))
    

    
    

    
    return loader
    
    
_buckets = [(60,60)]    
    
def read_data(source_path, target_path, opt):
    """
    From tensorflow tutorial translate.py
    Read data from source and target files and put into buckets.
    Args:
    source_path: path to the files with token-ids for the source language.
    target_path: path to the file with token-ids for the target language;
      it must be aligned with the source file: n-th line contains the desired
      output for n-th line from the source_path.
    max_size: maximum number of lines to read, all other will be ignored;
      if 0 or None, data files will be read completely (no limit).

    Returns:
    data_set: a list of length len(_buckets); data_set[n] contains a list of
      (source, target) pairs read from the provided data files that fit
      into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
      len(target) < _buckets[n][1]; source and target are lists of token-ids.
    """
    data_set = [[] for _ in _buckets]
    with tf.gfile.GFile(source_path, mode="r") as source_file:
        with tf.gfile.GFile(target_path, mode="r") as target_file:
            source, target = source_file.readline(), target_file.readline()            
            counter = 0
            while source and target and (not opt.max_train_data_size or counter < opt.max_train_data_size):
                counter += 1
                if counter % 100000 == 0:
                    print("  reading data line %d" % counter)
                    sys.stdout.flush()
                source_ids = [int(x) for x in source.split()]
                target_ids = [int(x) for x in target.split()]
                target_ids.append(data_utils.EOS_ID)
                for bucket_id, (source_size, target_size) in enumerate(_buckets):
                    if opt.minlen <len(source_ids) < min(source_size, opt.maxlen) and opt.minlen <len(target_ids) < min(target_size, opt.maxlen):
                        data_set[bucket_id].append([source_ids, target_ids])
                        break
                source, target = source_file.readline(), target_file.readline()
            
            
            
    return data_set    
    
    
    
def prepare_data_for_cnn(seqs_x, opt): 
    maxlen=opt.maxlen
    filter_h=opt.filter_shape
    lengths_x = [len(s) for s in seqs_x]
    # print lengths_x
    if maxlen != None:
        new_seqs_x = []
        new_lengths_x = []
        for l_x, s_x in zip(lengths_x, seqs_x):
            if l_x < maxlen:
                new_seqs_x.append(s_x)
                new_lengths_x.append(l_x)
        lengths_x = new_lengths_x
        seqs_x = new_seqs_x
        
        if len(lengths_x) < 1  :
            return None, None
    
    pad = filter_h -1
    x = []   
    for rev in seqs_x:    
        xx = []
        for i in xrange(pad):
            xx.append(0)
        for idx in rev:
            xx.append(idx)
        while len(xx) < maxlen + 2*pad:
            xx.append(0)
        x.append(xx)
    x = np.array(x,dtype='int32')
    return x
    
    
# def prepare_data_for_machine_translation(pair_x, opt):
#     maxlen=opt.maxlen
#     filter_h=opt.filter_shape
#     def padding(p):
#         pad = filter_h -1
#         new_p = []
#         pdb.set_trace()
#         for it in p:
#             if len(it)>= maxlen:
#                 return None
#             else:
#                 new_p.append([0]*pad + it + [0]*(maxlen-len(it)+pad))
#         return np.array(new_p)
#     return [padding(pair) for pair in pair_x]
    
    
    
    

    
    
          
    

def tensors_key_in_file(file_name):
    """Return tensors key in a checkpoint file.
    Args:
    file_name: Name of the checkpoint file.
    """
    try:
        reader = pywrap_tensorflow.NewCheckpointReader(file_name)
        return reader.get_variable_to_shape_map()
    except Exception as e:  # pylint: disable=broad-except
        print(str(e))
        return None

     
def get_minibatches_idx(n, minibatch_size, shuffle=False):
    idx_list = np.arange(n, dtype="int32")

    if shuffle:
        np.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    # if (minibatch_start != n):
    #     # Make a minibatch out of what is left
    #     minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)
    
    
# def normalizing_L1(x, axis):
#     norm = tf.sqrt(tf.reduce_sum(tf.square(x), axis=axis, keep_dims=True))
#     normalized = x / (norm)
#     return normalized   
    
def normalizing(x, axis):    
    norm = tf.sqrt(tf.reduce_sum(tf.square(x), axis=axis, keep_dims=True))
    normalized = x / (norm)   
    return normalized
    
def _p(pp, name):
    return '%s_%s' % (pp, name)

def dropout(X, trng, p=0.):
    if p != 0:
        retain_prob = 1 - p
        X = X / retain_prob * trng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
    return X

""" used for initialization of the parameters. """

def ortho_weight(ndim):
    W = np.random.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype(config.floatX)
    
def uniform_weight(nin,nout=None, scale=0.05):
    if nout == None:
        nout = nin
    W = np.random.uniform(low=-scale, high=scale, size=(nin, nout))
    return W.astype(config.floatX)
    
def normal_weight(nin,nout=None, scale=0.05):
    if nout == None:
        nout = nin
    W = np.random.randn(nin, nout) * scale
    return W.astype(config.floatX)
    
def zero_bias(ndim):
    b = np.zeros((ndim,))
    return b.astype(config.floatX)

"""auxiliary function for KDE"""
def log_mean_exp(A,b,sigma):
    a=-0.5*((A-theano.tensor.tile(b,[A.shape[0],1]))**2).sum(1)/(sigma**2)
    max_=a.max()
    return max_+theano.tensor.log(theano.tensor.exp(a-theano.tensor.tile(max_,a.shape[0])).mean())

'''calculate KDE'''
def cal_nkde(X,mu,sigma):
    s1,updates=theano.scan(lambda i,s: s+log_mean_exp(mu,X[i,:],sigma), sequences=[theano.tensor.arange(X.shape[0])],outputs_info=[np.asarray(0.,dtype="float32")])
    E=s1[-1]
    Z=mu.shape[0]*theano.tensor.log(sigma*np.sqrt(np.pi*2))
    return (Z-E)/mu.shape[0]


""" BLEU score"""
# def cal_BLEU(generated, reference):
#     #the maximum is bigram, so assign the weight into 2 half.
#     BLEUscore = 0.0
#     for g in generated:
#         BLEUscore += nltk.translate.bleu_score.sentence_bleu(reference, g)
#     BLEUscore = BLEUscore/len(generated)
#     return BLEUscore

def cal_ROUGE(generated, reference, is_corpus = False):
    # ref and sample are both dict
    # scorers = [
    #     (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
    #     (Meteor(),"METEOR"),
    #     (Rouge(), "ROUGE_L"),
    #     (Cider(), "CIDEr")
    # ]
    # output rouge 1-4 and rouge L and rouge L from pycocoevaluate
    
    
    ROUGEscore = [0.0]*6
    for idx, g in enumerate(generated):
        score = [0.0]*6
        if is_corpus:
            for order in range(4):
                score[order] = rouge_n(g.split(), [x.split() for x in reference[0]], order+1, 0.5)
            score[4] = rouge_l(g.split(), [x.split() for x in reference[0]], 0.5)
            score[5], _ = Rouge().compute_score(reference, {0: [g]})
            
            
        else:
            for order in range(4):
                score[order] = rouge_n(g.split(), [reference[0][idx].split()], order+1, 0.5)
            score[4] = rouge_l(g.split(), [reference[0][idx].split()], 0.5)
            score[5], _ = Rouge().compute_score({0: [reference[0][idx]]}, {0: [g]})
            #pdb.set_trace()
        #print g, score
        ROUGEscore = [ r+score[idx]  for idx,r in enumerate(ROUGEscore)] 
        #BLEUscore += nltk.translate.bleu_score.sentence_bleu(reference, g, weight)
    ROUGEscore = [r/len(generated) for r in ROUGEscore]
    return ROUGEscore
    
    


def cal_BLEU(generated, reference, is_corpus = False):
    #print 'in BLEU score calculation'
    #the maximum is bigram, so assign the weight into 2 half.
    BLEUscore = [0.0,0.0,0.0]
    for idx, g in enumerate(generated):
        if is_corpus:
            score, scores = Bleu(4).compute_score(reference, {0: [g]})
        else:
            score, scores = Bleu(4).compute_score({0: [reference[0][idx]]} , {0: [g]})
        #print g, score
        for i, s in zip([0,1,2],score[1:]):
            BLEUscore[i]+=s
        #BLEUscore += nltk.translate.bleu_score.sentence_bleu(reference, g, weight)
    BLEUscore[0] = BLEUscore[0]/len(generated)
    BLEUscore[1] = BLEUscore[1]/len(generated)
    BLEUscore[2] = BLEUscore[2]/len(generated)
    return BLEUscore
    
def cal_BLEU_4(generated, reference, is_corpus = False):
    #print 'in BLEU score calculation'
    #the maximum is bigram, so assign the weight into 2 half.
    BLEUscore = [0.0,0.0,0.0,0.0]
    for idx, g in enumerate(generated):
        if is_corpus:
            score, scores = Bleu(4).compute_score(reference, {0: [g]})
        else:
            score, scores = Bleu(4).compute_score({0: [reference[0][idx]]} , {0: [g]})
        #print g, score
        for i, s in zip([0,1,2,3],score):
            BLEUscore[i]+=s
        #BLEUscore += nltk.translate.bleu_score.sentence_bleu(reference, g, weight)
    BLEUscore[0] = BLEUscore[0]/len(generated)
    BLEUscore[1] = BLEUscore[1]/len(generated)
    BLEUscore[2] = BLEUscore[2]/len(generated)
    BLEUscore[3] = BLEUscore[3]/len(generated)
    return BLEUscore
 
def prepare_for_bleu(sentence):
    sent=[x for x in sentence if x!=0]
    while len(sent)<4:
        sent.append(0)
    #sent = ' '.join([ixtoword[x] for x in sent])
    sent = ' '.join([str(x) for x in sent])
    return sent



def _clip_gradients_seperate_norm(grads_and_vars, clip_gradients):
  """Clips gradients by global norm."""
  gradients, variables = zip(*grads_and_vars)
  clipped_gradients = [clip_ops.clip_by_norm(grad, clip_gradients) for grad in gradients]
  return list(zip(clipped_gradients, variables))
