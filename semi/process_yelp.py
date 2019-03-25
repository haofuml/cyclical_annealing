import numpy as np
import random
import cPickle
from collections import Counter
from nltk import word_tokenize, sent_tokenize

def set_global_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)

def load_data(path, type, percent=-1):
    assert(type in ['train','dev','test'])
    assert(percent<0 or (percent>0 and percent<100))
    path0 = path + type + '.0'
    path1 = path + type + '.1'

    x0 = load_sent(path0, -1)
    x1 = load_sent(path1, -1)

    if percent>0:
        x0 = x0[:len(x0)*percent/100]
        x1 = x1[:len(x1)*percent/100]

    x = x0 + x1
    y = ([0]*len(x0)) + ([1]*len(x1))
    return x, y

def load_sent(path, max_size=-1):
    data = []
    with open(path) as f:
        for line in f:
            if len(data) == max_size:
                break
            data.append(line.split())
    return data

def build_vocab(data, min_occur=5):
    word2id = {'<pad>':0, '<go>':1, '<eos>':2, '<unk>':3}
    id2word = {0:'<pad>', 1:'<go>', 2:'<eos>', 3:'<unk>'}

    words = [word for sent in data for word in sent]
    cnt = Counter(words)
    for word in cnt:
        if cnt[word] >= min_occur:
            n = len(word2id)
            word2id[word] = n
            id2word[n] = word
    vocab_size = len(word2id)
    return word2id, id2word

def convert2id(x, word2id, add_go=True, add_eos=True):
    go = word2id['<go>']
    eos = word2id['<eos>']
    unk = word2id['<unk>']
    x_id = []

    for sent in x:
        sent_id = [word2id[w] if w in word2id else unk for w in sent]
        if add_go:
            sent_id.insert(0, go)
        if add_eos:
            sent_id.append(eos)
        x_id.append(sent_id)        
    return x_id

def get_stats(x, y, type):
    assert(type in ['train','dev','test'])
    x_l = np.array([len(s) for s in x])
    y_sum = sum(y)

    print("%s data: max_len %d, min_len %d, avg_len %.1f, all_size %d, neg_size %d, pos_size %d" 
        % (type, x_l.max(), x_l.min(), x_l.mean(), len(y), len(y)-y_sum, y_sum))

def main():
    set_global_seeds(1)
    save_path = './data/yelp_short'
    percent = -1

    data_path = './data/yelp/sentiment.'
    train_x, train_y = load_data(data_path, 'train', percent)
    dev_x, dev_y = load_data(data_path, 'dev', percent)
    test_x, test_y = load_data(data_path, 'test', percent)

    word2id, id2word = build_vocab(train_x)
    print('vocab size:%d'%len(word2id))
    train_x= convert2id(train_x, word2id, add_go=False)
    dev_x= convert2id(dev_x, word2id, add_go=False)
    test_x= convert2id(test_x, word2id, add_go=False)

    get_stats(train_x, train_y, 'train')
    get_stats(dev_x, dev_y, 'dev')
    get_stats(test_x, test_y, 'test')

    save_path += '.p' if percent<0 else '_s%d.p'%percent
    with open(save_path, 'wb') as f:
        cPickle.dump((train_x, dev_x, test_x, train_y, dev_y, test_y, word2id, id2word), 
            f, cPickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()

