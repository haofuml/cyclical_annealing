#    Copyright (C) 2017 Tiancheng Zhao, Carnegie Mellon University

import os
import time

import numpy as np
import tensorflow as tf
from beeprint import pp

from config_utils_cyc import KgCVAEConfig as Config
from data_apis.corpus import SWDADialogCorpus
from data_apis.data_utils import SWDADataLoader
from models.cvae_cyc import KgRnnCVAE

import logger

# constants
tf.app.flags.DEFINE_string("word2vec_path", "./glove_twitter_27B_200d.txt", "The path to word2vec. Can be None.")
tf.app.flags.DEFINE_string("data_dir", "data/full_swda_clean_42da_sentiment_dialog_corpus.p", "Raw data directory.")
tf.app.flags.DEFINE_string("work_dir", "working", "Experiment results directory.")
tf.app.flags.DEFINE_bool("equal_batch", True, "Make each batch has similar length.")
tf.app.flags.DEFINE_bool("resume", False, "Resume from previous")
tf.app.flags.DEFINE_bool("forward_only", False, "Only do decoding")
tf.app.flags.DEFINE_bool("save_model", True, "Create checkpoints")
tf.app.flags.DEFINE_string("sub_dir", "cvae_cyc_elbo", "the dir to load checkpoint for forward only")
FLAGS = tf.app.flags.FLAGS


def main():
    # config for training
    config = Config()
    config.use_bow = False

    # config for validation
    valid_config = Config()
    valid_config.keep_prob = 1.0
    valid_config.dec_keep_prob = 1.0
    valid_config.batch_size = 60
    valid_config.use_bow = False

    # configuration for testing
    test_config = Config()
    test_config.keep_prob = 1.0
    test_config.dec_keep_prob = 1.0
    test_config.batch_size = 1
    test_config.use_bow = False

    pp(config)

    # get data set
    api = SWDADialogCorpus(FLAGS.data_dir, word2vec=FLAGS.word2vec_path, word2vec_dim=config.embed_size)
    dial_corpus = api.get_dialog_corpus()
    meta_corpus = api.get_meta_corpus()

    train_meta, valid_meta, test_meta = meta_corpus.get("train"), meta_corpus.get("valid"), meta_corpus.get("test")
    train_dial, valid_dial, test_dial = dial_corpus.get("train"), dial_corpus.get("valid"), dial_corpus.get("test")

    # convert to numeric input outputs that fits into TF models
    train_feed = SWDADataLoader("Train", train_dial, train_meta, config)
    valid_feed = SWDADataLoader("Valid", valid_dial, valid_meta, config)
    test_feed = SWDADataLoader("Test", test_dial, test_meta, config)

    # begin training
    sess_config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    # sess_config.gpu_options.allow_growth = True
    sess_config.gpu_options.per_process_gpu_memory_fraction = 0.45
    with tf.Session(config=sess_config) as sess:
        initializer = tf.random_uniform_initializer(-1.0 * config.init_w, config.init_w)
        scope = "model"
        with tf.variable_scope(scope, reuse=None, initializer=initializer):
            model = KgRnnCVAE(sess, config, api, log_dir=None if FLAGS.forward_only else log_dir, forward=False, scope=scope)
        with tf.variable_scope(scope, reuse=True, initializer=initializer):
            valid_model = KgRnnCVAE(sess, valid_config, api, log_dir=None, forward=False, scope=scope)
        with tf.variable_scope(scope, reuse=True, initializer=initializer):
            test_model = KgRnnCVAE(sess, test_config, api, log_dir=None, forward=True, scope=scope)

        test_model.prepare_mul_ref()

        logger.info("Created computation graphs")
        if api.word2vec is not None and not FLAGS.forward_only:
            logger.info("Loaded word2vec")
            sess.run(model.embedding.assign(np.array(api.word2vec)))

        # write config to a file for logging
        if not FLAGS.forward_only:
            with open(os.path.join(log_dir, "run.log"), "wb") as f:
                f.write(pp(config, output=False))

        # create a folder by force
        ckp_dir = os.path.join(log_dir, "checkpoints")
        if not os.path.exists(ckp_dir):
            os.mkdir(ckp_dir)

        ckpt = tf.train.get_checkpoint_state(ckp_dir)
        logger.info("Created models with fresh parameters.")
        sess.run(tf.global_variables_initializer())

        if ckpt:
            logger.info("Reading dm models parameters from %s" % ckpt.model_checkpoint_path)
            model.saver.restore(sess, ckpt.model_checkpoint_path)

        if not FLAGS.forward_only:
            dm_checkpoint_path = os.path.join(ckp_dir, model.__class__.__name__+ ".ckpt")
            global_t = 1
            patience = 10  # wait for at least 10 epoch before stop
            dev_loss_threshold = np.inf
            best_dev_loss = np.inf
            for epoch in range(config.max_epoch):
                logger.info(">> Epoch %d with lr %f" % (epoch, sess.run(model.learning_rate_cyc, {model.global_t: global_t})))

                # begin training
                if train_feed.num_batch is None or train_feed.ptr >= train_feed.num_batch:
                    train_feed.epoch_init(config.batch_size, config.backward_size,
                                          config.step_size, shuffle=True)
                global_t, train_loss = model.train(global_t, sess, train_feed, update_limit=config.update_limit)

                # begin validation
                logger.record_tabular("Epoch", epoch)
                logger.record_tabular("Mode", "Val")
                valid_feed.epoch_init(valid_config.batch_size, valid_config.backward_size,
                                      valid_config.step_size, shuffle=False, intra_shuffle=False)
                valid_loss = valid_model.valid("ELBO_VALID", sess, valid_feed)

                logger.record_tabular("Epoch", epoch)
                logger.record_tabular("Mode", "Test")
                test_feed.epoch_init(valid_config.batch_size, valid_config.backward_size,
                                  valid_config.step_size, shuffle=False, intra_shuffle=False)
                valid_model.valid("ELBO_TEST", sess, test_feed)

                # test_feed.epoch_init(test_config.batch_size, test_config.backward_size,
                #                      test_config.step_size, shuffle=True, intra_shuffle=False)
                # test_model.test_mul_ref(sess, test_feed, num_batch=5)

                done_epoch = epoch + 1
                # only save a models if the dev loss is smaller
                # Decrease learning rate if no improvement was seen over last 3 times.
                if config.op == "sgd" and done_epoch > config.lr_hold:
                    sess.run(model.learning_rate_decay_op)

                if valid_loss < best_dev_loss:
                    if valid_loss <= dev_loss_threshold * config.improve_threshold:
                        patience = max(patience, done_epoch * config.patient_increase)
                        dev_loss_threshold = valid_loss

                    # still save the best train model
                    if FLAGS.save_model:
                        logger.info("Save model!!")
                        model.saver.save(sess, dm_checkpoint_path)
                    best_dev_loss = valid_loss

                    if (epoch % 3) == 2:
                        tmp_model_path = os.path.join(ckp_dir, model.__class__.__name__+str(epoch)+".ckpt")
                        model.saver.save(sess, tmp_model_path)

                if config.early_stop and patience <= done_epoch:
                    logger.info("!!Early stop due to run out of patience!!")
                    break
            logger.info("Best validation loss %f" % best_dev_loss)
            logger.info("Done training")
        else:
            # begin validation
            # begin validation
            valid_feed.epoch_init(valid_config.batch_size, valid_config.backward_size,
                                  valid_config.step_size, shuffle=False, intra_shuffle=False)
            valid_model.valid("ELBO_VALID", sess, valid_feed)

            test_feed.epoch_init(valid_config.batch_size, valid_config.backward_size,
                                  valid_config.step_size, shuffle=False, intra_shuffle=False)
            valid_model.valid("ELBO_TEST", sess, test_feed)

            dest_f = open(os.path.join(log_dir, "test.txt"), "wb")
            test_feed.epoch_init(test_config.batch_size, test_config.backward_size,
                                 test_config.step_size, shuffle=False, intra_shuffle=False)
            test_model.test_mul_ref(sess, test_feed, num_batch=None, repeat=5, dest=dest_f)
            dest_f.close()

if __name__ == "__main__":
    log_dir = os.path.join(FLAGS.work_dir, FLAGS.sub_dir)
    if FLAGS.forward_only:        
        logger_dir = log_dir + "/testinfo"
    else:
        logger_dir =  log_dir +"/traininfo"

    with logger.session(dir=logger_dir, format_strs=['stdout', 'csv', 'log']):
        main()













