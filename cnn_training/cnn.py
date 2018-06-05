# python imports
import datetime
import json
import numpy as np
import os
from shutil import copyfile
import sys
import tensorflow as tf
import argparse
import tensorflow.contrib.metrics
import tensorflow.contrib.slim as slim

import cnn_architectures

from dataset import Dataset
import image_processing
import logging
'''
########################################################################################################################
FUNCTIONS
########################################################################################################################
'''


def parse_args():
    parser = argparse.ArgumentParser(description='Training a CNN classifier')

    parser.add_argument('-v', '--verbose', action='store_true', help='verbose output')
    parser.add_argument('-c', '--config', dest='config_path', type=str, required=True, help='path to the config file')
    args = parser.parse_args()
    return args


def build_train_graph(config, dataset):
    with tf.device('/cpu:0'):
        inputs, labels = image_processing.distorted_inputs(
            dataset,
            batch_size=config['parameters']['batch_size'],
            height=config['input']['height'],
            width=config['input']['width'],
            channels=config['input']['channels'],
            add_variations=config['parameters']['additional_variations'],
            num_preprocess_threads=8)

    with tf.device('/gpu:0'):
        logits, endpoints = cnn_architectures.create_model(
            config['model']['architecture'],
            inputs,
            is_training=True,
            num_classes=config['input']['classes'],
            reuse=None)

    if config['parameters']['loss'] == 'regression':
        labels = tf.cast(labels - config['parameters']['label_mean'], tf.float32)  # if needed,change to type int64
        mean_squared_error = tf.losses.mean_squared_error(labels=labels, predictions=logits)
        loss = tf.add_n([mean_squared_error] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name='total_loss')
        accuracy = tf.constant(0, shape=[], dtype=tf.float32)
    elif config['parameters']['loss'] == 'classification':
        labels = tf.cast(labels // 5, tf.int64)

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        loss = tf.add_n([cross_entropy_mean] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name='total_loss')

        correct_prediction = tf.equal(tf.argmax(logits, 1), labels)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf.summary.scalar('loss', loss, collections=['train'])
    tf.summary.scalar('accuracy', accuracy, collections=['train'])

    if config['output']['trainable_variables_to_summary']:
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var, collections=['train'])

    return loss, accuracy, tf.summary.merge_all(key='train')


def build_val_graph(config, dataset):
    with tf.device('/cpu:0'):
        inputs, labels = image_processing.inputs(
            dataset,
            batch_size=config['parameters']['batch_size'],
            height=config['input']['height'],
            width=config['input']['width'],
            channels=config['input']['channels'],
            num_preprocess_threads=8)

    with tf.device('/gpu:0'):
        logits, endpoints = cnn_architectures.create_model(
            config['model']['architecture'],
            inputs,
            is_training=False,
            num_classes=config['input']['classes'],
            reuse=True)

    if config['parameters']['loss'] == 'regression':
        labels = tf.cast(labels - config['parameters']['label_mean'], tf.float32)  # if needed,change to type int64
        mean_squared_error = tf.losses.mean_squared_error(labels=labels, predictions=logits)
        loss = tf.add_n([mean_squared_error] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name='total_loss')
        accuracy = tf.constant(0, shape=[], dtype=tf.float32)
    if config['parameters']['loss'] == 'classification':
        labels = tf.cast(labels // 5, tf.int64)

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        loss = tf.add_n([cross_entropy_mean] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name='total_loss')

        correct_prediction = tf.equal(tf.argmax(logits, 1), labels)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.name_scope('metrics'):
        m_loss, loss_update_op = tf.contrib.metrics.streaming_mean(loss, name='loss')
        m_accuracy, accuracy_update_op = tf.contrib.metrics.streaming_mean(accuracy, name='accuracy')

    stream_vars = [i for i in tf.local_variables() if 'metrics' in i.name]
    reset_op = [tf.variables_initializer(stream_vars)]

    tf.summary.scalar('loss', m_loss, collections=['validation'])
    tf.summary.scalar('accuracy', accuracy, collections=['validation'])

    if config['output']['trainable_variables_to_summary']:
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var, collections=['validation'])

    return m_loss, m_accuracy, tf.summary.merge_all(key='validation'), tf.group(loss_update_op,
                                                                                accuracy_update_op), reset_op


def train(config):
    train_dataset = Dataset(os.path.join(config['input']['path'], 'train', 'dataset.json'))
    val_dataset = Dataset(os.path.join(config['input']['path'], 'val', 'dataset.json'))

    with tf.Graph().as_default():
        with tf.name_scope('train') as scope:
            train_loss, train_accuracy, train_summary = build_train_graph(config, train_dataset)

        with tf.name_scope('val') as scope:
            val_loss, val_accuracy, val_summary, metric_update, metric_reset = build_val_graph(config, val_dataset)

        # load model parameter only
        if config['model']['mode'] == 'finetune':
            exclude = cnn_architectures.model_weight_excludes(config['model']['architecture'])
            variables_to_restore = slim.get_variables_to_restore(exclude=exclude)

            init_assign_op, init_feed_dict = slim.assign_from_checkpoint(config['model']['checkpoint'],
                                                                         variables_to_restore)
        # set learning rate
        global_step = tf.Variable(0, name='global_step')
        learning_rate = tf.train.exponential_decay(config['parameters']['base_lr'], global_step,
                                                   config['parameters']['step_size'], config['parameters']['gamma'])

        with tf.name_scope('train') as scope:
            tf.summary.scalar('learning_rate', learning_rate, collections=['train'])

        # minimize losses
        train_step = tf.train.MomentumOptimizer(
            learning_rate=learning_rate, momentum=config['parameters']['momentum']).minimize(
                train_loss, global_step=global_step)

        # batch operation
        batchnorm_updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if batchnorm_updates:
            batchnorm_updates = tf.group(*batchnorm_updates)

        # load all variables for continue_training
        if config['model']['mode'] == 'continue_training':
            variables_to_restore = slim.get_variables_to_restore()
            init_assign_op, init_feed_dict = slim.assign_from_checkpoint(config['model']['checkpoint'],
                                                                         variables_to_restore)

        # Create a saver
        saver = tf.train.Saver(max_to_keep=config['output']['keep_last_k_models'])

        # initialize
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            if config['model']['mode'] != 'scratch':
                sess.run(init_assign_op, init_feed_dict)

            # run summary writer
            summary_writer = tf.summary.FileWriter(config['output']['path'], sess.graph)

            # Start the queue runners.
            coord = tf.train.Coordinator()
            tf.train.start_queue_runners(sess=sess, coord=coord)
            logging.info('graph built')
            step_v = sess.run(global_step)

            while step_v < config['parameters']['iterations'] + 1:
                if step_v % config['parameters']['validation_iter'] == 0:
                    for x in range(val_dataset.num_images() // config['parameters']['batch_size'] + 1):
                        sess.run([metric_update])

                    loss_v, accuracy_v, summary_v, step_v = sess.run([val_loss, val_accuracy, val_summary, global_step])

                    logging.info('val: step {}, accuracy = {:.2f}, loss = {:.2f}'.format(step_v, accuracy_v, loss_v))
                    summary_writer.add_summary(summary_v, step_v)
                    sess.run(metric_reset)

                # train with summaries
                if step_v % config['output']['display_results'] == 0:
                    acc_v, _, loss_v, learn_v, _, step_v, summary_v = sess.run([
                        train_accuracy, train_step, train_loss, learning_rate, batchnorm_updates, global_step,
                        train_summary
                    ])

                    logging.info('train: step {}, accuracy = {:.2f}, loss = {:.2f}, lr = {:.5f}'.format(
                        step_v - 1, acc_v, loss_v, learn_v))

                    summary_writer.add_summary(summary_v, step_v - 1)
                else:
                    acc_v, _, loss_v, learn_v, _, step_v = sess.run(
                        [train_accuracy, train_step, train_loss, learning_rate, batchnorm_updates, global_step])

                # save the model
                if step_v % config['output']['save_iterations'] == 0:
                    checkpoint_path = os.path.join(config['output']['path'], 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step_v)
                    logging.info('Model saved in file: {}'.format(checkpoint_path))

            coord.request_stop()
            coord.join()


'''
########################################################################################################################
MAIN
########################################################################################################################
'''


def main(argv=None):
    # load arguments
    args = parse_args()

    # define logging level and format
    level = logging.ERROR
    if args.verbose:
        level = logging.DEBUG
    logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s ', datefmt='%d-%m-%Y %H:%M:%S', level=level)

    # load config
    with open(args.config_path) as config_file:
        config = json.load(config_file)

    # check config entries
    if config['model']['mode'] not in config['model']['mode_possibilities']:
        logging.error('Unknown model mode.')
        sys.exit(0)

    # setup output folder
    if tf.gfile.Exists(config['output']['path']) and config['model']['mode'] != 'continue_training':
        while True:
            sys.stdout.write('Output path already exists. Do you want to overwrite? [Y/n]: ')
            choice = input().lower()
            if choice == '' or choice == 'y':
                tf.gfile.DeleteRecursively(config['output']['path'])
                break
            elif choice == 'n':
                sys.exit(0)

    tf.gfile.MakeDirs(config['output']['path'])

    # save used config file to output folder
    ver = 0
    while True:
        if ver == 0:
            config_filename = 'config.json'
        else:
            config_filename = 'config_' + str(ver) + '.json'

        if tf.gfile.Exists(os.path.join(config['output']['path'], config_filename)):
            ver += 1
        else:
            copyfile(args.config_path, os.path.join(config['output']['path'], config_filename))
            break

    # start training
    train(config)


if __name__ == '__main__':
    tf.app.run()
