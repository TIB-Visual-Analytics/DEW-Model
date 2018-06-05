# python imports
import datetime
import json
import numpy as np
import os
from shutil import copyfile
import sys
import tensorflow as tf
import argparse
import tensorflow.contrib.slim as slim

# own imports
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


#
# def loss(logits, labels):
#     '''Args:
#     logits: Logits from inference().
#     labels: Labels of the input image pf batch size
#
#     Returns:
#     Loss tensor of type float.'''
#
#     # first the average cross entropy loss across the batch
#     labels = tf.cast(labels, tf.int64)  # if needed,change to type int64
#     cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
#     cross_entropy_mean = tf.reduce_mean(cross_entropy)
#     #tf.add_to_collection(tf.GraphKeys.LOSSES, cross_entropy_mean)
#     # The total loss is defined as the cross entropy loss plus all the weight
#     # decay terms (L2 loss).
#     return tf.add_n([cross_entropy_mean] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name='total_loss')
#


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

    labels = tf.cast(labels, tf.int64)  # if needed,change to type int64
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    loss = tf.add_n([cross_entropy_mean] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name='total_loss')

    correct_prediction = tf.equal(tf.argmax(logits, 1), labels)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf.summary.scalar('train/accuracy', accuracy, collections=['train'])
    tf.summary.scalar('train/loss', loss, collections=['train'])

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

    labels = tf.cast(labels, tf.int64)  # if needed,change to type int64
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    loss = tf.add_n([cross_entropy_mean] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name='total_loss')

    correct_prediction = tf.equal(tf.argmax(logits, 1), labels)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf.summary.scalar('val/accuracy', accuracy, collections=['validation'])
    tf.summary.scalar('val/loss', loss, collections=['validation'])

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var, collections=['validation'])

    return loss, accuracy, tf.summary.merge_all(key='validation')


def val(config):

    val_dataset = Dataset(os.path.join(config['input']['path']))

    with tf.Graph().as_default():

        with tf.name_scope('val') as scope:
            val_loss, val_accuracy, val_summary = build_val_graph(config, val_dataset)

        exclude = cnn_architectures.model_weight_excludes(config['model']['architecture'])
        variables_to_restore = slim.get_variables_to_restore()
        init_assign_op, init_feed_dict = slim.assign_from_checkpoint(config['model']['checkpoint'],
                                                                     variables_to_restore)

        # initialize
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            sess.run(init_assign_op, init_feed_dict)

            # Start the queue runners.
            coord = tf.train.Coordinator()
            tf.train.start_queue_runners(sess=sess, coord=coord)
            print('graph built')

            com_acc = 0.0
            com_loss = 0.0
            count = 0
            for x in range(val_dataset.num_images() // config['parameters']['batch_size'] + 1):
                acc_v, loss_v = sess.run([val_accuracy, val_loss])
                com_acc += acc_v
                com_loss += loss_v
                count += 1
            print('validation loss: {} validation_accuracy: {}'.format(com_loss / count, com_acc / count))

            logging.info('accuracy = {}, loss = {}'.format(acc_v, loss_v))


'''
########################################################################################################################
MAIN
########################################################################################################################
'''


def main(argv=None):
    # load arguments
    args = parse_args()

    # load config
    with open(args.config_path) as config_file:
        config = json.load(config_file)

    # start training
    val(config)


if __name__ == '__main__':
    tf.app.run()
