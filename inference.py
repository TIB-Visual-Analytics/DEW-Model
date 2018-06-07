import argparse
import csv
import json
import math
import numpy as np
import os
import sys
import tensorflow as tf
import tensorflow.contrib.slim as slim

sys.path.insert(1, 'CNN_Training')
import cnn_architectures
from dataset import Dataset
'''
########################################################################################################################
FUNCTIONS
########################################################################################################################
'''


def parse_args():
    parser = argparse.ArgumentParser(description='Date Estimation in the Wild Evaluation')
    parser.add_argument('-p', '--path', dest='path', type=str, required=True, help='Path to DEW')
    parser.add_argument('-m', '--model', dest='model', type=str, required=True, help='Path to the trained model')
    parser.add_argument(
        '-e', '--endpoints', dest='show_endpoints', type=int, default=0, help='Show Endpoints of the model')
    args = parser.parse_args()
    return args


def init_cnn(sess, args, config, images_placeholder):
    with tf.device('/gpu:0'):
        logits, _ = cnn_architectures.create_model(
            config['model']['architecture'],
            images_placeholder,
            is_training=False,
            num_classes=config['input']['classes'],
            reuse=None)

    saver = tf.train.Saver()
    saver.restore(sess, args.model)
    print('---------------------------')
    print('Restore model from: {}'.format(args.model))

    return tf.nn.softmax(tf.squeeze(logits))


def img_preprocess(img_encode, config):
    # decode the image
    img = tf.image.decode_jpeg(img_encode)
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)

    # normalize image
    # get correct amount of channel
    channels = tf.shape(img)[2]
    if config['input']['channels'] == 1:
        img = tf.cond(tf.equal(channels, 3), lambda: tf.image.rgb_to_grayscale(img), lambda: img)
    if config['input']['channels'] == 3:
        img = tf.cond(tf.equal(channels, 1), lambda: tf.image.grayscale_to_rgb(img), lambda: img)

    # get multicrops depending on the image orientation
    height = tf.to_float(tf.shape(img)[0])
    width = tf.to_float(tf.shape(img)[1])

    def crop_ver_image(img, config):
        # top, center, bottom
        ratio = config['input']['width'] / width
        height_new = tf.to_int32(height * ratio)
        offset = (height_new - config['input']['height']) // 2

        img = tf.expand_dims(img, 0)
        img = tf.squeeze(tf.image.resize_bilinear(img, size=[height_new, config['input']['width']]))
        img_array = []
        for i in range(3):
            #img_array.append(
            #    tf.expand_dims(
            #        tf.image.crop_to_bounding_box(img, i * offset, 0, config['input']['height'], config['input'][
            #            'width']), 0))
            img_crop = tf.image.crop_to_bounding_box(img, i * offset, 0, config['input']['height'],
                                                     config['input']['width'])
            img_crop = tf.expand_dims(img_crop, 0)
            img_array.append(img_crop)

        return tf.concat(img_array, axis=0)

    def crop_hor_image(img, config):
        # left, center, right
        ratio = 1.0 * config['input']['height'] / height
        width_new = tf.to_int32(width * ratio)
        offset = (width_new - config['input']['width']) // 2

        img = tf.expand_dims(img, 0)
        img = tf.squeeze(tf.image.resize_bilinear(img, size=[config['input']['height'], width_new]))
        img_array = []
        for i in range(3):
            img_crop = tf.image.crop_to_bounding_box(img, 0, i * offset, config['input']['height'],
                                                     config['input']['width'])
            img_crop = tf.expand_dims(img_crop, 0)
            img_array.append(img_crop)

        return tf.concat(img_array, axis=0)

    imgs = tf.cond(tf.less(width, height), lambda: crop_ver_image(img, config), lambda: crop_hor_image(img, config))
    # Rescale to [-1,1] instead of [0, 1)
    imgs = tf.subtract(imgs, 0.5)
    imgs = tf.multiply(imgs, 2.0)
    return imgs


def load_image_batch(sess, args, img_preprocess, img_encode, config, image_path):
    with tf.gfile.FastGFile( image_path, 'rb') as f:
        img = f.read()

    return sess.run(img_preprocess, feed_dict={img_encode: img})  # convert tensor to np.array


def get_predictions(sess, args, config, image_placeholder, endpoints, img_paths):
    print('Get predictions of DEW images ... ')

    img_encode = tf.placeholder(dtype=tf.string)
    img_pre = img_preprocess(img_encode, config)

    # feed forward batch of images in cnn and extract result
    predictions = []
    for i in range(len(img_paths)):
        imgs = load_image_batch(sess, args, img_pre, img_encode, config, img_paths[i])
        x = sess.run(endpoints, feed_dict={image_placeholder: imgs})

        # get average date predictions
        avg_prediction = 0

        if config['parameters']['loss'] == 'regression':
            for j in range(len(imgs)):
                avg_prediction += 1930 + int(x[j] + 0.5)
        elif config['parameters']['loss'] == 'classification':
            for j in range(len(imgs)):
                sum_prob = 0
                # print(x[j, :])
                for k in range(config['input']['classes']):
                    sum_prob += k * x[j, k]

                avg_prediction += 1930 + int(0.5 + sum_prob * (1999 - 1930) / (config['input']['classes'] - 1))

        predictions.append(int(0.5 + avg_prediction / 3))

    return predictions


def main():
    args = parse_args()

    # load config
    with open(os.path.join(os.path.dirname(args.model), 'config.json')) as config_file:
        config = json.load(config_file)


    # init tf session and get predictions
    with tf.Session() as sess:
        #sess.run(tf.global_variables_initializer())
        image_placeholder = tf.placeholder(
            tf.float32, shape=(3, config['input']['height'], config['input']['width'], config['input']['channels']))
        endpoints = init_cnn(sess, args, config, image_placeholder)
        predictions = get_predictions(sess, args, config, image_placeholder, endpoints, [args.path])
        print('Predicted year: {}'.format(predictions[0]))


if __name__ == '__main__':
    main()
