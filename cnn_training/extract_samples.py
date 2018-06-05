import tensorflow as tf
import numpy as np
import scipy.signal as sp
import random
from dataset import Dataset

import logging
import argparse
'''
########################################################################################################################
FUNCTIONS
########################################################################################################################
'''


def parse_args():
    parser = argparse.ArgumentParser(description='Training a CNN classifier')

    parser.add_argument('-v', '--verbose', action='store_true', help='verbose output')
    parser.add_argument('-c', '--config', dest='config_path', type=str, required=True, help='path to the config file')
    parser.add_argument('-o', '--output', dest='output_path', type=str, required=True, help='path to write the results')
    args = parser.parse_args()
    return args


def inputs(dataset, batch_size, height, width, channels, num_preprocess_threads=4):

    with tf.device('/cpu:0'):
        images, labels = batch_inputs(
            dataset,
            batch_size,
            height,
            width,
            channels,
            add_variations=False,
            train=False,
            num_preprocess_threads=num_preprocess_threads,
            num_readers=4)

    return images, labels


def parse_example_proto(example_serialized):

    # return tf.train.Example(features=tf.train.Features(feature={
    #     'image/width':
    #     tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
    #     'image/height':
    #     tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
    #     'image/label/values':
    #     tf.train.Feature(int64_list=tf.train.Int64List(value=label_values)),
    #     'image/label/names':
    #     tf.train.Feature(bytes_list=tf.train.BytesList(value=label_names)),
    #     'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(
    #         value=[filename.encode('utf-8')])),
    #     'image/encoded':
    #     tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw]))
    # }))

    # Dense features in Example proto.
    feature_map = {
        'image/encoded': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
        'image/label/values': tf.FixedLenFeature([1], dtype=tf.int64),
        'image/label/names': tf.VarLenFeature(dtype=tf.string),
        'image/filename': tf.FixedLenFeature([], dtype=tf.string, default_value='')
    }
    sparse_float32 = tf.VarLenFeature(dtype=tf.float32)
    # Sparse features in Example proto
    feature_map.update(
        {k: sparse_float32
         for k in ['image/bbox/xmin', 'image/bbox/ymin', 'image/bbox/width', 'image/bbox/height']})

    features = tf.parse_single_example(example_serialized, feature_map)

    image = tf.image.decode_jpeg(features['image/encoded'])

    im_height = tf.shape(image)[0]
    im_width = tf.shape(image)[1]

    xmin = tf.expand_dims(features['image/bbox/xmin'].values, 0)
    ymin = tf.expand_dims(features['image/bbox/ymin'].values, 0)
    width = tf.expand_dims(features['image/bbox/width'].values, 0)
    height = tf.expand_dims(features['image/bbox/height'].values, 0)
    #xmax = xmin + width
    #ymax = ymin + height

    # change value to [0,1)
    #xmin = tf.div(tf.to_float(xmin), tf.to_float(im_width))
    #ymin = tf.div(tf.to_float(ymin), tf.to_float(im_height))
    #xmax = tf.div(tf.to_float(xmax), tf.to_float(im_width))
    #ymax = tf.div(tf.to_float(ymax), tf.to_float(im_height))

    # Note that we impose an ordering of (y,x) just to make life difficult
    # WTF?
    bbox = tf.concat([ymin, xmin, height, width], 0)

    # Force the variable number of bounding boxes into the shape
    #[1,num_boxes,coords]
    bbox = tf.expand_dims(bbox, 0)
    bbox = tf.transpose(bbox, [0, 2, 1])
    return features['image/encoded'], tf.cast(
        features['image/label/values'], dtype=tf.int32), features['image/label/names'].values, bbox


def read_records(session, dataset, output_path):
    with tf.name_scope('extracting'):
        data_files = dataset.data_files()
        if data_files is None:
            raise ValueError('No data files found for this dataset')

        filename_queue = tf.train.string_input_producer(data_files, num_epochs=1, shuffle=False, capacity=1)

        reader = tf.TFRecordReader()
        _, example_serialized = reader.read(filename_queue)

        image_buffer, label_value, label_name, bbox = parse_example_proto(example_serialized)
        filename = tf.string_join([output_path, tf.as_string(label_value), '_', label_name, '.jpg'])
        images, filenames = tf.train.batch_join([[image_buffer, filename]], batch_size=1, capacity=100)

    # Start the queue runners.

    print(tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES))
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    session.run(init_op)
    coord = tf.train.Coordinator()
    tf.train.start_queue_runners(sess=session, coord=coord)

    for x in range(2):
        print(x)
        try:
            while True:
                # print(session.run([filename_queue]))
                filename_v, image_buffer_v = session.run([filenames, images])
                print(filename_v)
                file = tf.gfile.FastGFile(filename_v[0, 0].decode('utf-8'), mode='wb')
                file.write(image_buffer_v[0])

        except tf.errors.OutOfRangeError:
            session.run(tf.local_variables_initializer())

    coord.request_stop()
    coord.join()


def main(argv=None):
    # load arguments
    args = parse_args()

    dataset = Dataset(args.config_path)

    with tf.Session() as sess:
        read_records(sess, dataset, output_path=args.output_path)


if __name__ == '__main__':
    tf.app.run()
