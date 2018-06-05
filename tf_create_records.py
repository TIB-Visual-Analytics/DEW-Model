import csv
import os
import sys
import argparse
import logging
import re
import json

import numpy as np
import tensorflow as tf
from multiprocessing import Process, Manager


def parse_args():
    parser = argparse.ArgumentParser(description='Convert folder or file with annotation files to tf_records.')

    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help='verbose terminal output')
    parser.add_argument('-p', '--path', dest='path', required=True, type=str, help='path to images')
    parser.add_argument('-i', '--input', dest='input', required=True, type=str, help='input path or file')
    parser.add_argument('-o', '--output', dest='output', required=True, type=str, help='output path')
    parser.add_argument('-e', '--exclude_files', dest='exclude_file', type=str, help='files which should be ignored')
    parser.add_argument('-s', '--shards', dest='shards', type=int, default=128, help='count of shards')

    args = parser.parse_args()
    return args


def create_example(image_raw, label_values, label_names, filename):
    # convert to bytes
    label_names = [x.encode('utf-8') for x in label_names]

    return tf.train.Example(
        features=tf.train.Features(
            feature={
                'image/label/values': tf.train.Feature(int64_list=tf.train.Int64List(value=label_values)),
                'image/label/names': tf.train.Feature(bytes_list=tf.train.BytesList(value=label_names)),
                'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename.encode('utf-8')])),
                'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw])),
            }))


def convert_data_batch(d, s, image_label_list, label_id_map, out_path, ranges, thread_id, shards):

    session = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.01)))

    jpeg_data = tf.placeholder(dtype=tf.string)
    jpeg_decode = tf.image.decode_jpeg(jpeg_data, channels=3)

    num_shards_per_batch = shards // len(ranges)
    shard_ranges = np.linspace(ranges[thread_id][0], ranges[thread_id][1], num_shards_per_batch + 1).astype(int)

    for x in range(num_shards_per_batch):
        shard = thread_id * num_shards_per_batch + x
        record_path = os.path.join(out_path, '{}_{}.rec'.format(shard, shards - 1))
        with tf.python_io.TFRecordWriter(record_path) as writer:
            for i in range(shard_ranges[x], shard_ranges[x + 1]):
                image_name = image_label_list[i][0]
                image_path = image_label_list[i][0]
                if image_path is None:
                    logging.warning('{} not found in {}'.format(image_name, image_path))
                    continue
                # read file
                with tf.gfile.FastGFile(image_path, 'r') as f:
                    image_data = f.read()

                try:
                    var = session.run(jpeg_decode, {jpeg_data: image_data})
                except tf.python.errors.InvalidArgumentError:
                    logging.warning('{} is no valid jpeg'.format(image_name))
                    continue

                # write line
                label_names = image_label_list[i][1]
                label_values = [label_id_map[x] for x in label_names]

                writer.write(create_example(image_data, label_values, label_names, image_name).SerializeToString())
                # logging.info('write {} to record'.format(image_name))

                s.acquire()
                d['sample_count'] += 1
                s.release()


def convert_data(image_label_list, label_id_map, out_path, threads=8, shards=1024):
    # create output path
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # create_ranges
    spacing = np.linspace(0, len(image_label_list), threads + 1).astype(np.int)
    ranges = []
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])

    process_list = []
    manager = Manager()
    d = manager.dict()
    s = manager.Semaphore()
    d['sample_count'] = 0

    for x in range(threads):
        process = Process(
            target=convert_data_batch, args=[d, s, image_label_list, label_id_map, out_path, ranges, x, shards])
        process.start()
        process_list.append(process)

    for x in process_list:
        x.join()

    logging.info('write finished')
    return d['sample_count']


def convert_file(base_path, in_path, out_path, exclude_file, shards=128):
    logging.info('start conversion ... ')

    # get excluded image list
    logging.info('Get list of excluded images ... ')
    exclude_images = []
    if exclude_file is not None and os.path.isfile(exclude_file):
        with open(exclude_file, 'r') as f:
            content = csv.reader(f, delimiter=',')
            for line in content:
                if len(line) < 1:
                    continue
                exclude_images.append(os.path.join(line[0], line[1] + '.jpg'))

    # get minimum label_value
    first_loop = True
    logging.info('Get minimum label value ... ')
    with open(in_path, 'r') as f:
        content = csv.reader(f, delimiter=',')
        for line in content:
            if len(line) < 1:
                continue

            label = int(line[0])

            if first_loop:
                min_label = label
                first_loop = False

            if label < min_label:
                min_label = label

    logging.info('Minimum label: {}'.format(min_label))

    # get image paths and maximum label_value
    images_with_labels = {}
    class_id_map = {}
    max_label = 0
    cnt_excluded_images = 0
    logging.info('Get image list and labels ... ')
    with open(in_path, 'r') as f:
        content = csv.reader(f, delimiter=',')
        for line in content:
            label_name = line[0]
            label_value = int(line[0]) - min_label

            if label_value > max_label:
                max_label = label_value

            img_name = os.path.join(line[0], line[1] + '.jpg')
            if img_name in exclude_images:
                cnt_excluded_images += 1
                continue

            img_name = os.path.join(base_path, img_name)

            images_with_labels[img_name] = [label_name]
            class_id_map[label_name] = label_value

    logging.info('{} files excluded.'.format(cnt_excluded_images))

    # convert variables to a list
    image_label_list = [(k, v) for (k, v) in images_with_labels.items()]
    # convert data to tf records
    cnt_images = convert_data(image_label_list, class_id_map, out_path, threads=8, shards=shards)

    # write json file with informations about the dataset
    data_for_json = {
        'num_images': cnt_images,
        'num_classes': max_label + 1,
        'min_label': min_label,
        'multi_label': False,
        'dir_path': '.',
        'file_list': []
    }

    with open(out_path + '/' + 'dataset.json', 'w') as f:
        json.dump(data_for_json, f, sort_keys=False, indent=2)


def main():
    args = parse_args()
    level = logging.ERROR
    if args.verbose:
        level = logging.DEBUG

    logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', datefmt='%d-%m-%Y %H:%M:%S', level=level)

    if os.path.isfile(args.input):
        convert_file(args.path, args.input, args.output, args.exclude_file, args.shards)
    else:
        logging.error('Invalid input file or dir')
        return -1

    return 0


if __name__ == '__main__':
    sys.exit(main())
