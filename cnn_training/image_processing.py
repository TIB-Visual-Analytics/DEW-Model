from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import scipy.signal as sp
import random
'''
########################################################################################################################
FUNCTIONS
########################################################################################################################
'''


def inputs(dataset, batch_size, height, width, channels, num_preprocess_threads=4):
    """
    Generate batches of ImageNet images for evaluation.

    Use this function as the inputs for evaluating a network.

    Note that some (minimal) image preprocessing occurs during evaluation
    including central cropping and resizing of the image to fit the network.

    Args:
        dataset: instance of Dataset class specifying the dataset.
        batch_size: integer, number of examples in batch
        num_preprocess_threads: integer, total number of preprocessing threads but
                                None defaults to FLAGS.num_preprocess_threads.

    Returns:
        images: Images. 4D tensor of size [batch_size, FLAGS.image_size, image_size, 3].
        labels: 1-D integer Tensor of [FLAGS.batch_size].
    """
    # Force all input processing onto CPU in order to reserve the GPU for
    # the forward inference and back-propagation.
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


def distorted_inputs(dataset, batch_size, height, width, channels, add_variations, num_preprocess_threads=4):
    """
    Generate batches of distorted versions of ImageNet images.

    Use this function as the inputs for training a network.

    Distorting images provides a useful technique for augmenting the data
    set during training in order to make the network invariant to aspects
    of the image that do not effect the label.

    Args:
        dataset: instance of Dataset class specifying the dataset.
        batch_size: integer, number of examples in batch
        num_preprocess_threads: integer, total number of preprocessing threads but
                                None defaults to FLAGS.num_preprocess_threads.

    Returns:
        images: Images. 4D tensor of size [batch_size, FLAGS.image_size, FLAGS.image_size, 3].
        labels: 1-D integer Tensor of [batch_size].
    """

    # Force all input processing onto CPU in order to reserve the GPU for
    # the forward inference and back-propagation.
    with tf.device('/cpu:0'):
        images, labels = batch_inputs(
            dataset,
            batch_size,
            height,
            width,
            channels,
            add_variations,
            train=True,
            num_preprocess_threads=num_preprocess_threads,
            num_readers=4)
    return images, labels


def batch_inputs(dataset,
                 batch_size,
                 height,
                 width,
                 channels,
                 add_variations,
                 train,
                 num_preprocess_threads=4,
                 num_readers=4):
    """
    Contruct batches of training or evaluation examples from the image dataset.

    Args:
        dataset: instance of Dataset class specifying the dataset.
        See dataset.py for details.
        batch_size: integer
        train: boolean
        num_preprocess_threads: integer, total number of preprocessing threads
        num_readers: integer, number of parallel readers

    Returns:
        images: 4-D float Tensor of a batch of images
        labels: 1-D integer Tensor of [batch_size].

    Raises:
        ValueError: if data is not found
    """
    with tf.name_scope('batch_processing'):
        data_files = dataset.data_files()
        if data_files is None:
            raise ValueError('No data files found for this dataset')

        # Create filename_queue
        if train:
            filename_queue = tf.train.string_input_producer(data_files, shuffle=True, capacity=16)
        else:
            filename_queue = tf.train.string_input_producer(data_files, shuffle=False, capacity=1)

        if num_preprocess_threads % 4:
            raise ValueError('Please make num_preprocess_threads a multiple '
                             'of 4 (%d % 4 != 0).', num_preprocess_threads)

        if num_readers is None:
            num_readers = 8

        if num_readers < 1:
            raise ValueError('Please make num_readers at least 1')

        # Approximate number of examples per shard.
        examples_per_shard = 1024
        # Size the random shuffle queue to balance between good global
        # mixing (more examples) and memory use (fewer examples).
        # 1 image uses 299*299*3*4 bytes = 1MB
        # The default input_queue_memory_factor is 16 implying a shuffling queue
        # size: examples_per_shard * 16 * 1MB = 17.6GB
        min_queue_examples = examples_per_shard * 16
        if train:
            examples_queue = tf.RandomShuffleQueue(
                capacity=min_queue_examples + 3 * batch_size, min_after_dequeue=min_queue_examples, dtypes=[tf.string])
        else:
            examples_queue = tf.FIFOQueue(capacity=examples_per_shard + 3 * batch_size, dtypes=[tf.string])

        # Create multiple readers to populate the queue of examples.
        if num_readers > 1:
            enqueue_ops = []
            for _ in range(num_readers):
                reader = tf.TFRecordReader()
                _, value = reader.read(filename_queue)
                enqueue_ops.append(examples_queue.enqueue([value]))

            tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(examples_queue, enqueue_ops))
            example_serialized = examples_queue.dequeue()
        else:
            reader = dataset.reader()
            _, example_serialized = reader.read(filename_queue)

        images_and_labels = []
        for thread_id in range(num_preprocess_threads):
            # Parse a serialized Example proto to extract the image and metadata.
            image_buffer, label_index, _ = parse_example_proto(example_serialized)

            image = image_preprocessing(image_buffer, height, width, channels, add_variations, train, thread_id)
            images_and_labels.append([image, label_index])

        images, label_index_batch = tf.train.batch_join(
            images_and_labels, batch_size=batch_size, capacity=2 * num_preprocess_threads * batch_size)

        images = tf.cast(images, tf.float32)
        images = tf.reshape(images, shape=[batch_size, height, width, channels])

        # Display the training images in the visualizer.
        tf.summary.image('images', images)

        return images, tf.reshape(label_index_batch, [batch_size])


def parse_example_proto(example_serialized):
    # Dense features in Example proto.
    feature_map = {
        'image/encoded': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
        'image/label/values': tf.FixedLenFeature([1], dtype=tf.int64),
        'image/label/names': tf.VarLenFeature(dtype=tf.string),
        'image/filename': tf.FixedLenFeature([], dtype=tf.string, default_value='')
    }
    features = tf.parse_single_example(example_serialized, feature_map)

    # Force the variable number of bounding boxes into the shape  [1,num_boxes,coords]
    return features['image/encoded'], tf.cast(
        features['image/label/values'], dtype=tf.int32), features['image/label/names']


def image_preprocessing(image_buffer, height, width, channels, add_variations, train, thread_id=0):
    image = decode_jpeg(image_buffer)
    if train:
        image = distort_image(image, height, width, add_variations, thread_id)
    else:
        image = eval_image(image, height, width, thread_id)

    if channels == 1:
        image = tf.image.rgb_to_grayscale(image)

    # Rescale to [-1,1] instead of [0, 1)
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    return image


def decode_jpeg(image_buffer, scope=None):
    """
    Decode a JPEG string into one 3-D float image Tensor.

    Args:
        image_buffer: scalar string Tensor.
        scope: Optional scope for op_scope.
    Returns:
        3-D float Tensor with values ranging from [0, 1).
    """
    with tf.name_scope(scope, 'decode_jpeg', [image_buffer]):
        # Decode the string as an RGB JPEG.
        # Note that the resulting image contains an unknown height and width
        # that is set dynamically by decode_jpeg. In other words, the height
        # and width of image is unknown at compile-time.
        image = tf.image.decode_jpeg(image_buffer, channels=3)

        # After this point, all image pixels reside in [0,1)
        # until the very end, when they're rescaled to (-1, 1).  The various
        # adjust_* ops all require this range for dtype float.
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        return image


def distort_image(image, height, width, add_variations, thread_id=0, scope=None):
    """
    Distort one image for training a network.

    Distorting images provides a useful technique for augmenting the data
    set during training in order to make the network invariant to aspects
    of the image that do not effect the label.

    Args:
        image: 3-D float Tensor of image
        height: integer
        width: integer
        thread_id: integer indicating the preprocessing thread.
        scope: Optional scope for op_scope.
        Returns:
        3-D float Tensor of distorted image used for training.
    """

    with tf.name_scope(scope, 'distort_image', [image, height, width]):
        if not thread_id:
            tf.summary.image('1_original_image', tf.expand_dims(image, 0), collections=['train'])

        sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
            tf.shape(image),
            bounding_boxes=tf.constant(0, shape=[1, 0, 4]),
            min_object_covered=0.7,
            aspect_ratio_range=[0.75, 1.33],
            area_range=[0.7, 1.0],
            max_attempts=100,
            use_image_if_no_bounding_boxes=True)
        bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box
        if not thread_id:
            image_with_distorted_box = tf.image.draw_bounding_boxes(tf.expand_dims(image, 0), distort_bbox)
            tf.summary.image('2_images_with_distorted_bounding_box', image_with_distorted_box, collections=['train'])

        # Crop the image to the specified bounding box.
        image = tf.slice(image, bbox_begin, bbox_size)

        # This resizing operation may distort the images because the aspect ratio is not respected.
        # We select a resize method in a round robin fashion based on the thread number.
        # Note that Resize Method contains 4 enumerated resizing methods.
        resize_method = thread_id % 4
        image = tf.image.resize_images(image, [height, width], method=resize_method)

        # Restore the shape since the dynamic slice based upon the bbox_size loses the third dimension.
        image.set_shape([height, width, 3])
        if not thread_id:
            tf.summary.image('3_cropped_resized_image', tf.expand_dims(image, 0), collections=['train'])

        # Randomly flip the image horizontally.
        image = tf.image.random_flip_left_right(image)

        if add_variations:
            distorted_image = image
            with tf.name_scope('distort_color'):
                prob_color = tf.random_uniform(shape=[], minval=0, maxval=2, dtype=tf.int32)
                distorted_image = tf.cond(
                    tf.equal(prob_color, 0), lambda: distort_color(image, thread_id), lambda: distorted_image)

            with tf.name_scope('distort_resolution'):
                prob_resolution = tf.random_uniform(shape=[], minval=0, maxval=2, dtype=tf.int32)
                distorted_image = tf.cond(
                    tf.equal(prob_resolution, 0), lambda: distort_resolution(image, height, width),
                    lambda: distorted_image)

            # TODO: What happens with rgb noise on grayscale images?
            with tf.name_scope('distort_noise'):
                prob_noise = tf.random_uniform(shape=[], minval=0, maxval=2, dtype=tf.int32)
                distorted_image = tf.cond(
                    tf.equal(prob_noise, 0), lambda: distort_noise(image, height, width), lambda: distorted_image)

            with tf.name_scope('distort_channels'):
                prob_channel = tf.random_uniform(shape=[], minval=0, maxval=2, dtype=tf.int32)
                distorted_image = tf.cond(
                    tf.equal(prob_channel, 0), lambda: distort_channels(image), lambda: distorted_image)

            if not thread_id:
                tf.summary.image('4_distorted_image', tf.expand_dims(distorted_image, 0), collections=['train'])

            prob_variation = tf.random_uniform(shape=[], minval=0, maxval=2, dtype=tf.int32)
            image = tf.cond(tf.equal(prob_noise, 0), lambda: distorted_image, lambda: image)

            if not thread_id:
                tf.summary.image('5_selected_image', tf.expand_dims(image, 0), collections=['train'])
    return image


def eval_image(image, height, width, thread_id=0, scope=None):
    """
    Prepare one image for evaluation.

    Args:
        image: 3-D float Tensor
        height: integer
        width: integer
        scope: Optional scope for op_scope.
    Returns:
        3-D float Tensor of prepared image.
    """
    with tf.name_scope(scope, 'eval_image', [image, height, width]):
        if not thread_id:
            tf.summary.image('1_original_image', tf.expand_dims(image, 0), collections=['validation'])

        im_height = tf.shape(image)[0]
        im_width = tf.shape(image)[1]

        def crop_hor(image):
            offset = (im_width - im_height) // 2
            return tf.image.crop_to_bounding_box(image, 0, offset, im_height, im_height)

        def crop_ver(image):
            offset = (im_height - im_width) // 2
            return tf.image.crop_to_bounding_box(image, offset, 0, im_width, im_width)

        image = tf.cond(tf.less(im_height, im_width), lambda: crop_hor(image), lambda: crop_ver(image))

        # Crop the central region of the image with an area containing 87.5% of the original image.
        # image = tf.image.central_crop(image, central_fraction=0.875)

        if not thread_id:
            tf.summary.image('2_cropped_image', tf.expand_dims(image, 0), collections=['validation'])

        # Resize the image to the original height and width.
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_bilinear(image, [height, width], align_corners=False)
        image = tf.squeeze(image, [0])

        if not thread_id:
            tf.summary.image('3_val_image', tf.expand_dims(image, 0), collections=['validation'])
        return image


'''
########################################################################################################################
OWN ADDITIONAL VARIATIONS
########################################################################################################################
'''


def distort_color(image, thread_id=0, scope=None):
    """Distort the color of the image.

    Each color distortion is non-commutative and thus ordering of the color ops
    matters. Ideally we would randomly permute the ordering of the color ops.
    Rather then adding that level of complication, we select a distinct ordering
    of color ops for each preprocessing thread.

    Args:
        image: Tensor containing single image.
        thread_id: preprocessing thread ID.
        scope: Optional scope for op_scope.
    Returns:
        color-distorted image
    """
    with tf.name_scope(scope, 'distort_color', [image]):
        color_ordering = thread_id % 2

        if color_ordering == 0:
            image = tf.image.random_brightness(image, max_delta=62. / 255.)
            image = tf.image.random_saturation(image, lower=0.75, upper=1.25)
            image = tf.image.random_hue(image, max_delta=0.1)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        elif color_ordering == 1:
            image = tf.image.random_brightness(image, max_delta=62. / 255.)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            image = tf.image.random_saturation(image, lower=0.75, upper=1.25)
            image = tf.image.random_hue(image, max_delta=0.1)

        # The random_* ops do not necessarily clamp.
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image


def distort_resolution(image, height, width):
    rand_scale = tf.random_uniform(shape=[], minval=0.33, maxval=0.67, dtype=tf.float32)
    image = tf.image.resize_images(image, [tf.to_int32(rand_scale * height), tf.to_int32(rand_scale * width)])
    image = tf.image.resize_images(image, [height, width])
    return image


def distort_noise(image, height, width):
    sigma = tf.random_uniform(shape=[], minval=0.01, maxval=0.05, dtype=tf.float32)
    noise = tf.random_normal(shape=[height, width, 3], stddev=sigma)
    image = tf.add(image, noise)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image


def distort_channels(image):
    image = tf.image.rgb_to_grayscale(image)
    image = tf.image.grayscale_to_rgb(image)
    return image
