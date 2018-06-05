import tensorflow as tf

slim = tf.contrib.slim


def scratchnet_arg_scope(weight_decay=0.0005):
    """Defines the ScratchNet arg scope.

  Args:
    weight_decay: The l2 regularization coefficient.

  Returns:
    An arg_scope.
  """
    with slim.arg_scope(
        [slim.fully_connected],
            activation_fn=None,
            weights_regularizer=slim.l2_regularizer(weight_decay),
            biases_initializer=tf.constant_initializer(0.1)):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d], padding='SAME') as arg_sc:
            return arg_sc


def scratchnet(inputs, num_classes=1000, is_training=True, dropout_keep_prob=0.6, scope='scratchnet'):
    """Build the model
    Args:

    Returns:
        Logits.
    """
    with tf.variable_scope(scope, 'scratchnet', [inputs]) as sc:
        end_points_collection = sc.name + '_end_points'
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], outputs_collections=end_points_collection):
            net = slim.conv2d(inputs, 32, 3, scope='conv11')
            net = slim.conv2d(net, 64, 3, scope='conv12')
            net = slim.max_pool2d(net, 3, scope='pool1')
            net = slim.conv2d(net, 64, 3, scope='conv21')
            net = slim.conv2d(net, 128, 3, scope='conv22')
            net = slim.max_pool2d(net, 3, scope='pool2')
            net = slim.conv2d(net, 96, 3, scope='conv31')
            net = slim.conv2d(net, 192, 3, scope='conv32')
            net = slim.max_pool2d(net, 3, scope='pool3')
            net = slim.conv2d(net, 128, 3, scope='conv41')
            net = slim.conv2d(net, 256, 3, scope='conv42')
            net = slim.max_pool2d(net, 3, scope='pool4')
            net = slim.conv2d(net, 160, 3, scope='conv51')
            net = slim.conv2d(net, 320, 3, scope='conv52')
            net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)
            tf.add_to_collection(end_points_collection, net)
            # net = slim.avg_pool2d(net, 7, padding='VALID', scope='pool5')
            net = tf.squeeze(net)
            net = slim.dropout(net, keep_prob=0.6, is_training=is_training, scope='dropout')
            net = slim.fully_connected(net, num_classes, scope='fc6')

            end_points = slim.utils.convert_collection_to_dict(end_points_collection)

            return net, end_points


scratchnet.default_image_size = 100
