from . import *
import tensorflow as tf
import numpy as np

slim = tf.contrib.slim

architectures_dict = {
    'scratchnet': {
        'model': scratchnet.scratchnet,
        'exclude': ['scratchnet/fc6'],
        'scope_name': 'scratchnet',
        'arg_scope': scratchnet.scratchnet_arg_scope
    },
    'vgg_16': {
        'model': vgg.vgg_16,
        'exclude': ['vgg_16/fc6', 'vgg_16/fc7', 'vgg_16/fc8'],
        'scope_name': 'vgg_16',
        'arg_scope': vgg.vgg_arg_scope
    },
    'vgg_19': {
        'model': vgg.vgg_19,
        'exclude': ['vgg_19/fc6', 'vgg_19/fc7', 'vgg_19/fc8'],
        'scope_name': 'vgg_19',
        'arg_scope': vgg.vgg_arg_scope
    },
    'inception_v4': {
        'model': inception_v4.inception_v4,
        'exclude': [],
        'scope_name': 'InceptionV4',
        'arg_scope': inception_utils.inception_arg_scope
    },
    'resnet_v1_50': {
        'model': resnet_v1.resnet_v1_50,
        'exclude': ['resnet_v1_50/logits'],
        'scope_name': 'resnet_v1_50',
        'arg_scope': resnet_v1.resnet_arg_scope
    },
    'resnet_v1_101': {
        'model': resnet_v1.resnet_v1_101,
        'exclude': ['resnet_v1_101/logits'],
        'scope_name': 'resnet_v1_101',
        'arg_scope': resnet_v1.resnet_arg_scope
    },
    'resnet_v1_152': {
        'model': resnet_v1.resnet_v1_152,
        'exclude': ['resnet_v1_152/logits'],
        'scope_name': 'resnet_v1_152',
        'arg_scope': resnet_v1.resnet_arg_scope
    },
    'inception_resnet_v2': {
        'model': inception_resnet_v2.inception_resnet_v2,
        'exclude': [],
        'scope_name': '',
        'arg_scope': inception_resnet_v2.inception_resnet_v2_arg_scope
    }
}


def create_model(architecture, inputs, num_classes, is_training=True, scope=None, reuse=None):
    if architecture not in architectures_dict:
        logging.error('CNN model not found')
        return None

    model_config = architectures_dict[architecture]

    model = model_config['model']
    arg_scope = model_config['arg_scope']
    #with tf.variable_scope(scope, 'vgg_16', [inputs]) as sc:
    if scope is None:
        scope = model_config['scope_name']
    with slim.arg_scope(arg_scope()):
        with tf.variable_scope(scope, default_name=model_config['scope_name'], values=[inputs], reuse=reuse) as sc:
            logits, endpoints = model(inputs, num_classes=num_classes, is_training=is_training, scope=sc)

    logits = tf.squeeze(logits)

    return logits, endpoints


def model_weight_excludes(architecture):
    if architecture not in architectures_dict:
        logging.error('CNN model not found')
        return []

    model_config = architectures_dict[architecture]

    return model_config['exclude']
