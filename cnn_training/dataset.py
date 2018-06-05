import os
import os.path
import re

import json

import logging


class Dataset():

    def __init__(self, path):
        self._path = path
        try:
            with open(path) as config_file:
                self._config = json.load(config_file)
        except ValueError as e:
            logging.error('Dataset file not found')

        self._num_classes = self._config['num_classes']
        self._num_images = self._config['num_images']
        self._multi_label = self._config.get('multi_label', False)
        self._dir_path = self._config.get('dir_path', None)
        self._file_list = self._config.get('file_list', [])

    def num_images(self):
        return self._num_images

    def multi_label(self):
        return self._multi_label

    def num_classes(self):
        return self._num_classes

    def data_files(self):
        dir_path = os.path.dirname(self._path)
        if not os.path.isabs(dir_path):
            dir_path = os.path.join(dir_path, self._dir_path)

        file_re = re.compile(r'\d+_\d+\.rec')

        result = []
        for f in os.listdir(dir_path):
            if re.match(file_re, f):
                result.append(os.path.join(dir_path, f))
        return result
