import numpy as np
import tarfile
import os
import pickle
import urllib.request
import tensorflow as tf


class CIFAR10Dataset:

    SOURCE_URL = 'https://www.cs.toronto.edu/~kriz/'
    WORK_DIRECTORY = 'CIFAR-10-data'

    def __init__(self):
        local_file = self.maybe_download('cifar-10-python.tar.gz')
        self.extract_data(local_file)

    def maybe_download(self, filename):
        if not tf.gfile.Exists(self.WORK_DIRECTORY):
            tf.gfile.MakeDirs(self.WORK_DIRECTORY)

        filepath = os.path.join(self.WORK_DIRECTORY, filename)

        if not tf.gfile.Exists(filepath):
            filepath, _ = urllib.request.urlretrieve(self.SOURCE_URL + filename,
                                                     filepath)
            with tf.gfile.GFile(filepath) as file:
                size = file.size()

            print('Successfully downloaded {} {} bytes.'.format(filename, size))

        return filepath

    def extract_data(self, file_path):
        with tarfile.open(file_path) as tar:
            tar.extractall(path=self.WORK_DIRECTORY)

    def unpickle(self, file_path):
        with open(file_path, 'rb') as file:
            data = pickle.load(file, encoding='bytes')

        return data

    def convert_into_2d(self, array):
        data_1 = np.reshape(array[:, :1024], [array.shape[0], 32, 32])
        data_2 = np.reshape(array[:, 1024:2048], [array.shape[0], 32, 32])
        data_3 = np.reshape(array[:, 2048:], [array.shape[0], 32, 32])

        data = np.stack((data_1, data_2, data_3), axis=3)

        return data

    def get_train_set(self):
        data = None
        labels = []

        for i in range(1, 6):
            data_batch = self.unpickle(
                self.WORK_DIRECTORY + '/cifar-10-batches-py/data_batch_{}'.format(i)
            )

            if data is None:
                data = data_batch[b'data']
            else:
                data = np.append(data, data_batch[b'data'], axis=0)

            labels += data_batch[b'labels']

        data = self.convert_into_2d(data)

        return data, labels

    def get_test_set(self):
        data_batch = self.unpickle(
            self.WORK_DIRECTORY + '/cifar-10-batches-py/test_batch'
        )

        data = data_batch[b'data']
        labels = data_batch[b'labels']
        data = self.convert_into_2d(data)

        return data, labels

    def get_label_names(self):
        data = self.unpickle(
            self.WORK_DIRECTORY + '/cifar-10-batches-py/batches.meta'
        )

        # transform bytes to utf-8
        labels = [label.decode() for label in data[b'label_names']]

        return labels


dataset = CIFAR10Dataset()

x, y = dataset.get_train_set()

IMAGE_ID = 22

x_test, y_test = dataset.get_test_set()

label_names = dataset.get_label_names()

from PIL import Image

im = Image.fromarray(x[IMAGE_ID])
im.save('train.png')
print(label_names[y[IMAGE_ID]])

im = Image.fromarray(x_test[IMAGE_ID])
im.save('test.png')
print(label_names[y_test[IMAGE_ID]])
