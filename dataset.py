import numpy as np
import tarfile
import os
import pickle
import urllib.request
import tensorflow as tf


class CIFAR10Dataset:

    SOURCE_URL = 'https://www.cs.toronto.edu/~kriz/'
    WORK_DIRECTORY = 'CIFAR-10-data'

    class Dataset:
        def __init__(self, data, labels):
            self._data = data
            self._labels = labels
            self._batch_size = 0
            self._batch_index = 0

        @property
        def num_examples(self):
            return len(self._data)

        def next_batch(self, batch_size):
            if self._batch_size != batch_size:
                self._batch_size = batch_size
                self._batch_index = 0

            start = self._batch_index * self._batch_size
            end = start + self._batch_size

            if start >= self.num_examples:
                start = 0
                end = start + self._batch_size
            elif end > self.num_examples:
                end = self.num_examples

            self._batch_index += 1

            return self._data[start:end], self._labels[start:end]

    def __init__(self):
        local_file = self.maybe_download('cifar-10-python.tar.gz')
        self.extract_data(local_file)

        train_data, train_labels = self.get_train_set()
        self.train = self.Dataset(train_data, train_labels)

        test_data, test_labels = self.get_test_set()
        self.test = self.Dataset(test_data, test_labels)

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
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar, path=self.WORK_DIRECTORY)

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
        labels = np.asarray(labels)
        labels = np.reshape(labels, (-1))
        one_hot_labels = np.zeros((labels.shape[0], labels.max() + 1))
        one_hot_labels[np.arange(labels.shape[0]), labels] = 1

        return data, one_hot_labels

    def get_test_set(self):
        data_batch = self.unpickle(
            self.WORK_DIRECTORY + '/cifar-10-batches-py/test_batch'
        )

        data = data_batch[b'data']
        labels = data_batch[b'labels']

        data = self.convert_into_2d(data)
        labels = np.asarray(labels)
        labels = np.reshape(labels, (-1))
        one_hot_labels = np.zeros((labels.shape[0], labels.max() + 1))
        one_hot_labels[np.arange(labels.shape[0]), labels] = 1

        return data, one_hot_labels

    def get_label_names(self):
        data = self.unpickle(
            self.WORK_DIRECTORY + '/cifar-10-batches-py/batches.meta'
        )

        # transform bytes to utf-8
        labels = [label.decode() for label in data[b'label_names']]

        return labels
