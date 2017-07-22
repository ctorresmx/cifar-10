from dataset import CIFAR10Dataset
import tensorflow as tf
import fire


class CIFARClassifier:
    def __init__(self, epochs=10, batch_size=100):
        self.epochs = epochs
        self.batch_size = batch_size
        self.dataset = CIFAR10Dataset()

    def _conv_layer(self, input, weight_shape, bias_shape):
        weight_init = tf.random_normal_initializer(stddev=0.1)
        conv_weights = tf.get_variable('W', weight_shape,
                                       initializer=weight_init)

        bias_init = tf.constant_initializer(value=0.1)
        conv_biases = tf.get_variable('b', bias_shape, initializer=bias_init)

        conv = tf.nn.conv2d(input, conv_weights, strides=[1, 1, 1, 1],
                            padding='SAME')

        return tf.nn.relu(tf.nn.bias_add(conv, conv_biases))

    def _pool_layer(self, input):
        return tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                              padding='SAME')

    def _fc_layer(self, input, weight_shape, bias_shape):
        weight_init = tf.random_normal_initializer(stddev=0.1)
        fc_weights = tf.get_variable('W', weight_shape,
                                     initializer=weight_init)

        bias_init = tf.constant_initializer(value=0.1)
        fc_biases = tf.get_variable('b', bias_shape, initializer=bias_init)

        return tf.nn.relu(tf.matmul(input, fc_weights) + fc_biases)

    def _eval_in_batches(self, input, real_y, session):
        test_logits = self.model(input)
        correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(test_logits), 1),
                                      tf.argmax(real_y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        total_batch = int(self.dataset.test.num_examples / self.batch_size)

        network_accuracy_total = 0
        for i in range(total_batch):
            batch_x, batch_y = self.dataset.test.next_batch(self.batch_size)

            feed_dict = {
                input: batch_x,
                real_y: batch_y,
            }

            network_accuracy = session.run(accuracy, feed_dict=feed_dict)
            value = session.run(test_logits, feed_dict=feed_dict)
            print(value)
            network_accuracy_total += network_accuracy

        return network_accuracy_total / total_batch

    def predict(self, image):
        pass

    def model(self, input, train=False, dropout_rate=0.5):
        with tf.variable_scope('conv_1'):
            conv_layer_1 = self._conv_layer(input, [5, 5, 3, 32], [32])
            max_pool_layer_1 = self._pool_layer(conv_layer_1)

        with tf.variable_scope('conv_2'):
            conv_layer_2 = self._conv_layer(max_pool_layer_1, [5, 5, 32, 64],
                                            [64])
            max_pool_layer_2 = self._pool_layer(conv_layer_2)

        pool_shape = max_pool_layer_2.get_shape().as_list()

        with tf.variable_scope('fc_1'):
            reshape = tf.reshape(max_pool_layer_2, [-1, pool_shape[1] *
                                                    pool_shape[2] *
                                                    pool_shape[3]])
            hidden = self._fc_layer(reshape, [pool_shape[1] * pool_shape[2] *
                                              pool_shape[3], 512], [512])

        with tf.variable_scope('fc_2'):
            if train:
                hidden = tf.nn.dropout(hidden, dropout_rate)
            output = self._fc_layer(hidden, [512, 10], [10])

        return output

    def train(self):
        with tf.variable_scope('shared_variables') as scope:
            input = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
            real_y = tf.placeholder(tf.float32, shape=[None, 10])
            logits = self.model(input, train=True)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=real_y,
                                                                          logits=logits))

            optimizer = tf.train.AdamOptimizer().minimize(loss)

            with tf.Session() as session:
                tf.global_variables_initializer().run()

                for epoch in range(self.epochs):
                    total_batch = int(self.dataset.train.num_examples /
                                      self.batch_size)

                    for i in range(total_batch):
                        batch_x, batch_y = self.dataset.train.next_batch(self.batch_size)

                        feed_dict = {
                            input: batch_x,
                            real_y: batch_y,
                        }

                        session.run(optimizer, feed_dict=feed_dict)

                    print('Epoch {}'.format(epoch))

                scope.reuse_variables()
                network_accuracy = self._eval_in_batches(input, real_y,
                                                         session)

                print('The final accuracy over the CIFAR-10 data is \
                      {:2f}%'.format(network_accuracy * 100))


if __name__ == '__main__':
    fire.Fire(CIFARClassifier)
