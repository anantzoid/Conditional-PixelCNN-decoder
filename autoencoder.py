import tensorflow as tf
import numpy as np
from utils import *
import matplotlib.pyplot as plt
from models import PixelCNN

class AE(object):
    def __init__(self, X):
        self.fmap_out = 32
        self.fmap_in = conf.channel
        self.fan_in = X
        self.num_layers = 3
        self.filter_size = 3
        self.W = []
        self.strides = [1, 1, 1, 1]

        for i in range(self.num_layers):
            if i == self.num_layers -1 :
                self.fmap_out = 10 
            self.W.append(tf.Variable(tf.truncated_normal(shape=[self.filter_size, self.filter_size, self.fmap_in, self.fmap_out], stddev=0.1), name="W_%d"%i))
            b = tf.Variable(tf.ones(shape=[self.fmap_out], dtype=tf.float32), name="encoder_b_%d"%i)
            en_conv = tf.nn.conv2d(self.fan_in, self.W[i], self.strides, padding='SAME', name="encoder_conv_%d"%i)

            self.fan_in = tf.tanh(tf.add(en_conv, b))
            self.fmap_in = self.fmap_out

        self.fan_in = tf.reshape(self.fan_in, (-1, conf.img_width*conf.img_height*self.fmap_out))

    def decoder(self):
        self.W.reverse()
        for i in range(self.num_layers):
            if i == self.num_layers-1:
                self.fmap_out = conf.channel
            c = tf.Variable(tf.ones(shape=[self.fmap_out], dtype=tf.float32), name="decoder_b_%d"%i)
            de_conv = tf.nn.conv2d_transpose(self.fan_in, self.W[i], [tf.shape(X)[0], conf.img_height, conf.img_width, self.fmap_out], self.strides, padding='SAME', name="decoder_conv_%d"%i)
            self.fan_in = tf.tanh(tf.add(de_conv, c))
        self.y = self.fan_in    

    def generate(self, conf):
        n_examples = 10
        if conf.data == 'mnist':
            test_X_pure = data.train.next_batch(n_examples)[0].reshape(n_examples, conf.img_height, conf.img_width, conf.channel)
            test_X = binarize(test_X_pure)

        condition = sess.run(fan_in, feed_dict={X:test_X})
        samples = sess.run(y, feed_dict={X: test_X, decoder.h:condition})
        fig, axs = plt.subplots(2, n_examples, figsize=(n_examples, 2))
        for i in range(n_examples):
            axs[0][i].imshow(np.reshape(o_test_X[i], (conf.img_height, conf.img_width)))
            axs[1][i].imshow(np.reshape(samples[i], (conf.img_height, conf.img_width)))
        fig.show()
        plt.draw()
        plt.waitforbuttonpress()



def trainPixelCNNAE(conf, data):
    encoder_X = tf.placeholder(tf.float32, shape=[None, conf.img_height, conf.img_width, conf.channel])
    decoder_X = tf.placeholder(tf.float32, shape=[None, conf.img_height, conf.img_width, conf.channel])

    encoder = AE(encoder_X)
    conf.num_classes = int(encoder.fan_in.get_shape()[1])
    decoder = PixelCNN(decoder_X, conf, encoder.fan_in)
    y = decoder.pred

    loss = tf.reduce_mean(tf.square(encoder_X - y))
    trainer = tf.train.GradientDescentOptimizer(1e-4).minimize(loss)

    saver = tf.train.Saver(tf.trainable_variables())
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        if os.path.exists(conf.ckpt_file):
            saver.restore(sess, conf.ckpt_file)
            print "Model Restored"
        
        pointer = 0
        for i in range(conf.epochs):
            for j in range(conf.num_batches):
                if conf.data == 'mnist':
                    batch_X = binarize(data.train.next_batch(conf.batch_size)[0].reshape(conf.batch_size, conf.img_height, conf.img_width, conf.channel))
                else:
                    batch_X, pointer = get_batch(data, pointer, conf.batch_size)

                _, l = sess.run([trainer, loss], feed_dict={encoder_X: batch_X, decoder_X: batch_X})
            print "Epoch: %d, Cost: %f"%(i, l)

        saver.save(sess, conf.ckpt_file)
        generate_ae(sess, encoder_X, decoder_X, y, data, conf)


if __name__ == "__main__":
    class Conf(object):
        pass

    conf = Conf()
    conf.conditional=True
    conf.data='mnist'
    conf.data_path='data'
    conf.f_map=32
    conf.grad_clip=1
    conf.layers=5
    conf.samples_path='samples/ae'
    conf.ckpt_path='ckpts/ae'
    conf.epochs=10
    conf.batch_size = 100

    if conf.data == 'mnist':
        from tensorflow.examples.tutorials.mnist import input_data
        data = input_data.read_data_sets("data/")
        conf.img_height = 28
        conf.img_width = 28
        conf.channel = 1
        train_size = data.train.num_examples
    else:
        from keras.datasets import cifar10
        data = cifar10.load_data()
        # TODO normalize pixel values
        data = data[0][0]
        data = np.transpose(data, (0, 2, 3, 1))
        conf.img_height = 32
        conf.img_width = 32
        conf.channel = 3
        train_size = data.shape[0] 

    conf.num_batches = train_size // conf.batch_size
    conf = makepaths(conf)
    trainPixelCNNAE(conf, data)

