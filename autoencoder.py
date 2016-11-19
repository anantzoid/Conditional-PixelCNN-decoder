import tensorflow as tf
import numpy as np
from utils import *
import matplotlib.pyplot as plt
from models import PixelCNN
from layers import conv_op 

def get_weights(shape, name):
    return tf.Variable(tf.truncated_normal(shape=shape, stddev=0.1), name=name)

def get_biases(shape, name):
    return tf.Variable(tf.constant(shape=shape, value=0.1, dtype=tf.float32), name=name)

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


class AE(object):
    def __init__(self, X):
        self.num_layers = 6
        self.fmap_out = [8, 32]
        self.fmap_in = conf.channel
        self.fan_in = X
        self.filter_size = 4
        self.W = []
        self.strides = [1, 1, 1, 1]

        W_conv1 = get_weights([5, 5, conf.channel, 100], "W_conv1")
        b_conv1 = get_biases([100], "b_conv1")
        conv1 = tf.nn.relu(conv_op(X, W_conv1) + b_conv1)
        pool1 = max_pool_2x2(conv1)


        W_conv2 = get_weights([5, 5, 100, 150], "W_conv2")
        b_conv2 = get_biases([150], "b_conv2")
        conv2 = tf.nn.relu(conv_op(pool1, W_conv2) + b_conv2)
        pool2 = max_pool_2x2(conv2)

        W_conv3 = get_weights([3, 3, 150, 200], "W_conv3")
        b_conv3 = get_biases([200], "b_conv3")
        conv3 = tf.nn.relu(conv_op(pool2, W_conv3) + b_conv3)
        conv3_reshape = tf.reshape(conv3, (-1, 7*7*200))

        W_fc = get_weights([7*7*200, 10], "W_fc")
        b_fc = get_biases([10], "b_fc")
        self.pred = tf.nn.softmax(tf.add(tf.matmul(conv3_reshape, W_fc), b_fc))

def trainPixelCNNAE(conf, data):
    encoder_X = tf.placeholder(tf.float32, shape=[None, conf.img_height, conf.img_width, conf.channel])
    decoder_X = tf.placeholder(tf.float32, shape=[None, conf.img_height, conf.img_width, conf.channel])

    encoder = AE(encoder_X)
    decoder = PixelCNN(decoder_X, conf, encoder.pred)
    y = decoder.pred
    tf.scalar_summary('loss', decoder.loss)

    #loss = tf.reduce_mean(tf.square(encoder_X - y))
    #trainer = tf.train.GradientDescentOptimizer(1e-4).minimize(loss)
    trainer = tf.train.RMSPropOptimizer(1e-3)
    gradients = trainer.compute_gradients(decoder.loss)

    clipped_gradients = [(tf.clip_by_value(_[0], -conf.grad_clip, conf.grad_clip), _[1]) for _ in gradients]
    optimizer = trainer.apply_gradients(clipped_gradients)


    saver = tf.train.Saver(tf.trainable_variables())
    with tf.Session() as sess:
        merged = tf.merge_all_summaries()
        writer = tf.train.SummaryWriter('/tmp/mnist_ae', sess.graph)

        sess.run(tf.initialize_all_variables())

        if os.path.exists(conf.ckpt_file):
            saver.restore(sess, conf.ckpt_file)
            print "Model Restored"
        
        pointer = 0
        step = 0
        for i in range(conf.epochs):
            for j in range(conf.num_batches):
                if conf.data == 'mnist':
                    batch_X = binarize(data.train.next_batch(conf.batch_size)[0].reshape(conf.batch_size, conf.img_height, conf.img_width, conf.channel))
                else:
                    batch_X, pointer = get_batch(data, pointer, conf.batch_size)

                _, l, summary = sess.run([optimizer, decoder.loss, merged], feed_dict={encoder_X: batch_X, decoder_X: batch_X})
                writer.add_summary(summary, step)
                step += 1
            print "Epoch: %d, Cost: %f"%(i, l)
            if (i+1)%10 == 0:
                saver.save(sess, conf.ckpt_file)
                generate_ae(sess, encoder_X, decoder_X, y, data, conf, str(i))

        writer.close()
        '''
        data = input_data.read_data_sets("data/")
        n_examples = 10
        if conf.data == 'mnist':
            test_X_pure = data.train.next_batch(n_examples)[0].reshape(n_examples, conf.img_height, conf.img_width, conf.channel)
            test_X = binarize(test_X_pure)

        samples = sess.run(y, feed_dict={encoder_X:test_X, decoder_X:test_X})
        fig, axs = plt.subplots(2, n_examples, figsize=(n_examples, 2))
        for i in range(n_examples):
            axs[0][i].imshow(np.reshape(test_X_pure[i], (conf.img_height, conf.img_width)))
            axs[1][i].imshow(np.reshape(samples[i], (conf.img_height, conf.img_width)))
        fig.show()
        plt.draw()
        plt.waitforbuttonpress()
        '''


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
    conf.summary_path='/tmp/mnist_ae'
    conf.epochs=50
    conf.batch_size = 64
    conf.num_classes = 10

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
        data = data[0][0]
        data /= 255.0
        data = np.transpose(data, (0, 2, 3, 1))
        conf.img_height = 32
        conf.img_width = 32
        conf.channel = 3
        train_size = data.shape[0] 

    conf.num_batches = train_size // conf.batch_size
    conf = makepaths(conf)
    if tf.gfile.Exists(conf.summary_path):
        tf.gfile.DeleteRecursively(conf.summary_path)
    tf.gfile.MakeDirs(conf.summary_path)

    trainPixelCNNAE(conf, data)

