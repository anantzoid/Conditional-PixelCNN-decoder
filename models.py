import tensorflow as tf
from layers import *

class PixelCNN(object):
    def __init__(self, X, conf, h=None):
        self.X = X
        if conf.data == "mnist":
            self.X_norm = X
        else:
            '''
                Image normalization for CIFAR-10 was supposed to be done here
            '''
            self.X_norm = X
        v_stack_in, h_stack_in = self.X_norm, self.X_norm

        if conf.conditional is True:
            if h is not None:
                self.h = h
            else:
                self.h = tf.placeholder(tf.float32, shape=[None, conf.num_classes]) 
        else:
            self.h = None

        for i in range(conf.layers):
            filter_size = 3 if i > 0 else 7
            mask = 'b' if i > 0 else 'a'
            residual = True if i > 0 else False
            i = str(i)
            with tf.variable_scope("v_stack"+i):
                v_stack = GatedCNN([filter_size, filter_size, conf.f_map], v_stack_in, mask=mask, conditional=self.h).output()
                v_stack_in = v_stack

            with tf.variable_scope("v_stack_1"+i):
                v_stack_1 = GatedCNN([1, 1, conf.f_map], v_stack_in, gated=False, mask=mask).output()

            with tf.variable_scope("h_stack"+i):
                h_stack = GatedCNN([1, filter_size, conf.f_map], h_stack_in, payload=v_stack_1, mask=mask, conditional=self.h).output()

            with tf.variable_scope("h_stack_1"+i):
                h_stack_1 = GatedCNN([1, 1, conf.f_map], h_stack, gated=False, mask=mask).output()
                if residual:
                    h_stack_1 += h_stack_in # Residual connection
                h_stack_in = h_stack_1

        with tf.variable_scope("fc_1"):
            fc1 = GatedCNN([1, 1, conf.f_map], h_stack_in, gated=False, mask='b').output()

        if conf.data == "mnist":
            with tf.variable_scope("fc_2"):
                self.fc2 = GatedCNN([1, 1, 1], fc1, gated=False, mask='b', activation=False).output()
            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.fc2, self.X))
            self.pred = tf.nn.sigmoid(self.fc2)
        else:
            color_dim = 256
            with tf.variable_scope("fc_2"):
                self.fc2 = GatedCNN([1, 1, conf.channel * color_dim], fc1, gated=False, mask='b', activation=False).output()
                self.fc2 = tf.reshape(self.fc2, (-1, color_dim))

            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(self.fc2, tf.cast(tf.reshape(self.X, [-1]), dtype=tf.int32)))

            '''
                Since this code was not run on CIFAR-10, I'm not sure which 
                would be a suitable way to generate 3-channel images. Below are
                the 2 methods which may be used, with the first one (self.pred)
                being more likely.
            '''
            self.pred_sampling = tf.reshape(tf.multinomial(tf.nn.softmax(self.fc2), num_samples=1, seed=100), tf.shape(self.X))
            self.pred = tf.reshape(tf.argmax(tf.nn.softmax(self.fc2), dimension=tf.rank(self.fc2) - 1), tf.shape(self.X))


class ConvolutionalEncoder(object):
    def __init__(self, X, conf):
        '''
            This is the 6-layer architecture for Convolutional Autoencoder
            mentioned in the original paper: 
            Stacked Convolutional Auto-Encoders for Hierarchical Feature Extraction

            Note that only the encoder part is implemented as PixelCNN is taken
            as the decoder.
        '''

        W_conv1 = get_weights([5, 5, conf.channel, 100], "W_conv1")
        b_conv1 = get_bias([100], "b_conv1")
        conv1 = tf.nn.relu(conv_op(X, W_conv1) + b_conv1)
        pool1 = max_pool_2x2(conv1)

        W_conv2 = get_weights([5, 5, 100, 150], "W_conv2")
        b_conv2 = get_bias([150], "b_conv2")
        conv2 = tf.nn.relu(conv_op(pool1, W_conv2) + b_conv2)
        pool2 = max_pool_2x2(conv2)

        W_conv3 = get_weights([3, 3, 150, 200], "W_conv3")
        b_conv3 = get_bias([200], "b_conv3")
        conv3 = tf.nn.relu(conv_op(pool2, W_conv3) + b_conv3)
        conv3_reshape = tf.reshape(conv3, (-1, 7*7*200))

        W_fc = get_weights([7*7*200, 10], "W_fc")
        b_fc = get_bias([10], "b_fc")
        self.pred = tf.nn.softmax(tf.add(tf.matmul(conv3_reshape, W_fc), b_fc))


