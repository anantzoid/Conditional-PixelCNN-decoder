import tensorflow as tf
import numpy as np

def get_weights(shape, mask=None):
    weights_initializer = tf.contrib.layers.xavier_initializer()
    W = tf.get_variable("weights", shape, tf.float32, weights_initializer)

    if mask:
        filter_mid_x = shape[0]//2
        filter_mid_y = shape[1]//2
        mask_filter = np.ones(shape, dtype=np.float32)
        mask_filter[filter_mid_x, filter_mid_y+1:, :, :] = 0.
        mask_filter[filter_mid_x+1:, :, :, :] = 0.

        if mask == 'a':
            mask_filter[filter_mid_x, filter_mid_y, :, :] = 0.
            
        W *= mask_filter 
    return W


def get_bias(shape):
    return tf.get_variable("biases", shape, tf.float32, tf.zeros_initializer)

def conv_op(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

class PixelCNN():
    def __init__(self, W_shape, b_shape, fan_in, gated=True, payload=None, mask=None, activation=True):
        self.W_shape = W_shape
        self.b_shape = b_shape
        self.fan_in = fan_in
        self.payload = payload
        self.mask = mask
        self.activation = activation

        if gated:
            self.gated_conv()
        else:
            self.simple_conv()

    def gated_conv(self):
        W_f = get_weights(self.W_shape, mask=self.mask)
        b_f = get_bias(self.b_shape)
        W_g = get_weights(self.W_shape, mask=self.mask)
        b_g = get_bias(self.b_shape)
       
        conv_f = conv_op(self.fan_in, W_f)
        conv_g = conv_op(self.fan_in, W_g)
       
        if self.payload is not None:
            conv_f += self.payload
            conv_g += self.payload

        self.fan_out = tf.mul(tf.tanh(conv_f + b_f), tf.sigmoid(conv_g + b_g))

    def simple_conv(self):
        W = get_weights(self.W_shape, mask=self.mask)
        b = get_bias(self.b_shape)
        conv = conv_op(self.fan_in, W)
        if self.activation: 
            self.fan_out = tf.nn.relu(tf.add(conv, b))
        else:
            self.fan_out = tf.add(conv, b)


    def output(self):
        return self.fan_out 


