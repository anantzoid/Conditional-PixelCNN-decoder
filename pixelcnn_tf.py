import tensorflow as tf
import numpy as np

LAYERS = 1
F_MAP = 32
FILTER_SIZE = 7
CHANNEL = 1

def get_weights(shape):
    # TODO set init bounds
    return tf.Variable(tf.truncated_normal(shape=shape, stddev=0.1))

def get_bias(shape):
    return tf.Variable(tf.constant(shape=shape, value=0.1, dtype=tf.float32))

def conv(x, W):
    # TODO check strides
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def gated():
    #TODO for gating y = tanh(W1*X) <element-wise product> sigmoid(W2*X)
    # also check from figure about splitting 2p feature maps into p
    return None

X = tf.placeholder(tf.float32, shape=[None, 784])
X_image = tf.reshape(X, [-1, 28, 28, CHANNEL])
v_stack_in, h_stack_in = X_image, X_image

class Conv():
    def __init__(self, W_shape, b_shape, fan_in, gated=True):
        self.W_f = get_weights(W_shape)
        self.b_f = get_bias(b_shape)
        self.W_g = get_weights(W_shape)
        self.b_g = get_bias(b_shape)
        
        conv_f = conv(fan_in, self.W_f)
        conv_g = conv(fan_in, self.W_g)
        
        self.fan_out = tf.mul(tf.tanh(conv_f + self.b_f), tf.sigmoid(conv_g + self.b_g))

    def output(self):
        return self.fan_out 

for i in range(LAYERS):
    FILTER_SIZE = 3 if i > 0 else FILTER_SIZE
    CHANNEL = F_MAP if i > 0 else CHANNEL
    i = str(i)
    with tf.name_scope("v_stack"+i):
        v_stack = Conv([FILTER_SIZE, FILTER_SIZE, CHANNEL, F_MAP], [F_MAP], v_stack_in).output()
        '''
        v_W = get_weights([FILTER_SIZE, FILTER_SIZE, CHANNEL, F_MAP])
        v_b = get_bias([F_MAP])
        print v_stack_in.get_shape(), v_W.get_shape()
        v_stack = conv(v_stack_in, v_W)
        #TODO gating
        v_stack_gate = tf.nn.relu(v_stack + v_b)
        '''
        v_stack_in = v_stack
        print "v_stack", v_stack.get_shape()
    '''
    with tf.name_scope("v_stack_1"+i):
        v_W_1 = get_weights([1, 1, F_MAP, F_MAP])
        v_b_1 = get_bias([F_MAP])
        v_stack_1 = tf.nn.relu(conv(v_stack, v_W_1) + v_b_1)
        print "v_stack_1", v_stack_1.get_shape()
        
        
    #TODO masking
    with tf.name_scope("h_stack"+i):
        h_W = get_weights([1, FILTER_SIZE, CHANNEL, F_MAP])
        h_b = get_bias([F_MAP])
        h_stack = conv(h_stack_in, h_W)
        #TODO gating
        h_stack_gate = tf.nn.relu(h_stack + v_b)
        print "h_stack", h_stack.get_shape()

    with tf.name_scope("h_stack_1"+i):
        h_W_1 = get_weights([1, 1, F_MAP, F_MAP])
        h_b_1 = get_bias([F_MAP])
        # TODO replace i/p with gated o/p
        h_stack_1 = tf.nn.relu(conv(h_stack_gate, h_W_1) + h_b_1) 
        # TODO add residual conn.

        h_stack_in = h_stack_1
    '''

sess = tf.Session()
summary = tf.train.SummaryWriter('logs', sess.graph)
#Combine and Quantize into 255

