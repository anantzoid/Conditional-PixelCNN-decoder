# TODO masking, pred:final layer (X256)

import tensorflow as tf
import numpy as np
from models import PixelCNN

LAYERS = 3
F_MAP = 32
FILTER_SIZE = 7
CHANNEL = 1

X = tf.placeholder(tf.float32, shape=[None, 784])
X_image = tf.reshape(X, [-1, 28, 28, CHANNEL])
v_stack_in, h_stack_in = X_image, X_image

for i in range(LAYERS):
    FILTER_SIZE = 3 if i > 0 else FILTER_SIZE
    CHANNEL = F_MAP if i > 0 else CHANNEL
    mask = 'b' if i > 0 else 'a'
    i = str(i)

    with tf.name_scope("v_stack"+i):
        v_stack = PixelCNN([FILTER_SIZE, FILTER_SIZE, CHANNEL, F_MAP], [F_MAP], v_stack_in).output()
        v_stack_in = v_stack

    with tf.name_scope("v_stack_1"+i):
        v_stack_1 = PixelCNN([1, 1, F_MAP, F_MAP], [F_MAP], v_stack_in, gated=False).output()
        
    with tf.name_scope("h_stack"+i):
        h_stack = PixelCNN([1, FILTER_SIZE, CHANNEL, F_MAP], [F_MAP], h_stack_in, gated=True, payload=v_stack_1).output()

    with tf.name_scope("h_stack_1"+i):
        h_stack_1 = PixelCNN([1, 1, F_MAP, F_MAP], [F_MAP], h_stack, gated=False).output()
        h_stack_1 += h_stack_in
        h_stack_in = h_stack_1

pred = None
softmax = tf.nn.softmax(pred)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(softmax), reduction_indices=[1]))
#TODO gradient clipping
trainer = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)

sess = tf.Session()
#summary = tf.train.SummaryWriter('logs', sess.graph)
#Combine and Quantize into 255
