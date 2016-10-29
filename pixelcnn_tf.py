import tensorflow as tf
import numpy as np
from models import PixelCNN
from tensorflow.examples.tutorials.mnist import input_data

# TODO get mean if pixel value >1
mnist = input_data.read_data_sets("data/")
epochs = 10
batch_size = 50

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
        v_stack = PixelCNN([FILTER_SIZE, FILTER_SIZE, CHANNEL, F_MAP], [F_MAP], v_stack_in, mask=mask).output()
        v_stack_in = v_stack

    with tf.name_scope("v_stack_1"+i):
        v_stack_1 = PixelCNN([1, 1, F_MAP, F_MAP], [F_MAP], v_stack_in, gated=False, mask=mask).output()
        
    with tf.name_scope("h_stack"+i):
        h_stack = PixelCNN([1, FILTER_SIZE, CHANNEL, F_MAP], [F_MAP], h_stack_in, gated=True, payload=v_stack_1, mask=mask).output()

    with tf.name_scope("h_stack_1"+i):
        h_stack_1 = PixelCNN([1, 1, F_MAP, F_MAP], [F_MAP], h_stack, gated=False, mask=mask).output()
        h_stack_1 += h_stack_in
        h_stack_in = h_stack_1

with tf.name_scope("f_layer"):
    pred = PixelCNN([1, 1, F_MAP, 1],[1], h_stack_in, gated=False, mask='b', activation='sigmoid').output()
    pred = tf.reshape(pred, [batch_size, 784])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(X * tf.log(pred), reduction_indices=[1]))
#TODO gradient clipping
trainer = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
correct_preds = tf.equal(tf.argmax(X,1), tf.argmax(pred, 1))
accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))

#summary = tf.train.SummaryWriter('logs', sess.graph)

with tf.Session() as sess: 
    sess.run(tf.initialize_all_variables())
    for i in range(epochs):
        batch_X, batch_y = mnist.train.next_batch(batch_size)
        sess.run(trainer, feed_dict={X:batch_X})
        
        if i%1 == 0:
            print accuracy.eval(feed_dict={X:batch_X})
    print accuracy.eval(feed_dict={X:mnist.test.images})
