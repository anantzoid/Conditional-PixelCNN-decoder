# TODO
# kundan: concat payload instead of add
#       : arch: without 1X1 in 1st layer and last 2 layers
#       : replaces masking with n/2 filter
# check network arch
# autoencoder
# cost on test set
# make for imagenet data: upscale-downscale-q_level, mean pixel value if not mnist
# stats
# logger

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from datetime import datetime
import scipy.misc
import os
from models import PixelCNN

mnist = input_data.read_data_sets("data/")
epochs = 1
batch_size = 50
grad_clip = 1

img_height = 28
img_width = 28
channel = 1

LAYERS = 3
F_MAP = 32
FILTER_SIZE = 7

X = tf.placeholder(tf.float32, shape=[None, img_height, img_width, channel])
v_stack_in, h_stack_in = X, X

# TODO encapsulate
for i in range(LAYERS):
    FILTER_SIZE = 3 if i > 0 else FILTER_SIZE
    in_dim = F_MAP if i > 0 else channel
    mask = 'b' if i > 0 else 'a'
    i = str(i)
    with tf.variable_scope("v_stack"+i):
        v_stack = PixelCNN([FILTER_SIZE, FILTER_SIZE, F_MAP], v_stack_in, mask=mask).output()
        v_stack_in = v_stack

    with tf.variable_scope("v_stack_1"+i):
        v_stack_1 = PixelCNN([1, 1, F_MAP], v_stack_in, gated=False, mask=mask).output()

    with tf.variable_scope("h_stack"+i):
        h_stack = PixelCNN([1, FILTER_SIZE, F_MAP], h_stack_in, payload=v_stack_1, mask=mask).output()

    with tf.variable_scope("h_stack_1"+i):
        h_stack_1 = PixelCNN([1, 1, F_MAP], h_stack, gated=False, mask=mask).output()
        #h_stack_1 += h_stack_in # Residual connection
        h_stack_in = h_stack_1

with tf.variable_scope("fc_1"):
    fc1 = PixelCNN([1, 1, F_MAP], h_stack_in, gated=False, mask='b').output()

# handle Imagenet differently
with tf.variable_scope("fc_2"):
    fc2 = PixelCNN([1, 1, 1], fc1, gated=False, mask='b', activation=False).output()
pred = tf.nn.sigmoid(fc2)

loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(fc2, X))

trainer = tf.train.RMSPropOptimizer(1e-3)
gradients = trainer.compute_gradients(loss)

clipped_gradients = [(tf.clip_by_value(_[0], -grad_clip, grad_clip), _[1]) for _ in gradients]
optimizer = trainer.apply_gradients(clipped_gradients)


def binarize(images):
    return (0.0 < images).astype(np.float32)

def generate_and_save(sess):
    n_row, n_col = 5, 5
    samples = np.zeros((n_row*n_col, img_height, img_width, 1), dtype=np.float32)
    for i in xrange(img_height):
        for j in xrange(img_width):
            for k in xrange(1):
                next_sample = binarize(sess.run(pred, {X:samples}))
                samples[:, i, j, k] = next_sample[:, i, j, k]
    images = samples 
    images = images.reshape((n_row, n_col, img_height, img_width))
    images = images.transpose(1, 2, 0, 3)
    images = images.reshape((img_height * n_row, img_width * n_col))

    filename = '%s_%s.jpg' % ("sample", str(datetime.now()))
    scipy.misc.toimage(images, cmin=0.0, cmax=1.0).save(os.path.join("samples", filename))

num_batches = mnist.train.num_examples // batch_size
with tf.Session() as sess: 
    sess.run(tf.initialize_all_variables())
    for i in range(epochs):
        for j in range(num_batches):
            batch_X = binarize(mnist.train.next_batch(batch_size)[0] \
                    .reshape([batch_size, img_height, img_width, 1]))
            _, cost = sess.run([optimizer, loss], feed_dict={X:batch_X})

            print "Epoch: %d, Cost: %f"%(i, cost)
        generate_and_save(sess)

    generate_and_save(sess)

