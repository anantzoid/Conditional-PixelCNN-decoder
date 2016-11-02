import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from models import PixelCNN
from utils import *

mnist = input_data.read_data_sets("data/")
epochs = 50
batch_size = 100
grad_clip = 1
num_batches = mnist.train.num_examples // batch_size
ckpt_dir = "ckpts"
samples_dir = "samples"
if not os.path.isdir(ckpt_dir):
    os.makedirs(ckpt_dir)
ckpt_file = os.path.join(ckpt_dir, "model.ckpt")

img_height = 28
img_width = 28
channel = 1

LAYERS = 12
F_MAP = 32
FILTER_SIZE = 7

X = tf.placeholder(tf.float32, shape=[None, img_height, img_width, channel])
v_stack_in, h_stack_in = X, X

for i in range(LAYERS):
    FILTER_SIZE = 3 if i > 0 else FILTER_SIZE
    in_dim = F_MAP if i > 0 else channel
    mask = 'b' if i > 0 else 'a'
    residual = True if i > 0 else False
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
        if residual:
            h_stack_1 += h_stack_in # Residual connection
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

saver = tf.train.Saver()
with tf.Session() as sess: 
    if os.path.exists(ckpt_file):
        saver.restore(sess, ckpt_file)
        print "Model Restored"
    else:
        sess.run(tf.initialize_all_variables())

    for i in range(epochs):
        for j in range(num_batches):
            batch_X = binarize(mnist.train.next_batch(batch_size)[0] \
                    .reshape([batch_size, img_height, img_width, 1]))
            _, cost = sess.run([optimizer, loss], feed_dict={X:batch_X})

            print "Epoch: %d, Cost: %f"%(i, cost)

    generate_and_save(sess, X, pred, img_height, img_width, epochs, samples_dir)
    saver.save(sess, ckpt_file)
