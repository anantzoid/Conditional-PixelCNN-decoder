
# coding: utf-8

# In[1]:

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from utils import *


# In[2]:

import matplotlib.pyplot as plt
#get_ipython().magic(u'matplotlib inline')

mnist = input_data.read_data_sets("data/")

# In[3]:

img_height = 28
img_width = 28
channel = 1

num_layers = 3
filter_size = 3
fmap_in = channel
fmap_out = 32
strides = [1, 1, 1, 1]

batch_size = 50

from models import PixelCNN
class Conf(object):
    pass

conf = Conf()
conf.ckpt_path='ckpts'
conf.conditional=True
conf.data='mnist'
conf.data_path='data'
conf.epochs=50
conf.f_map=32
conf.grad_clip=1
conf.layers=5
conf.samples_path='samples'
conf.num_classes = 10
conf.img_height = 28
conf.img_width = 28
conf.channel = 1
conf.num_batches = mnist.train.num_examples // batch_size
conf.type='train'



X = tf.placeholder(shape=[None, img_height, img_width, channel], dtype=tf.float32)

fan_in = X
W = []
for i in range(num_layers):
    if i == num_layers -1 :
        fmap_out = 10 
    W.append(tf.Variable(tf.truncated_normal(shape=[filter_size, filter_size, fmap_in, fmap_out], stddev=0.1), name="W_%d"%i))
    b = tf.Variable(tf.ones(shape=[fmap_out], dtype=tf.float32), name="encoder_b_%d"%i)
    en_conv = tf.nn.conv2d(fan_in, W[i], strides, padding='SAME', name="encoder_conv_%d"%i)

    fan_in = tf.tanh(tf.add(en_conv, b))
    fmap_in = fmap_out

fan_in = tf.reshape(fan_in, (-1, conf.img_width*conf.img_height*fmap_out))
conf.num_classes = int(fan_in.get_shape()[1])

# TODO
# Make X enter from model input
model = PixelCNN(conf)
# output is model.pre
# define loss function here after getting prediction
y = model.pred

'''
W.reverse()
for i in range(num_layers):
    if i == num_layers-1:
        fmap_out = channel
    c = tf.Variable(tf.ones(shape=[fmap_out], dtype=tf.float32), name="decoder_b_%d"%i)
    de_conv = tf.nn.conv2d_transpose(fan_in, W[i], [tf.shape(X)[0], img_height, img_width, fmap_out], strides, padding='SAME', name="decoder_conv_%d"%i)
    fan_in = tf.tanh(tf.add(de_conv, c))
y = fan_in    
'''


# In[10]:

loss = tf.reduce_mean(tf.square(X - y))
trainer = tf.train.GradientDescentOptimizer(1e-4).minimize(loss)


# In[5]:
'''
import cPickle
data =  cPickle.load(open('cifar-100-python/test', 'r'))['data']
data = np.reshape(data, (data.shape[0], 3, 32, 32))
data = np.transpose(data, (0, 2, 3, 1))
#data = (data - np.mean(data))/np.std(data)
'''


# In[ ]:
epochs = 5
num_batches = 1#mnist.train.num_examples // batch_size

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    
    for i in range(epochs):
        for j in range(num_batches):
            batch_X = binarize(mnist.train.next_batch(batch_size)[0].reshape(batch_size, img_height, img_width, channel))
            condition = sess.run([fan_in], feed_dict={X:batch_X})
            # TODO shape of condition does  not match: (1, 10, 28, 28, 32) for (?, 10)
            _, l = sess.run([trainer, loss], feed_dict={X:batch_X, model.X:batch_X, model.h: condition[0]})
            #batch_X = data[:10]/255.0
            #_, l = sess.run([trainer, loss], feed_dict={X:batch_X})
        print l

    n_examples = 10
    #test_X = mnist.train.next_batch(n_examples)[0].reshape(n_examples, img_height, img_width, channel)
 
    o_test_X = mnist.test.next_batch(10)[0].reshape(10, img_height, img_width, channel)
    test_X = binarize(o_test_X)
    condition = sess.run(fan_in, feed_dict={X:test_X})
    samples = sess.run(y, feed_dict={X: test_X, model.X:test_X, model.h:condition})
    print samples.shape
    #test_X = data[:10]
    #samples = sess.run(y, feed_dict={X:test_X/255.0})
    fig, axs = plt.subplots(2, n_examples, figsize=(10,2))
    for i in range(n_examples):
        axs[0][i].imshow(np.reshape(o_test_X[i], (img_height, img_width)), cmap='binary')
        axs[1][i].imshow(np.reshape(samples[i], (img_height, img_width)), cmap='binary')
    fig.show()
    plt.draw()
    plt.waitforbuttonpress()


