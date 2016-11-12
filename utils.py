import numpy as np
import os
import scipy.misc
from datetime import datetime


def binarize(images):
    return (np.random.uniform(size=images.shape) < images).astype(np.float32)

def generate_and_save(sess, X, h, pred, conf):
    n_row, n_col = 5, 5
    samples = np.zeros((n_row*n_col, conf.img_height, conf.img_width, conf.channel), dtype=np.float32)
    # TODO make it generic
    labels = one_hot([1,2,3,4,5]*5)

    for i in xrange(conf.img_height):
        for j in xrange(conf.img_width):
            for k in xrange(conf.channel):
                next_sample = sess.run(pred, {X:samples, h: labels})
                if conf.data == "mnist":
                    next_sample = binarize(next_sample)
                samples[:, i, j, k] = next_sample[:, i, j, k]
    images = samples 
    images = images.reshape((n_row, n_col, conf.img_height, conf.img_width))
    images = images.transpose(1, 2, 0, 3)
    images = images.reshape((conf.img_height * n_row, conf.img_width * n_col))

    filename = datetime.now().strftime('%Y_%m_%d_%H_%M')+".jpg"
    scipy.misc.toimage(images, cmin=0.0, cmax=1.0).save(os.path.join(conf.samples_path, filename))


def get_batch(data, pointer, batch_size):
    if (batch_size + 1) * pointer >= data.shape[0]:
        pointer = 0
    batch = data[batch_size * pointer : batch_size * (pointer + 1)]
    pointer += 1
    return [batch, pointer]

def one_hot(batch_y, num_classes):
    y_ = np.zeros((batch_y.shape[0], num_classes))
    y_[np.arange(batch_y.shape[0]), batch_y] = 1
    return y_


