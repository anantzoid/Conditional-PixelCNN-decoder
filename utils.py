import numpy as np
import os
import scipy.misc
from datetime import datetime


def binarize(images):
    return (np.random.uniform(size=images.shape) < images).astype(np.float32)

def generate_and_save(sess, X, pred, conf):
    n_row, n_col = 5, 5
    samples = np.zeros((n_row*n_col, conf.img_height, conf.img_width, 1), dtype=np.float32)
    for i in xrange(conf.img_height):
        for j in xrange(conf.img_width):
            for k in xrange(1):
                next_sample = binarize(sess.run(pred, {X:samples}))
                samples[:, i, j, k] = next_sample[:, i, j, k]
    images = samples 
    images = images.reshape((n_row, n_col, conf.img_height, conf.img_width))
    images = images.transpose(1, 2, 0, 3)
    images = images.reshape((conf.img_height * n_row, conf.img_width * n_col))

    filename = datetime.now().strftime('%Y_%m_%d_%H_%M')+".jpg"
    scipy.misc.toimage(images, cmin=0.0, cmax=1.0).save(os.path.join(conf.samples_path, filename))


