import numpy as np
import os
import scipy.misc
from datetime import datetime


def binarize(images):
    return (np.random.uniform(size=images.shape) < images).astype(np.float32)

def generate_samples(sess, X, h, pred, conf, suff):
    n_row, n_col = 5, 5
    samples = np.zeros((n_row*n_col, conf.img_height, conf.img_width, conf.channel), dtype=np.float32)
    # TODO make it generic
    labels = one_hot(np.array([1,2,3,4,5]*5), conf.num_classes)

    for i in xrange(conf.img_height):
        for j in xrange(conf.img_width):
            for k in xrange(conf.channel):
                data_dict = {X:samples}
                if conf.conditional is True:
                    data_dict[h] = labels
                next_sample = sess.run(pred, feed_dict=data_dict)
                if conf.data == "mnist":
                    next_sample = binarize(next_sample)
                samples[:, i, j, k] = next_sample[:, i, j, k]
        np.save('preds_'+suff+'.npy', samples)
        print "height:%d"%i
    save_images(samples, n_row, n_col, conf, suff)


def generate_ae(sess, encoder_X, decoder_X, y, data, conf):
    n_row, n_col = 5, 5
    samples = np.zeros((n_row*n_col, conf.img_height, conf.img_width, conf.channel), dtype=np.float32)
    if conf.data == 'mnist':
        labels = binarize(data.train.next_batch(n_row*n_col)[0].reshape(n_row*n_col, conf.img_height, conf.img_width, conf.channel))
    else:
        labels = get_batch(data, 0, n_row*n_col) 

    for i in xrange(conf.img_height):
        for j in xrange(conf.img_width):
            for k in xrange(conf.channel):
                next_sample = sess.run(y, {encoder_X: labels, decoder_X: samples})
                if conf.data == 'mnist':
                    next_sample = binarize(next_sample)
                samples[:, i, j, k] = next_sample[:, i, j, k]

    save_images(samples, n_row, n_col, conf, '')

def save_images(samples, n_row, n_col, conf, suff):
    images = samples 
    if conf.data == "mnist":
        images = images.reshape((n_row, n_col, conf.img_height, conf.img_width))
        images = images.transpose(1, 2, 0, 3)
        images = images.reshape((conf.img_height * n_row, conf.img_width * n_col))
    else:
        images = images.reshape((n_row, n_col, conf.img_height, conf.img_width, conf.channel))
        images = images.transpose(1, 2, 0, 3, 4)
        images = images.reshape((conf.img_height * n_row, conf.img_width * n_col, conf.channel))

    filename = datetime.now().strftime('%Y_%m_%d_%H_%M')+suff+".jpg"
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


def makepaths(conf):
    ckpt_full_path = os.path.join(conf.ckpt_path, "data=%s_bs=%d_layers=%d_fmap=%d"%(conf.data, conf.batch_size, conf.layers, conf.f_map))
    if not os.path.exists(ckpt_full_path):
        os.makedirs(ckpt_full_path)
    conf.ckpt_file = os.path.join(ckpt_full_path, "model.ckpt")

    conf.samples_path = os.path.join(conf.samples_path, "epoch=%d_bs=%d_layers=%d_fmap=%d"%(conf.epochs, conf.batch_size, conf.layers, conf.f_map))
    if not os.path.exists(conf.samples_path):
        os.makedirs(conf.samples_path)

    return conf
