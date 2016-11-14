import tensorflow as tf
import numpy as np
import argparse
from models import PixelCNN
from utils import *

def train(conf, data):
    X = tf.placeholder(tf.float32, shape=[None, conf.img_height, conf.img_width, conf.channel])
    model = PixelCNN(conf, X)

    trainer = tf.train.RMSPropOptimizer(1e-3)
    gradients = trainer.compute_gradients(model.loss)

    clipped_gradients = [(tf.clip_by_value(_[0], -conf.grad_clip, conf.grad_clip), _[1]) for _ in gradients]
    optimizer = trainer.apply_gradients(clipped_gradients)

    saver = tf.train.Saver(tf.trainable_variables())

    with tf.Session() as sess: 
        sess.run(tf.initialize_all_variables())
        if os.path.exists(conf.ckpt_file):
            saver.restore(sess, conf.ckpt_file)
            print "Model Restored"
       
        pointer = 0
        for i in range(conf.epochs):
            for j in range(conf.num_batches):
                if conf.data == "mnist":
                    batch_X, batch_y = data.train.next_batch(conf.batch_size)
                    batch_X = binarize(batch_X.reshape([conf.batch_size, \
                            conf.img_height, conf.img_width, conf.channel]))
                    batch_y = one_hot(batch_y, conf.num_classes) 
                else:
                    batch_X, pointer = get_batch(data, pointer, conf.batch_size)

                _, cost = sess.run([optimizer, model.loss], feed_dict={X:batch_X, model.h:batch_y})

            print "Epoch: %d, Cost: %f"%(i, cost)

        saver.save(sess, conf.ckpt_file)
        generate_samples(sess, X, model.h, model.pred, conf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='train')
    parser.add_argument('--data', type=str, default='mnist')
    parser.add_argument('--layers', type=int, default=5)
    parser.add_argument('--f_map', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--grad_clip', type=int, default=1)
    parser.add_argument('--conditional', type=bool, default=False)
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--ckpt_path', type=str, default='ckpts')
    parser.add_argument('--samples_path', type=str, default='samples')
    conf = parser.parse_args()
  
    if conf.data == 'mnist':
        from tensorflow.examples.tutorials.mnist import input_data
        if not os.path.exists(conf.data_path):
            os.makedirs(conf.data_path)
        data = input_data.read_data_sets(conf.data_path)
        conf.num_classes = 10
        conf.img_height = 28
        conf.img_width = 28
        conf.channel = 1
        conf.num_batches = data.train.num_examples // conf.batch_size
    else:
        from keras.datasets import cifar10
        data = cifar10.load_data()
        data = data[0][0]
        data = np.transpose(data, (0, 2, 3, 1))
        conf.img_height = 32
        conf.img_width = 32
        conf.channel = 3
        raise ValueError("Specify num_classes")
        conf.num_classes = 10
        conf.num_batches = data.shape[0] // conf.batch_size


    conf = makepaths(conf) 
    train(conf, data)
