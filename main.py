import tensorflow as tf
import numpy as np
import argparse
from models import PixelCNN
from utils import *

def train(conf, data):
    model = PixelCNN(conf)
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(model.fc2, model.X))

    trainer = tf.train.RMSPropOptimizer(1e-3)
    gradients = trainer.compute_gradients(loss)

    clipped_gradients = [(tf.clip_by_value(_[0], -conf.grad_clip, conf.grad_clip), _[1]) for _ in gradients]
    optimizer = trainer.apply_gradients(clipped_gradients)

    saver = tf.train.Saver(tf.trainable_variables())
    with tf.Session() as sess: 
        if os.path.exists(conf.ckpt_file):
            saver.restore(sess, conf.ckpt_file)
            print "Model Restored"
        else:
            sess.run(tf.initialize_all_variables())

        for i in range(conf.epochs):
            for j in range(conf.num_batches):
                batch_X = binarize(data.train.next_batch(conf.batch_size)[0] \
                        .reshape([conf.batch_size, conf.img_height, conf.img_width, conf.channel]))
                _, cost = sess.run([optimizer, loss], feed_dict={model.X:batch_X})

                print "Epoch: %d, Cost: %f"%(i, cost)

        generate_and_save(sess, model.X, model.pred, conf)
        saver.save(sess, conf.ckpt_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='train')
    parser.add_argument('--data', type=str, default='mnist')
    parser.add_argument('--layers', type=int, default=12)
    parser.add_argument('--f_map', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--grad_clip', type=int, default=1)
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--ckpt_path', type=str, default='ckpts')
    parser.add_argument('--samples_path', type=str, default='samples')
    conf = parser.parse_args()
  
    if conf.data == 'mnist':
        from tensorflow.examples.tutorials.mnist import input_data
        if not os.path.exists(conf.data_path):
            os.makedirs(conf.data_path)
        data = input_data.read_data_sets(conf.data_path)
        conf.img_height = 28
        conf.img_width = 28
        conf.channel = 1
        conf.num_batches = 10#mnist.train.num_examples // conf.batch_size
        conf.filter_size = 7

    ckpt_full_path = os.path.join(conf.ckpt_path, "data=%s_bs=%d_layers=%d_fmap=%d"%(conf.data, conf.batch_size, conf.layers, conf.f_map))
    if not os.path.exists(ckpt_full_path):
        os.makedirs(ckpt_full_path)
    conf.ckpt_file = os.path.join(ckpt_full_path, "model.ckpt")

    conf.samples_path = os.path.join(conf.samples_path, "epoch=%d_bs=%d_layers=%d_fmap=%d"%(conf.epochs, conf.batch_size, conf.layers, conf.f_map))
    if not os.path.exists(conf.samples_path):
        os.makedirs(conf.samples_path)

    train(conf, data)
