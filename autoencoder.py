import tensorflow as tf
import numpy as np
from utils import *
from models import *

def trainAE(conf, data):
    encoder_X = tf.placeholder(tf.float32, shape=[None, conf.img_height, conf.img_width, conf.channel])
    decoder_X = tf.placeholder(tf.float32, shape=[None, conf.img_height, conf.img_width, conf.channel])

    encoder = ConvolutionalEncoder(encoder_X, conf)
    decoder = PixelCNN(decoder_X, conf, encoder.pred)
    y = decoder.pred
    tf.scalar_summary('loss', decoder.loss)

    trainer = tf.train.RMSPropOptimizer(1e-3)
    gradients = trainer.compute_gradients(decoder.loss)

    clipped_gradients = [(tf.clip_by_value(_[0], -conf.grad_clip, conf.grad_clip), _[1]) for _ in gradients]
    optimizer = trainer.apply_gradients(clipped_gradients)

    saver = tf.train.Saver(tf.trainable_variables())
    with tf.Session() as sess:
        merged = tf.merge_all_summaries()
        writer = tf.train.SummaryWriter(conf.summary_path, sess.graph)

        sess.run(tf.initialize_all_variables())

        if os.path.exists(conf.ckpt_file):
            saver.restore(sess, conf.ckpt_file)
            print "Model Restored"

        # TODO The training part below and in main.py could be generalized
        if conf.epochs > 0:
            print "Started Model Training..."
        pointer = 0
        step = 0
        for i in range(conf.epochs):
            for j in range(conf.num_batches):
                if conf.data == 'mnist':
                    batch_X = binarize(data.train.next_batch(conf.batch_size)[0].reshape(conf.batch_size, conf.img_height, conf.img_width, conf.channel))
                else:
                    batch_X, pointer = get_batch(data, pointer, conf.batch_size)

                _, l, summary = sess.run([optimizer, decoder.loss, merged], feed_dict={encoder_X: batch_X, decoder_X: batch_X})
                writer.add_summary(summary, step)
                step += 1

            print "Epoch: %d, Cost: %f"%(i, l)
            if (i+1)%10 == 0:
                saver.save(sess, conf.ckpt_file)
                generate_ae(sess, encoder_X, decoder_X, y, data, conf, str(i))

        writer.close()
        generate_ae(sess, encoder_X, decoder_X, y, data, conf, '')

