import tensorflow as tf
from layers import GatedCNN

class PixelCNN():
    def __init__(self, conf):

        self.X = tf.placeholder(tf.float32, shape=[None, conf.img_height, conf.img_width, conf.channel])
        v_stack_in, h_stack_in = self.X, self.X

        for i in range(conf.layers):
            filter_size = 3 if i > 0 else conf.filter_size
            in_dim = conf.f_map if i > 0 else conf.channel
            mask = 'b' if i > 0 else 'a'
            residual = True if i > 0 else False
            i = str(i)
            with tf.variable_scope("v_stack"+i):
                v_stack = GatedCNN([filter_size, filter_size, conf.f_map], v_stack_in, mask=mask).output()
                v_stack_in = v_stack

            with tf.variable_scope("v_stack_1"+i):
                v_stack_1 = GatedCNN([1, 1, conf.f_map], v_stack_in, gated=False, mask=mask).output()

            with tf.variable_scope("h_stack"+i):
                h_stack = GatedCNN([1, filter_size, conf.f_map], h_stack_in, payload=v_stack_1, mask=mask).output()

            with tf.variable_scope("h_stack_1"+i):
                h_stack_1 = GatedCNN([1, 1, conf.f_map], h_stack, gated=False, mask=mask).output()
                if residual:
                    h_stack_1 += h_stack_in # Residual connection
                h_stack_in = h_stack_1

        with tf.variable_scope("fc_1"):
            fc1 = GatedCNN([1, 1, conf.f_map], h_stack_in, gated=False, mask='b').output()

        # handle Imagenet differently
        with tf.variable_scope("fc_2"):
            self.fc2 = GatedCNN([1, 1, 1], fc1, gated=False, mask='b', activation=False).output()
        self.pred = tf.nn.sigmoid(self.fc2)
