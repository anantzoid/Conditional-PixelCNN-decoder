import tensorflow as tf
from layers import GatedCNN

class PixelCNN():
    def __init__(self, conf):
        
        self.X = tf.placeholder(tf.float32, shape=[None, conf.img_height, conf.img_width, conf.channel])
        self.X_norm = self.X if conf.data == "mnist" else tf.div(self.X, 255.0)
        v_stack_in, h_stack_in = self.X_norm, self.X_norm
        # TODO norm for multichannel: dubtract mean and divide by std feature-wise
        self.h = tf.placeholder(tf.float32, shape=[None, conf.num_classes])

        for i in range(conf.layers):
            filter_size = 3 if i > 0 else 7
            mask = 'b' if i > 0 else 'a'
            residual = True if i > 0 else False
            i = str(i)
            with tf.variable_scope("v_stack"+i):
                v_stack = GatedCNN([filter_size, filter_size, conf.f_map], v_stack_in, mask=mask, conditional=self.h).output()
                v_stack_in = v_stack

            with tf.variable_scope("v_stack_1"+i):
                v_stack_1 = GatedCNN([1, 1, conf.f_map], v_stack_in, gated=False, mask=mask).output()

            with tf.variable_scope("h_stack"+i):
                h_stack = GatedCNN([1, filter_size, conf.f_map], h_stack_in, payload=v_stack_1, mask=mask, conditional=self.h).output()

            with tf.variable_scope("h_stack_1"+i):
                h_stack_1 = GatedCNN([1, 1, conf.f_map], h_stack, gated=False, mask=mask).output()
                if residual:
                    h_stack_1 += h_stack_in # Residual connection
                h_stack_in = h_stack_1

        with tf.variable_scope("fc_1"):
            fc1 = GatedCNN([1, 1, conf.f_map], h_stack_in, gated=False, mask='b').output()

        if conf.data == "mnist":
            with tf.variable_scope("fc_2"):
                self.fc2 = GatedCNN([1, 1, 1], fc1, gated=False, mask='b', activation=False).output()
            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.fc2, self.X))
            self.pred = tf.nn.sigmoid(self.fc2)
        else:
            color_dim = 256
            with tf.variable_scope("fc_2"):
                self.fc2 = GatedCNN([1, 1, conf.channel * color_dim], fc1, gated=False, mask='b', activation=False).output()
                #fc2_shape = self.fc2.get_shape()
                #self.fc2 = tf.reshape(self.fc2, (int(fc2_shape[0]), int(fc2_shape[1]), int(fc2_shape[2]), conf.channel, -1)) 
                #fc2_shape = self.fc2.get_shape()
                #self.fc2 = tf.nn.softmax(tf.reshape(self.fc2, (-1, int(fc2_shape[-1])))) 
                self.fc2 = tf.reshape(self.fc2, (-1, color_dim))

            #self.loss = self.categorical_crossentropy(self.fc2, self.X)
            #self.X_flat = tf.reshape(self.X, [-1])
            #self.fc2_flat = tf.cast(tf.argmax(self.fc2, dimension=tf.rank(self.fc2) - 1), dtype=tf.float32)
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(self.fc2, tf.cast(tf.reshape(self.X, [-1]), dtype=tf.int32)))
            #self.loss = tf.reduce_mean(-tf.reduce_sum(X_ohehot * tf.log(self.fc2), reduction_indices=[1]))

            # NOTE or check without argmax
            self.pred = tf.reshape(tf.argmax(tf.nn.softmax(self.fc2), dimension=tf.rank(self.fc2) - 1), tf.shape(self.X))

