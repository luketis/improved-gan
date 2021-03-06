import tensorflow as tf
import numpy as np
from itertools import count
from ops import lrelu, conv2d, linear

def discriminator(self, image, reuse=False, y=None, prefix=""):

    num_classes = 1001

    if reuse:
        tf.get_variable_scope().reuse_variables()

    batch_size = int(image.get_shape()[0])
    assert batch_size == 2 * self.batch_size

    """
    # L1 distance to average value of corresponding pixel in positive and negative batch
    # Included as a feature to prevent early mode collapse
    b, r, c, ch = [int(e) for e in image.get_shape()]
    pos = tf.slice(image, [0, 0, 0, 0], [self.batch_size, r, c, ch])
    neg = tf.slice(image, [self.batch_size, 0, 0, 0], [self.batch_size, r, c, ch])
    pos = tf.reshape(pos, [self.batch_size, -1])
    neg = tf.reshape(neg, [self.batch_size, -1])
    mean_pos = tf.reduce_mean(pos, 0, keep_dims=True)
    mean_neg = tf.reduce_mean(neg, 0, keep_dims=True)

    # difference from mean, with each example excluding itself from the mean
    pos_diff_pos = (1. + 1. / (self.batch_size - 1.)) * pos - mean_pos
    pos_diff_neg = pos - mean_neg
    neg_diff_pos = neg - mean_pos
    neg_diff_neg = (1. + 1. / (self.batch_size - 1.)) * neg - mean_neg

    diff_feat = tf.concat(0, [tf.concat(1, [pos_diff_pos, pos_diff_neg]),
                              tf.concat(1, [neg_diff_pos, neg_diff_neg])])

    with tf.variable_scope("d_diff_feat"):
        scale = tf.get_variable("d_untied_scale", [128 * 128 * 3 * 2], tf.float32,
                                 tf.random_normal_initializer(mean=1., stddev=0.1))

    diff_feat = diff_feat = tf.exp(- tf.abs(scale) * tf.abs(diff_feat))
    diff_feat = self.set_batch_normx(diff_feat, name="d_set_batch_normx_diff_feat")
    """

    noisy_image = image + tf.random_normal([batch_size, 128, 128, 3],
            mean=0.0,
            stddev=.1)

    print("Discriminator shapes")
    print("image: ", image.get_shape())

    def conv_tower(inp, dims, names, suffix, d_hs, d_ws, k_ws, k_hs):
        h = inp
        for dim, name, d_h, d_w, k_w, k_h, i in zip(dims, names, d_hs, d_ws, k_ws, k_hs, count()):
            h = lrelu(batch_norm(conv2d(h, dim, name=name + suffix, d_h=d_h, d_w=d_w,
                                        k_w=k_w, k_h=k_h), "d_batch_norm_" + str(i) + suffix))
            print("\th"+ str(i)+ " ", h.get_shape())
        return h

    def tower(batch_norm, suffix):
        assert not self.y_dim
        print("\ttower "+suffix)

    
        d_wh = [2,2,2,1,1,2]
        k_wh = [3 for i in range(6)]

        h5 = conv_tower(noisy_image, [self.df_dim, self.df_dim*2, self.df_dim*4, self.df_dim*4, self.df_dim*4, self.df_dim*8],
                        ['d_h' + str(i) + "_conv" for i in range(6)], d_wh, d_wh, k_wh, k_wh)

        h6 = lrelu(batch_norm(conv2d(h5, self.df_dim*8, name='d_h6_conv' + suffix,
            k_w=3, k_h=3), "d_batch_norm_6" + suffix))
        print("\th6 ", h6.get_shape())

        # return tf.reduce_mean(h6, [1, 2])
        h6_reshaped = tf.reshape(h6, [batch_size, -1])
        print('\th6_reshaped: ', h6_reshaped.get_shape())

        h7 = lrelu(batch_norm(linear(h6_reshaped, self.df_dim * 40, scope="d_h7" + suffix), "d_batch_norm_7" + suffix))

        return h7

    h = tower(self.set_batch_normx, "")
    print("h: ", h.get_shape())

    n_kernels = 300
    dim_per_kernel = 50
    x = linear(h, n_kernels * dim_per_kernel, scope="d_h")
    activation = tf.reshape(x, (batch_size, n_kernels, dim_per_kernel))

    big = np.zeros((batch_size, batch_size), dtype='float32')
    big += np.eye(batch_size)
    big = tf.expand_dims(big, 1)

    abs_dif = tf.reduce_sum(tf.abs(tf.expand_dims(activation, 3) - tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)), 2)
    mask = 1. - big
    masked = tf.exp(-abs_dif) * mask
    def half(tens, second):
        m, n, _ = tens.get_shape()
        m = int(m)
        n = int(n)
        return tf.slice(tens, [0, 0, second * self.batch_size], [m, n, self.batch_size])

    # TODO: speedup by allocating the denominator directly instead of constructing it by sum
    #       (current version makes it easier to play with the mask and not need to rederive
    #        the denominator)

    def feat_reduce(halve):
        halved = half(masked, halve)
        return tf.reduce_sum(halved, 2) / tf.reduce_sum(halved)

    f1, f2 = feat_reduce(0), feat_reduce(1)


    minibatch_features = [f1, f2]

    x = tf.concat(1, [h] + minibatch_features)
    print("x: ", x.get_shape())
    # x = tf.nn.dropout(x, .5)

    class_logits = linear(x, num_classes, 'd_indiv_logits')


    image_means = tf.reduce_mean(image, 0, keep_dims=True)
    mean_sub_image = image - image_means
    image_vars = tf.reduce_mean(tf.square(mean_sub_image), 0)

    generated_class_logits = tf.squeeze(tf.slice(class_logits, [0, num_classes - 1], [batch_size, 1]))
    positive_class_logits = tf.slice(class_logits, [0, 0], [batch_size, num_classes - 1])

    """
    # make these a separate matmul with weights initialized to 0, attached only to generated_class_logits, or things explode
    generated_class_logits = tf.squeeze(generated_class_logits) + tf.squeeze(linear(diff_feat, 1, stddev=0., scope="d_indivi_logits_from_diff_feat"))
    assert len(generated_class_logits.get_shape()) == 1
    # re-assemble the logits after incrementing the generated class logits
    class_logits = tf.concat(1, [positive_class_logits, tf.expand_dims(generated_class_logits, 1)])
    """

    mx = tf.reduce_max(positive_class_logits, 1, keep_dims=True)
    safe_pos_class_logits = positive_class_logits - mx

    gan_logits = tf.log(tf.reduce_sum(tf.exp(safe_pos_class_logits), 1)) + tf.squeeze(mx) - generated_class_logits
    assert len(gan_logits.get_shape()) == 1

    probs = tf.nn.sigmoid(gan_logits)

    return [tf.slice(class_logits, [0, 0], [self.batch_size, num_classes]),
            tf.slice(probs, [0], [self.batch_size]),
           tf.slice(gan_logits, [0], [self.batch_size]),
           tf.slice(probs, [self.batch_size], [self.batch_size]),
           tf.slice(gan_logits, [self.batch_size], [self.batch_size])]
