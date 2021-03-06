import tensorflow as tf
from ops import variables_on_gpu0
from ops import avg_grads
IMSIZE = 128
filename = '/media/NAS_SHARED/imagenet/imagenet_train_labeled_' + str(IMSIZE) + '.tfrecords'


def read_and_decode(filename_queue, labels=False):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    feat_dict = {
                 'image_raw': tf.FixedLenFeature([], tf.string),
                 'label' : tf.FixedLenFeature([], tf.int64)
                } if labels else {'image_raw': tf.FixedLenFeature([], tf.string),}

    features = tf.parse_single_example(
            serialized_example,
            features=feat_dict)

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image.set_shape(128 * 128 * 3)
    image = tf.reshape(image, [128, 128, 3])

    image = tf.cast(image, tf.float32) * (2. / 255) - 1.

    if labels:
        label = tf.cast(features['label'], tf.int32)
        return image, label

    return image

def read_and_decode_with_labels(filename_queue):
    return read_and_decode(filename_queue, labels=True)


def sigmoid_kl_with_logits(logits, targets):
    # broadcasts the same target value across the whole batch
    # this is implemented so awkwardly because tensorflow lacks an x log x op
    assert isinstance(targets, float)

    if targets in [0., 1.]:
        entropy = 0.
    else:
        entropy = - targets * np.log(targets) - (1. - targets) * np.log(1. - targets)

    return tf.nn.sigmoid_cross_entropy_with_logits(logits, tf.ones_like(logits) * targets) - entropy


def get_vars(self):
    t_vars = tf.trainable_variables()
    self.d_vars = [var for var in t_vars if var.name.startswith('d_')]
    self.g_vars = [var for var in t_vars if var.name.startswith('g_')]

    for x in self.d_vars:
        assert x not in self.g_vars

    for x in self.g_vars:
        assert x not in self.d_vars

    for x in t_vars:
        assert x in  self.g_vars or x in self.d_vars, x.name

    self.all_vars = t_vars

def build_model(self):
    all_d_grads = []
    all_g_grads = []
    config = self.config
    d_opt = tf.train.AdamOptimizer(config.discriminator_learning_rate, beta1=config.beta1)
    g_opt = tf.train.AdamOptimizer(config.generator_learning_rate, beta1=config.beta1)

    for idx, device in enumerate(self.devices):
        with tf.device("/%s" % device):
            with tf.name_scope("device_%s" % idx):
                with variables_on_gpu0():
                    build_model_single_gpu(self, idx)
                    d_grads = d_opt.compute_gradients(self.d_losses[-1], var_list=self.d_vars)
                    g_grads = g_opt.compute_gradients(self.g_losses[-1], var_list=self.g_vars)
                    all_d_grads.append(d_grads)
                    all_g_grads.append(g_grads)
                    tf.get_variable_scope().reuse_variables()
    avg_d_grads = avg_grads(all_d_grads)
    avg_g_grads = avg_grads(all_g_grads)
    self.d_optim = d_opt.apply_gradients(avg_d_grads)
    self.g_optim = g_opt.apply_gradients(avg_g_grads)

def build_model_single_gpu(self, gpu_idx):
    assert not self.y_dim

    if gpu_idx == 0:
        filename_queue = tf.train.string_input_producer([filename]) #  num_epochs=self.config.epoch)
        self.get_image, self.get_label = read_and_decode_with_labels(filename_queue)

        with tf.variable_scope("misc"):
            chance = 1. # TODO: declare this down below and make it 1. - 1. / num_classes
            avg_error_rate = tf.get_variable('avg_error_rate', [],
                    initializer=tf.constant_initializer(0.),
                    trainable=False)
            num_error_rate = tf.get_variable('num_error_rate', [],
                    initializer=tf.constant_initializer(0.),
                    trainable=False)

    images, sparse_labels = tf.train.shuffle_batch([self.get_image, self.get_label],
                    batch_size=self.batch_size,
                    num_threads=2,
                    capacity=1000 + 3 * self.batch_size,
                    min_after_dequeue=1000,
                    name='real_images_and_labels')
    if gpu_idx == 0:
        self.sample_images= tf.placeholder(tf.float32, [self.sample_size] + self.image_shape,
                                        name='sample_images')
        self.sample_labels = tf.placeholder(tf.int32, [self.sample_size], name="sample_labels")

        self.reference_G, self.reference_zs = self.generator(is_ref=True)
        # Since I don't know how to turn variable reuse off, I can only activate it once.
        # So here I build a dummy copy of the discriminator before turning variable reuse on for the generator.
        dummy_joint = tf.concat(0, [images, self.reference_G])
        dummy = self.discriminator(dummy_joint, reuse=False, prefix="dummy")

    G, zs = self.generator(is_ref=False)

    if gpu_idx == 0:
        G_means = tf.reduce_mean(G, 0, keep_dims=True)
        G_vars = tf.reduce_mean(tf.square(G - G_means), 0, keep_dims=True)
        G = tf.Print(G, [tf.reduce_mean(G_means), tf.reduce_mean(G_vars)], "generator mean and average var", first_n=1)
        image_means = tf.reduce_mean(images, 0, keep_dims=True)
        image_vars = tf.reduce_mean(tf.square(images - image_means), 0, keep_dims=True)
        images = tf.Print(images, [tf.reduce_mean(image_means), tf.reduce_mean(image_vars)], "image mean and average var", first_n=1)
        self.Gs = []
        self.zses = []
    self.Gs.append(G)
    self.zses.append(zs)

    joint = tf.concat(0, [images, G])
    class_logits, D_on_data, D_on_data_logits, D_on_G, D_on_G_logits = self.discriminator(joint, reuse=True, prefix="joint ")
    # D_on_G_logits = tf.Print(D_on_G_logits, [D_on_G_logits], "D_on_G_logits")

    self.d_sum = tf.histogram_summary("d", D_on_data)
    self.d__sum = tf.histogram_summary("d_", D_on_G)
    self.G_sum = tf.image_summary("G", G)

    d_label_smooth = self.d_label_smooth
    d_loss_real = sigmoid_kl_with_logits(D_on_data_logits, 1. - d_label_smooth)
    class_loss_weight = 1.
    d_loss_class = class_loss_weight * tf.nn.sparse_softmax_cross_entropy_with_logits(class_logits,
            tf.to_int64(sparse_labels))
    error_rate = 1. - tf.reduce_mean(tf.to_float(tf.nn.in_top_k(class_logits, sparse_labels, 1)))
    # self.d_loss_class = tf.Print(self.d_loss_class, [error_rate], "gpu " + str(gpu_idx) + " current minibatch error rate")
    if gpu_idx == 0:
        update = tf.assign(num_error_rate, num_error_rate + 1.)
        with tf.control_dependencies([update]):
            # Start off as a true average for 1st 100 samples
            # Then switch to a running average to compensate for ongoing learning
            tc = tf.maximum(.01, 1. / num_error_rate)
        update = tf.assign(avg_error_rate, (1. - tc) * avg_error_rate + tc * error_rate)
        with tf.control_dependencies([update]):
            d_loss_class = tf.Print(d_loss_class,
                [avg_error_rate], "running top-1 error rate")
    # Do not smooth the negative targets.
    # If we use positive targets of alpha and negative targets of beta,
    # then the optimal discriminator function is D(x) = (alpha p_data(x) + beta p_generator(x)) / (p_data(x) + p_generator(x)).
    # This means if we want to get less extreme values, we shrink alpha.
    # Increasing beta makes the generator self-reinforcing.
    # Note that using this one-sided label smoothing also shifts the equilibrium
    # value to alpha/2.
    d_loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(D_on_G_logits,
            tf.zeros_like(D_on_G_logits))
    g_loss = sigmoid_kl_with_logits(D_on_G_logits, self.generator_target_prob)
    d_loss_class = tf.reduce_mean(d_loss_class)
    d_loss_real = tf.reduce_mean(d_loss_real)
    d_loss_fake = tf.reduce_mean(d_loss_fake)
    g_loss = tf.reduce_mean(g_loss)
    if gpu_idx == 0:
        self.g_losses = []
    self.g_losses.append(g_loss)

    d_loss = d_loss_real + d_loss_fake + d_loss_class
    if gpu_idx == 0:
        self.d_loss_reals = []
        self.d_loss_fakes = []
        self.d_loss_classes = []
        self.d_losses = []
    self.d_loss_reals.append(d_loss_real)
    self.d_loss_fakes.append(d_loss_fake)
    self.d_loss_classes.append(d_loss_class)
    self.d_losses.append(d_loss)

    # self.g_loss_sum = tf.scalar_summary("g_loss", self.g_loss)
    # self.d_loss_sum = tf.scalar_summary("d_loss", self.d_loss)

    if gpu_idx == 0:
        get_vars(self)
