import os
import time
from glob import glob
import tensorflow as tf

from ops import *
from utils import *

filename = "/media/NAS_SHARED/imagenet/imagenet_train_128.tfrecords"

class DCGAN(object):
    def __init__(self, sess, image_size=108, is_crop=True,
                 batch_size=64, image_shape=[64, 64, 3],
                 y_dim=None, z_dim=100, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='default',
                 d_label_smooth=.25,
                 generator_target_prob=1.,
                 checkpoint_dir=None, sample_dir='samples',
                 generator=None,
                 generator_func=None, train=None, train_func=None,
                 generator_cls = None,
                 discriminator_func=None,
                 encoder_func=None,
                 build_model=None,
                 build_model_func=None, config=None,
                 devices=None,
                 disable_virt_batch_norm=False,
                 sample_size=64,
		 out_init_b=0.,
                 out_stddev=.15):
        """

        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            y_dim: (optional) Dimension of dim for y. [None]
            z_dim: (optional) Dimension of dim for Z. [100]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
            dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
            c_dim: (optional) Dimension of image color. [3]
        """
        self.disable_virt_batch_norm = disable_virt_batch_norm
        self.devices = devices
        self.d_label_smooth = d_label_smooth
        self.out_init_b = out_init_b
	self.out_stddev = out_stddev
        self.config = config
        self.generator_target_prob = generator_target_prob

        if generator is not None:
            generator.dcgan = self
        else:
            if generator_func is None:
                generator_func = default_generator
            if generator_cls is None:
                generator_cls = Generator
            generator = generator_cls(self, generator_func)

        self.generator = generator

        if discriminator_func is None:
            discriminator_func = default_discriminator

        self.discriminator = Discriminator(self, discriminator_func)

        if train is not None:
            self.train = train
            train.dcgan = self
        else:
            if train_func is None:
                train_func = default_train
            self.train = Train(self, train_func)

        if build_model is not None:
            assert build_model_func is None

            build_model.gan = self
            self.build_model = build_model
        else:
            if build_model_func is None:
                build_model_func = default_build_model

            self.build_model = BuildModel(self, build_model_func)

        self.sess = sess
        self.is_crop = is_crop
        self.batch_size = batch_size
        self.image_size = image_size
        self.sample_size = sample_size
        self.image_shape = image_shape
        self.sample_dir = sample_dir

        self.y_dim = y_dim
        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        self.c_dim = 3

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_batch_norm1 = batch_norm(batch_size, name='d_batch_norm1')
        self.d_batch_norm2 = batch_norm(batch_size, name='d_batch_norm2')

        if not self.y_dim:
            self.d_batch_norm3 = batch_norm(batch_size, name='d_batch_norm3')

        self.g_batch_norm0 = batch_norm(batch_size, name='g_batch_norm0')
        self.g_batch_norm1 = batch_norm(batch_size, name='g_batch_norm1')
        self.g_batch_norm2 = batch_norm(batch_size, name='g_batch_norm2')

        if not self.y_dim:
            self.g_batch_norm3 = batch_norm(batch_size, name='g_batch_norm3')
        # Not used by all generators

        self.g_batch_norm4 = batch_norm(batch_size, name='g_batch_norm4')
        self.g_batch_norm5 = batch_norm(batch_size, name='g_batch_norm5')

        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.build_model()

    def set_batch_norm(self, tensor, name, batch_size=None, name_func=None):
        # the batch size argument is actually unused
        assert name.startswith('g_') or name.startswith('d_'), name

        if name_func is not None:
            setattr(self, name, name_func(name=name))

        elif not hasattr(self, name):
            setattr(self, name, batch_norm(batch_size, name=name))

        my_batch_norm = getattr(self, name)

        return my_batch_norm(tensor)

    def set_batch_norm2(self, tensor, name):
        return self.set_batch_norm(tensor, name, name_func=batch_norm_second_half)

    def set_batch_norm1(self, tensor, name):
        return self.set_batch_norm(tensor, name, name_func=batch_norm_first_half)

    def set_batch_normx(self, tensor, name):
        return self.set_batch_norm(tensor, name, name_func=batch_norm_cross)

    def virt_batch_norm(self, tensor, name, half=None, clss=VIRT_BATCH_NORM):

        if self.disable_virt_batch_norm:
            class Dummy(object):
                def __init__(self, tensor, ignored, half):
                    self.reference_output=tensor
                def __call__(self, x):
                    return x
            VIRT_BATCH_NORM_cls = Dummy
        else:
            VIRT_BATCH_NORM_cls = clss

        if not hasattr(self, name):

            my_virt_batch_norm = VIRT_BATCH_NORM_cls(tensor, name, half=half)
            setattr(self, name, my_virt_batch_norm)

            return my_virt_batch_norm.reference_output

        my_virt_batch_norm = getattr(self, name)

        return my_virt_batch_norm(tensor)

    def virt_batch_norm_log(self, tensor, name, half=None):
        return self.virt_batch_norm(tensor, name, half=half, clss=VIRT_BATCH_NORM_LOG)


    def virt_batch_norm_log_pixel(self, tensor, name, half=None):
        return self.virt_batch_norm(tensor, name, half=half, clss=VIRT_BATCH_NORM_LOG_PIXEL)
    

    def virt_batch_norm1(self, tensor, name):
        return self.virt_batch_norm(tensor, name, half=1)

    def virt_batch_norm2(self, tensor, name):
        return self.virt_batch_norm(tensor, name, half=2)


    def save(self, checkpoint_dir, step):
        model_name = "DCGAN.model"
        model_dir = "%s_%s" % (self.dataset_name, self.batch_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")

        model_dir = "%s_%s" % (self.dataset_name, self.batch_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            print "Bad checkpoint: ", ckpt
            return False


class BuildModel(object):
    """
    A class that builds the generator forward prop when called.

    Parameters
    ----------
    dcgan: The DCGAN object to build within.
    func: The function to do it with.
    """
    def __init__(self, dcgan, func):
        self.dcgan = dcgan
        self.func = func

    def __call__(self):
        return self.func(self.dcgan)

class Generator(object):
    """
    A class that builds the generator forward prop when called.

    Parameters
    ----------
    dcgan: The DCGAN object to build the generator within.
    func: The function to do it with.
    """
    def __init__(self, dcgan, func):
        self.dcgan = dcgan
        self.func = func

    def __call__(self, z, y=None):
        return self.func(self.dcgan, z, y)


class Discriminator(object):
    """
    A class that builds the discriminator forward prop when called.

    Parameters
    ----------
    dcgan: The DCGAN object to build the discriminator within.
    func: The function to do it with.
    """
    def __init__(self, dcgan, func):
        self.dcgan = dcgan
        self.func = func

    def __call__(self, image, reuse=False, y=None, prefix=""):
        return self.func(self.dcgan, image, reuse, y, prefix)

class Train(object):
    """
    A class that runs the training loop when called.

    Parameters
    ----------
    dcgan: The DCGAN object to train.
    func: The function to do it with.
    """
    def __init__(self, dcgan, func):
        self.dcgan = dcgan
        self.func = func

    def __call__(self, config):
        return self.func(self.dcgan, config)


class VIRT_BATCH_NORM(object):
    """
    Virtual Batch Normalization
    """

    def __init__(self, x, name, epsilon=1e-5, half=None, log=False, per_pixel=False):
        """
        x is the reference batch
        """
        assert isinstance(epsilon, float)

        self.half = half
        shape = x.get_shape().as_list()
        needs_reshape = len(shape) != 4
        self.log = log
        self.per_pixel = per_pixel
        if needs_reshape:
            x, shape, orig_shape = reshape(x, shape)

        with tf.variable_scope(name) as scope:

            assert name.startswith("d_") or name.startswith("g_")

            self.epsilon = epsilon
            self.name = name

            if self.half is None:
                half = x
            elif self.half == 1:
                half = tf.slice(x, [0, 0, 0, 0],
                                      [shape[0] // 2, shape[1], shape[2], shape[3]])
            elif self.half == 2:
                half = slice_x_half_batch_norm(x, shape)
            else:
                assert False


            if self.per_pixel:
                self.mean = tf.reduce_mean(half, [0], keep_dims=True)
                self.mean_sq = tf.reduce_mean(tf.square(half), [0], keep_dims=True)
            else:
                self.mean = tf.reduce_mean(half, [0, 1, 2], keep_dims=True)
                self.mean_sq = tf.reduce_mean(tf.square(half), [0, 1, 2], keep_dims=True)

            self.batch_size = int(half.get_shape()[0])

            assert x is not None
            assert self.mean is not None
            assert self.mean_sq is not None

            out = self._normalize(x, self.mean, self.mean_sq, "reference")

            if needs_reshape:
                out = tf.reshape(out, orig_shape)

            self.reference_output = out

    def __call__(self, x):

        shape = x.get_shape().as_list()
        needs_reshape = len(shape) != 4

        if needs_reshape:
            x, shape, orig_shape = self.reshape(x, shape)

        with tf.variable_scope(self.name) as scope:

            new_coeff = 1. / (self.batch_size + 1.)
            old_coeff = 1. - new_coeff

            if self.per_pixel:
                new_mean = x
                new_mean_sq = tf.square(x)
            else:
                new_mean = tf.reduce_mean(x, [1, 2], keep_dims=True)
                new_mean_sq = tf.reduce_mean(tf.square(x), [1, 2], keep_dims=True)

            mean = new_coeff * new_mean + old_coeff * self.mean
            mean_sq = new_coeff * new_mean_sq + old_coeff * self.mean_sq
            out = self._normalize(x, mean, mean_sq, "live")

            if needs_reshape:
                out = tf.reshape(out, orig_shape)

            return out

    def _normalize(self, x, mean, mean_sq, message):
        # make sure this is called with a variable scope
        shape = x.get_shape().as_list()

        assert len(shape) == 4

        if self.log:
            self.gamma_driver = tf.get_variable("gamma_driver", [shape[-1]],
                                initializer=tf.random_normal_initializer(0., 0.02))
            gamma = tf.exp(self.gamma_driver)
            gamma = tf.reshape(gamma, [1, 1, 1, -1])
            self.beta = tf.get_variable("beta", [shape[-1]],
                                        initializer=tf.constant_initializer(0.))
            beta = tf.reshape(self.beta, [1, 1, 1, -1])

        elif self.per_pixel:
            self.gamma_driver = tf.get_variable("gamma_driver", shape[1:],
                                initializer=tf.random_normal_initializer(0., 0.02))
            gamma = tf.exp(self.gamma_driver)
            gamma = tf.expand_dims(gamma, 0)
            self.beta = tf.get_variable("beta", shape[1:],
                                initializer=tf.constant_initializer(0.))
            beta = tf.expand_dims(self.beta, 0)

        else:
            self.gamma = tf.get_variable("gamma", [shape[-1]],
                                initializer=tf.random_normal_initializer(1., 0.02))
            gamma = tf.reshape(self.gamma, [1, 1, 1, -1])
            self.beta = tf.get_variable("beta", [shape[-1]],
                                initializer=tf.constant_initializer(0.))
            beta = tf.reshape(self.beta, [1, 1, 1, -1])

        assert self.epsilon is not None
        assert mean_sq is not None
        assert mean is not None

        std = tf.sqrt(self.epsilon + mean_sq - tf.square(mean))
        out = x - mean
        out = out / std

        # out = tf.Print(out, [tf.reduce_mean(out, [0, 1, 2]),
        #    tf.reduce_mean(tf.square(out - tf.reduce_mean(out, [0, 1, 2], keep_dims=True)), [0, 1, 2])],
        #    message, first_n=-1)

        out = out * gamma
        out = out + beta

        return out

class VIRT_BATCH_NORM_LOG(VIRT_BATCH_NORM):
    """
    Virtual Batch Normalization, Log scale for the scale parameter
    """

    def __init__(self, x, name, epsilon=1e-5, half=None):
        super(VIRT_BATCH_NORM_LOG, self).__init__(x, name, epsilon=epsilon, half=half, log=True)


class VIRT_BATCH_NORM_LOG_PIXEL(VIRT_BATCH_NORM):
    """
    Virtual Batch Normalization, Log scale for the scale parameter, per-Pixel normalization
    """

    def __init__(self, x, name, epsilon=1e-5, half=None):
        super(VIRT_BATCH_NORM_LOG_PIXEL, self).__init__(x, name, epsilon=epsilon, half=half, per_pixel=True)

