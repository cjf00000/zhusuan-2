#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import time

import numpy as np
import tensorflow as tf
from six.moves import range
from tensorflow.contrib import layers
import zhusuan as zs

from examples import conf
from examples.utils import dataset, multi_gpu, save_image_collections
from examples.utils.multi_gpu import FLAGS

def minibatch_discrimination(f, num_kernels=100, dim_per_kernel=5):
    # f: n * a
    a = f.get_shape()[1]
    b = num_kernels
    c = dim_per_kernel
    T = tf.get_variable('mbd', shape=[a, b, c],
            initializer=layers.xavier_initializer())
    # M: n * b * c
    M = tf.tensordot(f, T, [[1], [0]])
    # c: n * n * b
    M0 = tf.expand_dims(M, 0)       # 1 * n * b * c
    M1 = tf.expand_dims(M, 1)       # n * 1 * b * c
    c = tf.reduce_sum(tf.abs(M0 - M1), -1)  # n * n * b
    c = tf.exp(-c)
    o = tf.reduce_sum(c, 1)         # n * b
    return tf.concat((f, o), 1)


@zs.reuse('generator')
def generator(observed, n, n_z, is_training):
    # TODO virtual BN
    with zs.BayesianNet(observed=observed) as generator:
        normalizer_params = {'is_training': is_training,
                             'updates_collections': None}
        ngf = 64
        z_min = -tf.ones([n, n_z])
        z_max = tf.ones([n, n_z])
        z = zs.Uniform('z', z_min, z_max)
        lx_z = layers.fully_connected(z, num_outputs=ngf*8*4*4,
                                      normalizer_fn=layers.batch_norm,
                                      normalizer_params=normalizer_params)
        lx_z = tf.reshape(lx_z, [-1, 4, 4, ngf*8])
        lx_z = layers.conv2d_transpose(lx_z, ngf*4, 5, stride=2,
                                       normalizer_fn=layers.batch_norm,
                                       normalizer_params=normalizer_params)
        lx_z = layers.conv2d_transpose(lx_z, ngf*2, 5, stride=2,
                                       normalizer_fn=layers.batch_norm,
                                       normalizer_params=normalizer_params)
        lx_z = layers.conv2d_transpose(lx_z, 3, 5, stride=2,
                                       activation_fn=tf.nn.sigmoid)
    return generator, lx_z

@zs.reuse('classifier')
def classifier(x, n_y, is_training):
# TODO: leaky relu
    normalizer_params = {'is_training': is_training,
                         'updates_collections': None}

    # ==================== Simple ==================
    ndf = 32
    lc_x = layers.conv2d(x, ndf*2, 5, stride=2,
                         normalizer_fn=layers.batch_norm,
                         normalizer_params=normalizer_params)
    lc_x = layers.conv2d(lc_x, ndf*4, 5, stride=2,
                         normalizer_fn=layers.batch_norm,
                         normalizer_params=normalizer_params)
    lc_x = layers.conv2d(lc_x, ndf*8, 5, stride=2,
                         normalizer_fn=layers.batch_norm,
                         normalizer_params=normalizer_params)
    lc_x = layers.flatten(lc_x)
#
#    # ==================== VGG ======================
#    # 32 * 32 * 96
#    lc_x = layers.conv2d(x, 96, 3,
#                         normalizer_fn=layers.batch_norm,
#                         normalizer_params=normalizer_params)
#    print('Discriminator arch')
#    print(lc_x.get_shape())
#    lc_x = layers.conv2d(lc_x, 96, 3,
#                         normalizer_fn=layers.batch_norm,
#                         normalizer_params=normalizer_params)
#    print(lc_x.get_shape())
#    lc_x = layers.conv2d(lc_x, 96, 3, stride=2,
#                         normalizer_fn=layers.batch_norm,
#                         normalizer_params=normalizer_params)
#    print(lc_x.get_shape())
#    lc_x = layers.dropout(lc_x, keep_prob=0.5, is_training=is_training)
#    print(lc_x.get_shape())
#    print()
#
#    # 16 * 16 * 192
#    lc_x = layers.conv2d(lc_x, 192, 3,
#                         normalizer_fn=layers.batch_norm,
#                         normalizer_params=normalizer_params)
#    print(lc_x.get_shape())
#    lc_x = layers.conv2d(lc_x, 192, 3,
#                         normalizer_fn=layers.batch_norm,
#                         normalizer_params=normalizer_params)
#    print(lc_x.get_shape())
#    lc_x = layers.conv2d(lc_x, 192, 3, stride=2,
#                         normalizer_fn=layers.batch_norm,
#                         normalizer_params=normalizer_params)
#    print(lc_x.get_shape())
#    lc_x = layers.dropout(lc_x, keep_prob=0.5, is_training=is_training)
#    print(lc_x.get_shape())
#    print()
#
#    # 8 * 8 * 192
#    lc_x = layers.conv2d(lc_x, 192, 3,
#                         normalizer_fn=layers.batch_norm,
#                         normalizer_params=normalizer_params)
#    print(lc_x.get_shape())
#    lc_x = layers.conv2d(lc_x, 192, 3,
#                         normalizer_fn=layers.batch_norm,
#                         normalizer_params=normalizer_params)
#    print(lc_x.get_shape())
#    lc_x = layers.conv2d(lc_x, 192, 3, stride=2,
#                         normalizer_fn=layers.batch_norm,
#                         normalizer_params=normalizer_params)
#    print(lc_x.get_shape())
#    print()
#
#    # 4 * 4 * 192
#    lc_x = tf.reduce_max(lc_x, reduction_indices=[1, 2])
#    print(lc_x.get_shape())

    # 192
    lc_x = minibatch_discrimination(lc_x)
    lc_x = layers.batch_norm(lc_x, is_training=is_training, updates_collections=None)
    print(lc_x.get_shape())
    # 292
    class_logits = layers.fully_connected(lc_x, n_y, activation_fn=None)
    print(class_logits.get_shape())
    # 10

    return class_logits


class BatchGenerator():
    def __init__(self, x, y, bs):
        self.x = x
        self.y = y
        self.cnt = 0
        self.bs = bs

    def shuffle(self):
        perm = np.random.permutation(self.x.shape[0])
        self.x = self.x[perm]
        self.y = self.y[perm]

    def next_batch(self):
        prev_cnt = self.cnt
        next_cnt = self.cnt + self.bs
        if next_cnt > self.x.shape[0]:
            self.shuffle()
            self.cnt = self.bs
            prev_cnt, next_cnt = 0, self.bs
        else:
            self.cnt = next_cnt
        return self.x[prev_cnt:next_cnt], self.y[prev_cnt:next_cnt]

    def batches(self):
        ret = []
        bs = self.bs
        for i in range(self.x.shape[0] // bs + 1):
            x = self.x[i*bs : (i+1)*bs]
            y = self.y[i*bs : (i+1)*bs]
            if x.shape[0] != 0:
                ret.append((x, y))
        return ret


if __name__ == "__main__":
    tf.set_random_seed(1234)

    # Load CIFAR
    data_path = os.path.join(conf.data_dir, 'cifar10',
                             'cifar-10-python.tar.gz')
    np.random.seed(1234)
    x_train, y_train, x_test, y_test = \
        dataset.load_cifar10(data_path, normalize=True, one_hot=True)
    _, n_xl, _, n_channels = x_train.shape
    n_y = y_train.shape[1]
    n_sup = 4000

    batch_size = 32 * FLAGS.num_gpus
    perm = np.random.permutation(x_train.shape[0])
    b_sup  = BatchGenerator(x_train[perm[:n_sup]], y_train[perm[:n_sup]], batch_size)
    b_uns  = BatchGenerator(x_train[perm[n_sup:]], y_train[perm[n_sup:]], batch_size)
    b_test = BatchGenerator(x_test, y_test, batch_size)

    # Define model parameters
    n_z = 100

    # Define training/evaluation parameters
    lb_samples = 1
    epoches = 1000
    gen_size = 100
    iters = x_train.shape[0] // batch_size
    learning_rate = 3e-4

    # Build the computation graph
    is_training = tf.placeholder(tf.bool, shape=[], name='is_training')
    x_sup_ph = tf.placeholder(tf.float32, shape=(None, n_xl, n_xl, n_channels), name='x_sup')
    x_uns_ph = tf.placeholder(tf.float32, shape=(None, n_xl, n_xl, n_channels), name='x_uns')
    y_ph = tf.placeholder(tf.float32, shape=(None, n_y))
    one = tf.ones_like(tf.reduce_sum(y_ph, -1))
    zero = tf.zeros_like(one)
    y_real = tf.stack((zero, one), 1)
    y_fake = tf.stack((one, zero), 1)
    learning_rate_ph = tf.placeholder(tf.float32, shape=[], name='lr')
    optimizer = tf.train.AdamOptimizer(learning_rate_ph, beta1=0.5)

    def build_tower_graph(x_sup_ph, x_uns_ph, id_):
        segment_size = tf.shape(x_sup_ph)[0] // FLAGS.num_gpus
        def shard(x):
            return x[id_*segment_size : (id_+1)*segment_size]
        # build the losses
        gen, x_gen = generator(None, segment_size, n_z, is_training)
        x_sup_logits = classifier(shard(x_sup_ph), n_y, is_training)
        x_uns_logits = classifier(shard(x_uns_ph), n_y, is_training)
        x_gen_logits = classifier(x_gen, n_y, is_training)
        def mult_to_bin(x_logits):
            one_class = tf.reduce_logsumexp(x_logits, -1)
            zero_class = tf.zeros_like(one_class)
            bin_logits = tf.stack((zero_class, one_class), 1)
            return bin_logits

        accuracy = tf.nn.in_top_k(x_sup_logits, tf.argmax(shard(y_ph), 1), k=1)
#accuracy = tf.Print(accuracy, [accuracy, tf.argmax(x_sup_logits, 1), tf.argmax(shard(y_ph), 1)])
        accuracy = tf.reduce_sum(tf.cast(accuracy, tf.float32))

        # TODO label smoothing
        x_uns_bin_logits = mult_to_bin(x_uns_logits)
        x_gen_bin_logits = mult_to_bin(x_gen_logits)
        loss_sup = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=shard(y_ph),   logits=x_sup_logits))
        loss_uns = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=shard(y_real), logits=x_uns_bin_logits))
        loss_gen = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=shard(y_fake), logits=x_gen_bin_logits))
        clf_loss = loss_sup + loss_uns + loss_gen
        gen_loss = -loss_gen

        # optimizer ops
        clf_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                         scope='classifier')
        gen_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                         scope='generator')
        clf_grads = optimizer.compute_gradients(clf_loss, var_list=clf_var_list)
        gen_grads = optimizer.compute_gradients(gen_loss, var_list=gen_var_list)
        grads = clf_grads + gen_grads
        return grads, gen_loss, clf_loss, accuracy

    tower_losses = []
    tower_grads = []
    for i in range(FLAGS.num_gpus):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('tower_%d' % i):
                grads, gen_loss, clf_loss, accuracy = build_tower_graph(x_sup_ph, x_uns_ph, i)
                tower_losses.append([gen_loss, clf_loss, accuracy])
                tower_grads.append(grads)
    gen_loss, clf_loss, accuracy = multi_gpu.average_losses(tower_losses)
    accuracy *= FLAGS.num_gpus
    grads = multi_gpu.average_gradients(tower_grads)
    train = optimizer.apply_gradients(grads)
    _, eval_x_gen = generator(None, gen_size, n_z, is_training)

    gen_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                     scope='generator')

    # Run the inference
    with multi_gpu.create_session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(1, epoches + 1):
            time_train = -time.time()
            gen_losses, clf_losses = [], []
            for t in range(iters):
                xs, ys = b_sup.next_batch()
                xu, _ = b_uns.next_batch()
                _, g_loss, c_loss = sess.run(
                    [train, gen_loss, clf_loss],
                    feed_dict={x_sup_ph: xs, y_ph: ys, x_uns_ph: xu,
                               learning_rate_ph: learning_rate,
                               is_training: True})
#print(g_loss, c_loss)
                gen_losses.append(g_loss)
                clf_losses.append(c_loss)
                if t % 100 == 0:
                    print(t, time_train+time.time(), np.mean(gen_losses), np.mean(clf_losses))
                    time_test = -time.time()
                    images = sess.run(eval_x_gen,
                                      feed_dict={is_training: False})
                    name = "results/improvedgan/improvedgan.epoch.{}.png".format(epoch)
                    save_image_collections(images, name, scale_each=True)

                    # Test
                    a = 0.0
                    for xx, yy in b_test.batches():
                        acc = sess.run(accuracy, 
                                       feed_dict={x_sup_ph: xx, y_ph: yy, is_training: False})
                        a += acc
                    a /= x_test.shape[0]
                    time_test += time.time()
                    print('Testing ({:.2f}s) accuracy = {}'.format(time_test, a))

            time_train += time.time()
            if True:
                print('Epoch={} ({:.2f} s): Gen loss = {} Clf loss = {}'.
                      format(epoch, time_train, 
                             np.mean(gen_losses), np.mean(clf_losses)))

