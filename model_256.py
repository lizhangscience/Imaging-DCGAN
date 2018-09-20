# Original Version: Taehoon Kim (http://carpedm20.github.io)
#   + Source: https://github.com/carpedm20/DCGAN-tensorflow/blob/e30539fb5e20d5a0fed40935853da97e9e55eee8/model.py
#   + License: MIT
# [2016-08-05] Modifications for Completion: Brandon Amos (http://bamos.github.io)
#   + License: MIT

from __future__ import division
import os
import time
import math
import itertools
from glob import glob
import tensorflow as tf
from six.moves import xrange
import pylab as pl
pl.switch_backend('agg')


from ops import *
from utils import *

SUPPORTED_EXTENSIONS = ["png", "jpg", "jpeg"]

def dataset_files(root):
    """Returns a list of all image files in the given directory"""
    return list(itertools.chain.from_iterable(
        glob(os.path.join(root, "*.{}".format(ext))) for ext in SUPPORTED_EXTENSIONS))

def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))

class DCGAN(object):
    def __init__(self, sess, data_path, image_size=256, is_crop=False,
                 batch_size=10, sample_size=94, lowres=8,
                 z_dim=100, gf_dim=8, df_dim=8,
                 gfc_dim=1024, dfc_dim=1024, c_dim=1,
                 checkpoint_dir=None, lam=0.1):
        """

        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            lowres: (optional) Low resolution image/mask shrink factor. [8]
            z_dim: (optional) Dimension of dim for Z. [100]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            gfc_dim: (optional) Dimension of gen untis for for fully connected layer. [1024]
            dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
            c_dim: (optional) Dimension of image color. [3]
        """
        # Currently, image size must be a (power of 2) and (8 or higher).
        # assert(image_size & (image_size - 1) == 0 and image_size >= 8)

        self.sess = sess
        self.is_crop = is_crop
        self.batch_size = batch_size
        self.image_size = image_size
        self.sample_size = sample_size
        self.image_shape = [image_size, image_size,c_dim]

        self.lowres = lowres
        self.lowres_size = image_size // lowres
        self.lowres_shape = [self.lowres_size, self.lowres_size, c_dim]

        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        # self.gfc_dim = gfc_dim
        # self.dfc_dim = dfc_dim

        self.lam = lam

        self.c_dim = c_dim

        #get data input
        self.data, self.data_idx = get_data(data_path)
        self.data_shuffle = self.shuffle_data()


        # print(np.shape(self.data))
        # exit()

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bns = [
            batch_norm(name='d_bn{}'.format(i,)) for i in range(6)]

        log_size = int(math.log(image_size) / math.log(2))
        print('logsize ',log_size)
        self.g_bns = [
            batch_norm(name='g_bn{}'.format(i,)) for i in range(log_size)]

        self.checkpoint_dir = checkpoint_dir
        self.build_model()

        self.model_name = "DCGAN.model"


    def shuffle_data(self):
        np.random.shuffle(self.data_idx)
        return self.data[self.data_idx,:,:]

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.images = tf.placeholder(
            tf.float32, [None] + self.image_shape, name='real_images')

        # self.images = tf.placeholder(
        #     tf.float32, [self.batch_size,self.image_size,self.image_size,1], name='real_images')

        # self.lowres_images = tf.reduce_mean(tf.reshape(self.images,
        #     [self.batch_size, self.lowres_size, self.lowres,
        #      self.lowres_size, self.lowres, self.c_dim]), [2, 4])

        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        self.z_sum = tf.summary.histogram("z", self.z)

        self.G = self.generator(self.z)

        # self.lowres_G = tf.reduce_mean(tf.reshape(self.G,
        #     [self.batch_size, self.lowres_size, self.lowres,
        #      self.lowres_size, self.lowres, self.c_dim]), [2, 4])

        print('G ',np.shape(self.G))

        self.D, self.D_logits = self.discriminator(self.images)

        self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)

        self.d_sum = tf.summary.histogram("d", self.D)
        self.d__sum = tf.summary.histogram("d_", self.D_)
        self.G_sum = tf.summary.image("G", self.G,max_outputs=self.batch_size)

        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits,
                                                    labels=tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_,
                                                    labels=tf.zeros_like(self.D_)))
        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_,
                                                    labels=tf.ones_like(self.D_)))

        self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)

        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver(max_to_keep=1)

        # Completion.
        self.mask = tf.placeholder(tf.float32, self.image_shape, name='mask')
        # self.lowres_mask = tf.placeholder(tf.float32, self.lowres_shape, name='lowres_mask')
        # self.contextual_loss = tf.reduce_sum(
        #     tf.contrib.layers.flatten(
        #         tf.abs(tf.multiply(self.mask, self.G) - tf.multiply(self.mask, self.images))), 1)


        self.contextual_loss = tf.reduce_sum(
            tf.contrib.layers.flatten(
            tf.abs(self.G - self.images)),1)


        # self.contextual_loss += tf.reduce_sum(
        #     tf.contrib.layers.flatten(
        #         tf.abs(tf.multiply(self.lowres_mask, self.lowres_G) - tf.multiply(self.lowres_mask, self.lowres_images))), 1)

        self.perceptual_loss = self.g_loss
        self.complete_loss = self.contextual_loss + self.lam*self.perceptual_loss
        self.grad_complete_loss = tf.gradients(self.complete_loss, self.z)

    def train(self, config):

        # get number of batches
        batch_idxs = min(len(self.data[:, 0, 0]), config.train_size) // self.batch_size


        d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                          .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                          .minimize(self.g_loss, var_list=self.g_vars)                
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        self.g_sum = tf.summary.merge(
            [self.z_sum, self.d__sum, self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = tf.summary.merge(
            [self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

        sample_z = np.random.uniform(-1, 1, size=(self.sample_size , self.z_dim))
        # sample_files = data[0:self.sample_size]

        # sample = [get_image(sample_file, self.image_size, is_crop=self.is_crop) for sample_file in sample_files]
        # sample = [self.data[i,:,:] for i in range(len(self.data[:,0,0]))]
        # sample_images = np.array(sample).astype(np.float32)
        # sample_images = np.reshape(sample_images,[len(self.data[:,0,0]),self.image_size,self.image_size,1])

        # print(np.shape(sample_images))
        # exit()

        counter = 1
        start_time = time.time()


        for epoch in range(config.epoch):

            #reshuffle data every epoch
            # self.data_shuffle = self.shuffle_data()
            self.data_shuffle = self.data

            for idx in range(0, batch_idxs):
                # print('IDX: ',idx)

                #setup batch data
                batch = [self.data[i + idx * self.batch_size, :, :] for i in range(self.batch_size)]
                batch_images = np.array(batch).astype(np.float32)
                # print(np.shape(batch_images))
                batch_images = np.reshape(batch_images,[self.batch_size,self.image_size,self.image_size,1])
                # print('batch_image ',np.shape(batch_images))


                #setup batch z
                batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]) \
                            .astype(np.float32)


                # Update D network
                _, summary_str = self.sess.run([d_optim, self.d_sum],
                    feed_dict={ self.images: batch_images, self.z: batch_z, self.is_training: True })
                self.writer.add_summary(summary_str, counter)


                # Update G network
                _, summary_str = self.sess.run([g_optim, self.g_sum],
                    feed_dict={ self.z: batch_z, self.is_training: True })
                self.writer.add_summary(summary_str, counter)


                # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                _, summary_str = self.sess.run([g_optim, self.g_sum],
                    feed_dict={ self.z: batch_z, self.is_training: True })
                self.writer.add_summary(summary_str, counter)


                errD_fake = self.d_loss_fake.eval({self.z: batch_z, self.is_training: False})
                errD_real = self.d_loss_real.eval({self.images: batch_images, self.is_training: False})
                errG = self.g_loss.eval({self.z: batch_z, self.is_training: False})

                counter += 1
                print("Epoch: [{:2d}] [{:4d}/{:4d}] time: {:4.4f}, d_loss: {:.8f}, g_loss: {:.8f}".format(
                    epoch, idx, batch_idxs, time.time() - start_time, errD_fake+errD_real, errG))


            if epoch % 10 == 0:
                self.save(config.checkpoint_dir, counter)


    def complete(self, config):
        def make_dir(name):
            # Works on python 2.7, where exist_ok arg to makedirs isn't available.
            p = os.path.join(config.outDir, name)
            if not os.path.exists(p):
                os.makedirs(p)
        make_dir('hats_imgs')
        make_dir('completed')
        make_dir('difference')
        make_dir('logs')
        make_dir('final')

        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        isLoaded = self.load(self.checkpoint_dir)
        assert(isLoaded)

        nImgs = config.batch_size

        # batch_idxs = int(np.ceil(nImgs/self.batch_size))
        batch_idxs = min(len(self.data[:, 0, 0]), config.train_size) // self.batch_size

        lowres_mask = np.zeros(self.lowres_shape)
        if config.maskType == 'random':
            fraction_masked = 0.2
            mask = np.ones(self.image_shape)
            mask[np.random.random(self.image_shape[:2]) < fraction_masked] = 0.0
        elif config.maskType == 'center':
            assert(config.centerScale <= 0.5)
            mask = np.ones(self.image_shape)
            sz = self.image_size
            l = int(self.image_size*config.centerScale)
            u = int(self.image_size*(1.0-config.centerScale))
            mask[l:u, l:u, :] = 0.0
        elif config.maskType == 'left':
            mask = np.ones(self.image_shape)
            c = self.image_size // 2
            mask[:,:c,:] = 0.0
        elif config.maskType == 'full':
            mask = np.ones(self.image_shape)
        elif config.maskType == 'grid':
            mask = np.zeros(self.image_shape)
            mask[::4,::4,:] = 1.0

        else:
            assert(False)

        for idx in range(0, batch_idxs):

            print('IDX: ',idx, '/',batch_idxs)
            # self.images = tf.placeholder(
            #     tf.float32, [None] + self.image_shape, name='real_images')

            l = idx*self.batch_size
            u = min((idx+1)*self.batch_size, nImgs)
            batchSz = u-l

            batch_images = 0
            # print('batch size ',self.batch_size)
            # print([i + idx*self.batch_size for i in range(self.batch_size)])

            frame_idx = [i + idx*self.batch_size for i in range(self.batch_size)]
            batch = [self.data[i + idx*self.batch_size, :, :] for i in range(self.batch_size)]
            batch_images = np.array(batch).astype(np.float32)
            batch_images = np.reshape(batch_images, [self.batch_size, self.image_size, self.image_size, 1])

            # print('batchimage size ',np.shape(batch_images))


            # if batchSz < self.batch_size:
            #     print(batchSz)
            #     padSz = ((0, int(self.batch_size-batchSz)), (0,0), (0,0), (0,0))
            #     batch_images = np.pad(batch_images, padSz, 'constant')
            #     batch_images = batch_images.astype(np.float32)

            zhats = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))
            m = 0
            v = 0

            nRows = np.ceil(batchSz/8)
            nCols = min(8, batchSz)

            save_image(batch_images[0,:,:,0], [nRows,nCols],
                        os.path.join(config.outDir, 'before.png'))
            masked_images = np.multiply(batch_images, mask)

            save_image(masked_images[0,:,:,0], [nRows,nCols],
                        os.path.join(config.outDir, 'masked.png'))


            for img in range(batchSz):
                with open(os.path.join(config.outDir, 'logs/hats_{:02d}.log'.format(img)), 'a') as f:
                    f.write('iter loss ' +
                            ' '.join(['z{}'.format(zi) for zi in range(self.z_dim)]) +
                            '\n')


            for i in range(config.nIter):
                fd = {
                    self.z: zhats,
                    self.images:batch_images,
                    self.is_training: False
                }

                # print('idx ',idx,' i ',i,' images ',np.shape(batch_images))

                run = [self.complete_loss, self.grad_complete_loss, self.G]
                loss, g, G_imgs = self.sess.run(run, feed_dict=fd)



                for img in range(batchSz):
                    with open(os.path.join(config.outDir, 'logs/hats_{:02d}.log'.format(img)), 'ab') as f:
                        f.write('{} {} '.format(i, loss[img]).encode())
                        np.savetxt(f, zhats[img:img+1])

                if i % config.outInterval == 0:
                    print(i, np.mean(loss))
                    # imgName = os.path.join(config.outDir,
                    #                        'hats_imgs/{:04d}.png'.format(i+idx*self.batch_size))
                    # nRows = np.ceil(batchSz/8)
                    # nCols = min(8, batchSz)
                    # save_images(G_imgs, [nRows,nCols], imgName)
                    #
                    # inv_masked_hat_images = np.multiply(G_imgs, 1.0-mask)
                    # completed = masked_images + inv_masked_hat_images
                    # imgName = os.path.join(config.outDir,
                    #                        'completed/{:04d}.png'.format(i+idx*self.batch_size))
                    # save_images(completed, [nRows,nCols], imgName)
                    #
                    #
                    diffimage = batch_images - G_imgs
                    # imgName = os.path.join(config.outDir,
                    #                        'difference/{:04d}.png'.format(i+idx*self.batch_size))
                    #
                    # save_images(diffimage, [nRows, nCols], imgName)
                    imgName = os.path.join(config.outDir,
                                           'difference/{0:04d}_{1}_0.png'.format(i,idx*self.batch_size))

                    pl.figure(1)
                    pl.clf()
                    pl.imshow(diffimage[0,:,:,0])
                    pl.colorbar()
                    pl.savefig(imgName)

                    imgName = os.path.join(config.outDir,
                                           'completed/{0:04d}_{1}.png'.format(i,idx*self.batch_size))
                    pl.figure(2)
                    pl.clf()
                    pl.imshow(G_imgs[0, :, :, 0])
                    pl.colorbar()
                    pl.savefig(imgName)


                    # print(np.shape(zhats))

                    imgName = os.path.join(config.outDir,
                                           'hats_imgs/{0:04d}_{1}_0_zhats.png'.format(i,idx*self.batch_size))
                    pl.figure(3)
                    pl.clf()
                    pl.imshow(zhats)
                    pl.savefig(imgName)

                    print(np.min(diffimage[:,:,:,0]),np.max(diffimage[:,:,:,0]))


                if config.approach == 'adam':
                    # Optimize single completion with Adam
                    m_prev = np.copy(m)
                    v_prev = np.copy(v)
                    m = config.beta1 * m_prev + (1 - config.beta1) * g[0]
                    v = config.beta2 * v_prev + (1 - config.beta2) * np.multiply(g[0], g[0])
                    m_hat = m / (1 - config.beta1 ** (i + 1))
                    v_hat = v / (1 - config.beta2 ** (i + 1))
                    zhats += - np.true_divide(config.lr * m_hat, (np.sqrt(v_hat) + config.eps))
                    zhats = np.clip(zhats, -1, 1)

                elif config.approach == 'hmc':
                    # Sample example completions with HMC (not in paper)
                    zhats_old = np.copy(zhats)
                    loss_old = np.copy(loss)
                    v = np.random.randn(self.batch_size, self.z_dim)
                    v_old = np.copy(v)

                    for steps in range(config.hmcL):
                        v -= config.hmcEps/2 * config.hmcBeta * g[0]
                        zhats += config.hmcEps * v
                        np.copyto(zhats, np.clip(zhats, -1, 1))
                        loss, g, _ = self.sess.run(run, feed_dict=fd)
                        v -= config.hmcEps/2 * config.hmcBeta * g[0]

                    for img in range(batchSz):
                        logprob_old = config.hmcBeta * loss_old[img] + np.sum(v_old[img]**2)/2
                        logprob = config.hmcBeta * loss[img] + np.sum(v[img]**2)/2
                        accept = np.exp(logprob_old - logprob)
                        if accept < 1 and np.random.uniform() > accept:
                            np.copyto(zhats[img], zhats_old[img])

                    config.hmcBeta *= config.hmcAnneal
                else:
                    assert(False)

            #saving final zhats
            np.save('zhat_completed',zhats)

            for i in range(len(diffimage[:,0,0,0])):
                pl.figure(4)
                pl.clf()
                # pl.imshow(diffimage[i, :, :, 0],interpolation='none',
                #           vmin = -0.05,
                #           vmax = 0.05)
                # vmin=np.min(diffimage[:,:,:,0]),
                # vmax=np.max(diffimage[:,:,:,0]))

                pl.imshow(diffimage[i, :, :, 0], interpolation='none')

                pl.colorbar()

                imgName = os.path.join(config.outDir,
                                       'final/{:04d}.png'.format(frame_idx[i]))
                pl.savefig(imgName)


    def complete_final(self,config):
        def make_dir(name):
            # Works on python 2.7, where exist_ok arg to makedirs isn't available.
            p = os.path.join(config.outDir, name)
            if not os.path.exists(p):
                os.makedirs(p)

        make_dir('final')

        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        isLoaded = self.load(self.checkpoint_dir)
        assert(isLoaded)

        # nImgs = config.batch_size

        # batch_idxs = int(np.ceil(nImgs/self.batch_size))
        # batch_idxs = min(len(self.data[:, 0, 0]), config.train_size) // self.batch_size

        batch = [self.data[i, :, :] for i in range(self.batch_size)]
        batch_images = np.array(batch).astype(np.float32)
        batch_images = np.reshape(batch_images, [self.batch_size, self.image_size, self.image_size, 1])

        if config.zhats != 'none':
            zhats = np.load(config.zhats)

        fd = {
            self.z: zhats,
            self.images: batch_images,
            self.is_training: False
        }
        run = [self.complete_loss, self.grad_complete_loss, self.G]
        loss, g, G_imgs = self.sess.run(run, feed_dict=fd)

        diffimage = batch_images - G_imgs

        for i in range(len(diffimage[:, 0, 0, 0])):
            pl.figure(4)
            pl.clf()
            pl.imshow(diffimage[i, :, :, 0], interpolation='none',
                      vmin=-0.05,
                      vmax=+0.05)
                      # vmin=np.min(diffimage[:, :, :, 0]),
                      # vmax=np.max(diffimage[:, :, :, 0]))
            pl.colorbar()

            imgName = os.path.join(config.outDir,
                                   'final/{:04d}.png'.format(i))
            pl.savefig(imgName)



    def discriminator(self, image, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            print('d img ', np.shape(image))

            h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
            print('d h0 ',np.shape(h0))
            h1 = lrelu(self.d_bns[0](conv2d(h0, self.df_dim*2, name='d_h1_conv'), self.is_training))
            print('d h1 ', np.shape(h1))
            h2 = lrelu(self.d_bns[1](conv2d(h1, self.df_dim*4, name='d_h2_conv'), self.is_training))
            print('d h2 ', np.shape(h2))
            h3 = lrelu(self.d_bns[2](conv2d(h2, self.df_dim*8, name='d_h3_conv'), self.is_training))
            print('d h3 ', np.shape(h3))

            h4 = lrelu(self.d_bns[3](conv2d(h3, self.df_dim * 16, name='d_h4_conv'), self.is_training))
            print('d h4 ', np.shape(h4))

            h5 = lrelu(self.d_bns[4](conv2d(h4, self.df_dim * 32, name='d_h5_conv'), self.is_training))
            print('d h5 ', np.shape(h5))

            hshape = h5.get_shape().as_list()
            print(hshape)

            # h4 = linear(tf.reshape(h3, [-1, 8192]), 1, 'd_h4_lin')
            h6 = linear(tf.reshape(h5, [-1, hshape[1]*hshape[2]*hshape[3]]), 1, 'd_h6_lin')
            print('d h6 ', np.shape(h6))

            return tf.nn.sigmoid(h6), h6

    def generator(self, z):
        with tf.variable_scope("generator") as scope:
            s_h, s_w = self.image_size, self.image_size
            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
            s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)
            s_h32, s_w32 = conv_out_size_same(s_h16, 2), conv_out_size_same(s_w16, 2)
            s_h64, s_w64 = conv_out_size_same(s_h32, 2), conv_out_size_same(s_w32, 2)

            # project `z` and reshape
            self.z_, self.h0_w, self.h0_b = linear(z, self.gf_dim * 32 * s_h64 * s_w64, 'g_h0_lin', with_w=True)

            hs = [None]
            hs[0] = tf.reshape(self.z_, [-1, s_h64, s_w64, self.gf_dim * 32])
            hs[0] = tf.nn.relu(self.g_bns[0](hs[0], self.is_training))
            print('h0 ',np.shape(hs[0]))

            hs.append(None)
            hs[1], _, _ = conv2d_transpose(hs[0], [self.batch_size, s_h32, s_w32, self.gf_dim * 16], name='g_h1',
                                           with_w=True)
            hs[1] = tf.nn.relu(self.g_bns[1](hs[1], self.is_training))
            print('h1 ', np.shape(hs[1]))

            hs.append(None)
            hs[2], _, _ = conv2d_transpose(hs[1], [self.batch_size, s_h16, s_w16, self.gf_dim * 8], name='g_h2',
                                           with_w=True)
            hs[2] = tf.nn.relu(self.g_bns[2](hs[2], self.is_training))
            print('h2 ', np.shape(hs[2]))

            hs.append(None)
            hs[3], _, _ = conv2d_transpose(hs[2], [self.batch_size, s_h8, s_w8, self.gf_dim * 4], name='g_h3',
                                           with_w=True)
            hs[3] = tf.nn.relu(self.g_bns[3](hs[3], self.is_training))
            print('h3 ', np.shape(hs[3]))

            hs.append(None)
            hs[4], _, _ = conv2d_transpose(hs[3], [self.batch_size, s_h4, s_w4, self.gf_dim * 2], name='g_h4', with_w=True)
            print('h4 ', np.shape(hs[4]))

            hs.append(None)
            hs[5], _, _ = conv2d_transpose(hs[4], [self.batch_size, s_h2, s_w2, self.gf_dim * 1], name='g_h5', with_w=True)
            print('h5 ', np.shape(hs[5]))

            hs.append(None)
            hs[6], _, _ = conv2d_transpose(hs[5], [self.batch_size, s_h, s_w, self.c_dim], name='g_h6', with_w=True)
            print('h6 ', np.shape(hs[6]))


            # for normalisations between 0 and 1 use 'tf.nn.sigmoid(hs[4])'

            # return tf.nn.sigmoid(hs[4])
            return tf.nn.tanh(hs[6])



    def save(self, checkpoint_dir, step):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, self.model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            return True
        else:
            return False
