from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange
from collections import namedtuple

from module import *
from utils import *


class cyclegan(object):
    def __init__(self, sess, args):
        self.sess = sess
        self.batch_size = args.batch_size
        self.image_size = args.fine_size
        self.load_size = args.load_size
        self.input_c_dim = args.input_nc
        self.output_c_dim = args.output_nc
        self.L1_lambda = args.L1_lambda
        self.dataset_dir = args.dataset_dir
        self.dataset_name=self.dataset_dir.split('/')[-1]

        self.discriminator = discriminator
        if args.use_resnet:
            self.generator = generator_resnet
        else:
            self.generator = generator_unet
        if args.use_lsgan:
            self.criterionGAN = mae_criterion
        else:
            self.criterionGAN = sce_criterion

        OPTIONS = namedtuple('OPTIONS', 'batch_size image_size \
                              gf_dim df_dim output_c_dim')
        self.options = OPTIONS._make((args.batch_size, args.fine_size,
                                      args.ngf, args.ndf, args.output_nc))

        self._build_model()
        self.saver = tf.train.Saver()
        self.pool = ImagePool(args.max_size)

    def _build_model(self):
        # self.real_data = tf.placeholder(tf.float32,
        #                                 [None, self.image_size, self.image_size,
        #                                  self.input_c_dim + self.output_c_dim],
        #                                 name='real_A_and_B_images')

        immy_a = self.build_input_image_op(self.dataset_dir+'/trainA',False)
        immy_b = self.build_input_image_op(self.dataset_dir+'/trainB',False)
        self.real_A,self.real_B = tf.train.shuffle_batch([immy_a,immy_b],self.batch_size,1000,600,8)

        self.fake_B = self.generator(self.real_A, self.options, False, name="generatorA2B")
        self.fake_A_ = self.generator(self.fake_B, self.options, False, name="generatorB2A")
        self.fake_A = self.generator(self.real_B, self.options, True, name="generatorB2A")
        self.fake_B_ = self.generator(self.fake_A, self.options, True, name="generatorA2B")

        self.DB_fake = self.discriminator(self.fake_B, self.options, reuse=False, name="discriminatorB")
        self.DA_fake = self.discriminator(self.fake_A, self.options, reuse=False, name="discriminatorA")
        self.g_loss_a2b = self.criterionGAN(self.DB_fake, tf.ones_like(self.DB_fake)) \
                          + self.L1_lambda * abs_criterion(self.real_A, self.fake_A_) \
                          + self.L1_lambda * abs_criterion(self.real_B, self.fake_B_)
        self.g_loss_b2a = self.criterionGAN(self.DA_fake, tf.ones_like(self.DA_fake)) \
                          + self.L1_lambda * abs_criterion(self.real_A, self.fake_A_) \
                          + self.L1_lambda * abs_criterion(self.real_B, self.fake_B_)

        self.fake_A_sample = tf.placeholder(tf.float32,
                                            [None, self.image_size, self.image_size,
                                             self.input_c_dim], name='fake_A_sample')
        self.fake_B_sample = tf.placeholder(tf.float32,
                                            [None, self.image_size, self.image_size,
                                             self.output_c_dim], name='fake_B_sample')
        self.DB_real = self.discriminator(self.real_B, self.options, reuse=True, name="discriminatorB")
        self.DA_real = self.discriminator(self.real_A, self.options, reuse=True, name="discriminatorA")
        self.DB_fake_sample = self.discriminator(self.fake_B_sample, self.options, reuse=True, name="discriminatorB")
        self.DA_fake_sample = self.discriminator(self.fake_A_sample, self.options, reuse=True, name="discriminatorA")
        self.db_loss_real = self.criterionGAN(self.DB_real, tf.ones_like(self.DB_real))
        self.db_loss_fake = self.criterionGAN(self.DB_fake_sample, tf.zeros_like(self.DB_fake_sample))
        self.db_loss = (self.db_loss_real + self.db_loss_fake) / 2
        self.da_loss_real = self.criterionGAN(self.DA_real, tf.ones_like(self.DA_real))
        self.da_loss_fake = self.criterionGAN(self.DA_fake_sample, tf.zeros_like(self.DA_fake_sample))
        self.da_loss = (self.da_loss_real + self.da_loss_fake) / 2

        self.g_a2b_sum = tf.summary.scalar("g_loss_a2b", self.g_loss_a2b)
        self.g_b2a_sum = tf.summary.scalar("g_loss_b2a", self.g_loss_b2a)
        self.db_loss_sum = tf.summary.scalar("db_loss", self.db_loss)
        self.da_loss_sum = tf.summary.scalar("da_loss", self.da_loss)
        self.db_loss_real_sum = tf.summary.scalar("db_loss_real", self.db_loss_real)
        self.db_loss_fake_sum = tf.summary.scalar("db_loss_fake", self.db_loss_fake)
        self.da_loss_real_sum = tf.summary.scalar("da_loss_real", self.da_loss_real)
        self.da_loss_fake_sum = tf.summary.scalar("da_loss_fake", self.da_loss_fake)
        self.db_sum = tf.summary.merge(
            [self.db_loss_sum, self.db_loss_real_sum, self.db_loss_fake_sum]
        )
        self.da_sum = tf.summary.merge(
            [self.da_loss_sum, self.da_loss_real_sum, self.da_loss_fake_sum]
        )

        immy_test_a = self.build_input_image_op(self.dataset_dir+'/testA',True)
        immy_test_b = self.build_input_image_op(self.dataset_dir+'/testB',True)

        self.test_A,self.test_B = tf.train.shuffle_batch([immy_test_a,immy_test_b],1,500,100,2)

        self.testB = self.generator(self.test_A, self.options, True, name="generatorA2B")
        self.testA = self.generator(self.test_B, self.options, True, name="generatorB2A")

        t_vars = tf.trainable_variables()
        self.db_vars = [var for var in t_vars if 'discriminatorB' in var.name]
        self.da_vars = [var for var in t_vars if 'discriminatorA' in var.name]
        self.g_vars_a2b = [var for var in t_vars if 'generatorA2B' in var.name]
        self.g_vars_b2a = [var for var in t_vars if 'generatorB2A' in var.name]
        for var in t_vars: print(var.name)

    def build_input_image_op(self,dir,is_test=False):
        samples = tf.train.match_filenames_once(dir+'/*.*')
        filename_queue = tf.train.string_input_producer(samples)
        image_raw = tf.read_file(filename_queue.dequeue())
        image = tf.image.decode_image(image_raw, channels=3)
        image.set_shape([None, None, 3])

        #change range of value o [-1,1]
        image = tf.image.convert_image_dtype(image,tf.float32)
        image = (image*2)-1
        
        if not is_test:
            #resize to load_size
            image = tf.image.resize_images(image,[self.load_size,self.load_size])

            #crop fine_size
            image = tf.random_crop(image,[self.image_size,self.image_size,3])

            #random flip left right
            image = tf.image.random_flip_left_right(image)
        else:
            image = tf.image.resize_images(image,[self.image_size,self.image_size])

        return image

    def train(self, args):
        """Train cyclegan"""
        self.da_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
            .minimize(self.da_loss, var_list=self.da_vars)
        self.db_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
            .minimize(self.db_loss, var_list=self.db_vars)
        self.g_a2b_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
            .minimize(self.g_loss_a2b, var_list=self.g_vars_a2b)
        self.g_b2a_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
            .minimize(self.g_loss_b2a, var_list=self.g_vars_b2a)

        image_summaries = []

        #summaries for training
        tf.summary.image('A_Real',self.real_A)
        tf.summary.image('A_to_B',self.fake_B)
        tf.summary.image('B_Real',self.real_B)
        tf.summary.image('B_to_A',self.fake_A)
        tf.summary.image('A_to_B_to_A',self.fake_A_)
        tf.summary.image('B_to_A_to_B',self.fake_B_)

        #summaries for test sample
        tf.summary.image('A_to_B_sample',self.testB)
        tf.summary.image('B_to_A_sample',self.testA)
        
        init_op = [tf.global_variables_initializer(),tf.local_variables_initializer()]
        self.sess.run(init_op)
        self.writer = tf.summary.FileWriter(args.checkpoint_dir, self.sess.graph)

        summary_op = tf.summary.merge_all()

        counter = 0
        start_time = time.time()

        if self.load(args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        coord = tf.train.Coordinator()
        tf.train.start_queue_runners()
        print('Thread running')

        for epoch in range(args.epoch):
            print('Start epoch: {}'.format(epoch))
            batch_idxs = args.num_sample

            for idx in range(0, batch_idxs):

                # Forward G network
                fake_A, fake_B = self.sess.run([self.fake_A, self.fake_B])
                [fake_A, fake_B] = self.pool([fake_A, fake_B])

                # Update G network + Update D network
                self.sess.run([self.g_a2b_optim,self.db_optim,self.g_b2a_optim,self.da_optim],feed_dict={self.fake_A_sample:fake_A,self.fake_B_sample:fake_B})
                
                counter += 1
                print(("Epoch: [%2d] [%4d/%4d] time: %4.4f" \
                       % (epoch, idx, batch_idxs, time.time() - start_time)))

                if np.mod(counter, 10) == 1:
                    summary_string = self.sess.run(summary_op,feed_dict={self.fake_A_sample:fake_A,self.fake_B_sample:fake_B})
                    self.writer.add_summary(summary_string,counter)

                if np.mod(counter, 1000) == 2:
                    self.save(args.checkpoint_dir, counter)

        coord.request_stop()
        coord.join(stop_grace_period_secs=10)

    def save(self, checkpoint_dir, step):
        model_name = "%s_%s" % (self.dataset_name, self.image_size)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        model_name = "%s_%s" % (self.dataset_name, self.image_size)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            return True
        else:
            return False

    def test(self, args):
        """Test cyclegan"""
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        if args.which_direction == 'AtoB':
            sample_files = glob('{}/*.*'.format(self.dataset_dir + '/testA'))
        elif args.which_direction == 'BtoA':
            sample_files = glob('{}/*.*'.format(self.dataset_dir + '/testB'))
        else:
            raise Exception('--which_direction must be AtoB or BtoA')

        if self.load(args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # write html for visual comparison
        index_path = os.path.join(args.test_dir, '{0}_index.html'.format(args.which_direction))
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        index.write("<th>name</th><th>input</th><th>output</th></tr>")

        out_var, in_var = (self.testB, self.test_A) if args.which_direction == 'AtoB' else (
            self.testA, self.test_B)

        for sample_file in sample_files:
            print('Processing image: ' + sample_file)
            sample_image = [load_test_data(sample_file)]
            sample_image = np.array(sample_image).astype(np.float32)
            image_path = os.path.join(args.test_dir,
                                      '{0}_{1}'.format(args.which_direction, os.path.basename(sample_file)))
            fake_img = self.sess.run(out_var, feed_dict={in_var: sample_image})
            save_images(fake_img, [1, 1], image_path)
            index.write("<td>%s</td>" % os.path.basename(image_path))
            index.write("<td><img src='%s'></td>" % (sample_file if os.path.isabs(sample_file) else (
            '..' + os.path.sep + sample_file)))
            index.write("<td><img src='%s'></td>" % (image_path if os.path.isabs(image_path) else (
            '..' + os.path.sep + image_path)))
            index.write("</tr>")
        index.close()
