from __future__ import division
import os
import time
import tensorflow as tf
import numpy as np
from six.moves import xrange
from collections import namedtuple
from scipy import misc

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
        self.with_flip=args.flip
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
        if args.phase=='train':
            self._build_model()
            self.saver = tf.train.Saver(max_to_keep=2)
            self.pool = ImagePool(args.max_size)

    def _build_model(self):
        # self.real_data = tf.placeholder(tf.float32,
        #                                 [None, self.image_size, self.image_size,
        #                                  self.input_c_dim + self.output_c_dim],
        #                                 name='real_A_and_B_images')
    
        immy_a,_ ,_= self.build_input_image_op(os.path.join(self.dataset_dir,'trainA'),False)
        immy_b,_ ,_= self.build_input_image_op(os.path.join(self.dataset_dir,'trainB'),False)
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

        immy_test_a,path_a,_ = self.build_input_image_op(os.path.join(self.dataset_dir,'testA'),True)
        immy_test_b,path_b,_ = self.build_input_image_op(os.path.join(self.dataset_dir,'testB'),True)

        self.test_A,self.test_path_a = tf.train.batch([immy_test_a,path_a],1,2,100)
        self.test_B,self.test_path_b = tf.train.batch([immy_test_b,path_b],1,2,100)

        self.testB = self.generator(self.test_A, self.options, True, name="generatorA2B")
        self.testA = self.generator(self.test_B, self.options, True, name="generatorB2A")

        t_vars = tf.trainable_variables()
        self.db_vars = [var for var in t_vars if 'discriminatorB' in var.name]
        self.da_vars = [var for var in t_vars if 'discriminatorA' in var.name]
        self.g_vars_a2b = [var for var in t_vars if 'generatorA2B' in var.name]
        self.g_vars_b2a = [var for var in t_vars if 'generatorB2A' in var.name]
        # for var in t_vars: print(var.name)

    def build_input_image_op(self,dir,is_test=False,num_epochs=None):
        samples = tf.train.match_filenames_once(dir+'/*.jpg')
        filename_queue = tf.train.string_input_producer(samples,num_epochs=num_epochs)
        sample_path = filename_queue.dequeue()
        image_raw = tf.read_file(sample_path)
        image = tf.image.decode_image(image_raw, channels=3)
        image.set_shape([None, None, self.input_c_dim])
        im_shape=tf.shape(image)

        #change range of value o [-1,1]
        image = tf.image.convert_image_dtype(image,tf.float32)
        image = (image*2)-1
        
        if not is_test:
            #resize to load_size
            image = tf.image.resize_images(image,[self.load_size,self.load_size])

            #crop fine_size
            image = tf.random_crop(image,[self.image_size,self.image_size,3])

            #random flip left right
            if self.with_flip:
                image = tf.image.random_flip_left_right(image)
        else:
            image = tf.image.resize_images(image,[self.image_size,self.image_size])

        return image,sample_path,im_shape

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
        def get_var_to_restore_list(ckpt_path, mask=[], prefix=""):
            """
            Get all the variable defined in a ckpt file and add them to the returned var_to_restore list. Allows for partially defined model to be restored fomr ckpt files.
            Args:
                ckpt_path: path to the ckpt model to be restored
                mask: list of layers to skip
                prefix: prefix string before the actual layer name in the graph definition
            """
            variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            variables_dict = {}
            for v in variables:
                name = v.name[:-2]
                skip=False
                #check for skip
                for m in mask:
                    if m in name:
                        skip=True
                        continue
                if not skip:
                    variables_dict[v.name[:-2]] = v
            #print(variables_dict)
            reader = tf.train.NewCheckpointReader(ckpt_path)
            var_to_shape_map = reader.get_variable_to_shape_map()
            var_to_restore = {}
            for key in var_to_shape_map:
                #print(key)
                if prefix+key in variables_dict.keys():
                    var_to_restore[key] = variables_dict[prefix+key]
            return var_to_restore

        print(" [*] Reading checkpoint...")

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            savvy = tf.train.Saver(var_list=get_var_to_restore_list(ckpt.model_checkpoint_path))
            savvy.restore(self.sess, ckpt.model_checkpoint_path)
            return True
        else:
            return False

    def test(self, args):
        """Test cyclegan""" 
        sample_op, sample_path,im_shape = self.build_input_image_op(self.dataset_dir,is_test=True,num_epochs=1)
        sample_batch,path_batch,im_shapes = tf.train.batch([sample_op,sample_path,im_shape],batch_size=self.batch_size,num_threads=4,capacity=self.batch_size*50,allow_smaller_final_batch=True)
        gen_name='generatorA2B' if args.which_direction=="AtoB" else 'generatorB2A'
        cycle_image_batch = self.generator(sample_batch,self.options,name=gen_name)

        #init everything
        self.sess.run([tf.global_variables_initializer(),tf.local_variables_initializer()])

        #start queue runners
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners()
        print('Thread running')

        if self.load(args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # write html for visual comparison
        if not os.path.exists(args.test_dir): #python 2 is dumb...
            os.makedirs(args.test_dir)

        index_path = os.path.join(args.test_dir, '{0}_index.html'.format(args.which_direction))
        index = open(index_path, "w+")
        index.write("<html><body><table><tr>")
        index.write("<th>name</th><th>input</th><th>output</th></tr>")

        print('Starting')
        batch_num=0
        while True:
            try:
                print('Processed images: {}'.format(batch_num*args.batch_size), end='\r')
                fake_imgs,sample_images,sample_paths,im_sps = self.sess.run([cycle_image_batch,sample_batch,path_batch,im_shapes])
                #iterate over each sample in the batch
                for rr in range(fake_imgs.shape[0]):
                    #create output destination
                    dest_path = sample_paths[rr].decode('UTF-8').replace(self.dataset_dir,args.test_dir)
                    parent_destination = os.path.abspath(os.path.join(dest_path, os.pardir))
                    if not os.path.exists(parent_destination):
                        os.makedirs(parent_destination)

                    fake_img = ((fake_imgs[rr]+1)/2)*255
                    im_sp = im_sps[rr]
                    fake_img = misc.imresize(fake_img,(im_sp[0],im_sp[1]))
                    misc.imsave(dest_path,fake_img)
                    index.write("<td>%s</td>" % os.path.basename(sample_paths[rr].decode('UTF-8')))
                    index.write("<td><img src='%s'></td>" % (sample_paths[rr].decode('UTF-8')))
                    index.write("<td><img src='%s'></td>" % (dest_path))
                    index.write("</tr>")
                batch_num+=1
            except Exception as e:
                print(e)
                break;

        print('Elaboration complete')
        index.close()
        coord.request_stop()
        coord.join(stop_grace_period_secs=10)