#!/usr/bin/env python3

# Original Version: Taehoon Kim (http://carpedm20.github.io)
#   + Source: https://github.com/carpedm20/DCGAN-tensorflow/blob/e30539fb5e20d5a0fed40935853da97e9e55eee8/main.py
#   + License: MIT
# [2016-08-05] Modifications for Inpainting: Brandon Amos (http://bamos.github.io)
#   + License: MIT

import os
import scipy.misc
import numpy as np

from model import DCGAN
from utils import pp, visualize, to_json

import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer("epoch", 200, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", 94, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 5, "The size of batch images [64]")
flags.DEFINE_integer("image_size", 64, "The size of image to use")
flags.DEFINE_string("dataset", "lfw-aligned-64", "Dataset directory.")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")

flags.DEFINE_string("outDir","testout","output directory")
flags.DEFINE_string("maskType","random","Mask type")
flags.DEFINE_integer("nIter",10, "number of completion iterations")
flags.DEFINE_string("approach","adam","optimisation scheme")
flags.DEFINE_integer("imgs",10,"number of images to complete")
flags.DEFINE_integer("outInterval",1, "output interval")
FLAGS = flags.FLAGS

if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
if not os.path.exists(FLAGS.sample_dir):
    os.makedirs(FLAGS.sample_dir)

# config = tf.ConfigProto(device_count={'GPU': 0 , 'CPU': 25})
config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    dcgan = DCGAN(sess, data_path=FLAGS.dataset,image_size=FLAGS.image_size, batch_size=FLAGS.batch_size,
                  is_crop=False, checkpoint_dir=FLAGS.checkpoint_dir)

    dcgan.train(FLAGS)
    # dcgan.complete(FLAGS)
