import sys
import os
import time
from datetime import datetime
import importlib # importlib.reload(utils)
import pprint

import tensorflow as tf
import numpy as np
from sigpy import upsample
import scipy.io as sio
import math
from torchkbnufft import KbNufft, KbNufftAdjoint

import matplotlib.pyplot as plt
from matplotlib import animation

import utils
import tf_utils
import parser_ops
#import UnrollNet

parser = parser_ops.get_parser()
args = parser.parse_args()
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # set it to the # of available GPUs

save_dir ='saved_models'
directory = os.path.join(save_dir, 'SSDU_' +
                         args.data_opt + '_' +
                         str(args.epochs) + 'Epochs_Rate' +
                         str(args.acc_rate) + '_' +
                         str(args.nb_unroll_blocks) + 'Unrolls_'
                         + args.mask_type + 'Selection')

if not os.path.exists(directory):
    os.makedirs(directory)

print('\n create a test model for the testing')
test_graph_generator = tf_utils.test_graph(directory)

#..............................................................................
start_time = time.time()
print('.....................SSDU Training.....................')
tf.reset_default_graph()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True

# .......................Load the Data.........................................
print('\n Loading ', args.data_opt, ' data, acc rate : ', args.acc_rate, 
      ', mask type :', args.mask_type)

kspace_dir, coil_dir, mask_dir = utils.get_train_directory(args)

# %% kspace and sensitivity maps are assumed to be in .h5 format and mask is assumed to be in .mat
# Users can change these formats based on their dataset
kspace_train = h5py.File(kspace_dir, "r")['kspace'][:]
sens_maps = h5py.File(coil_dir, "r")['sens_maps'][:]
original_mask = sio.loadmat(mask_dir)['mask']