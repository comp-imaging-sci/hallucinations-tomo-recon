"""
Copyright (c) Computational Imaging Science Lab @ UIUC (2021)
Author      : Varun Kelkar, Sayantan Bhadra
Email       : vak2@illinois.edu, sayantanbhadra@wustl.edu
"""

import sys,os
import numpy as np
import numpy.fft as fft
import tensorflow as tf
from numpy import linalg as LA
from shutil import copy2
import scipy
import scipy.io as spio
import argparse
import imageio as io
import importlib

parser = argparse.ArgumentParser()


# Define the pixelnorm function to normalize the latent vector
def pixelnorm_np(x, epsilon=1e-8):
        return x / (LA.norm(x,2)+epsilon)

# Dimensions of k-space
dim = 320

# Dimensions of latent space Z
Z_dim = 512

# Input the hyperparameters from user arguments
parser.add_argument("--snr", type=float, default=20, help="Measurement SNR")
parser.add_argument("--mask_type", type=str, default='mask', help="Name of the mask to be used.")
parser.add_argument("--learning_rate", type=float, default=1.e-3, help="Learning rate")
parser.add_argument("--reg_TV", type=float, default=0, help="TV regularization on the image")
parser.add_argument("--T_in", type=int, default=30000, help="Total number of iterations")
parser.add_argument("--iter_intvl", type=int, default=500, help="Save after these number of iterations")
parser.add_argument("--date", type=str, default='', help="Date")
parser.add_argument("--process", type=int, default=0, help="Condor process index")
parser.add_argument("--gt_filename", type=str, default='', help="Path to the ground truth file")
parser.add_argument("--meas_filename", type=str, default='', help="Path to the ground truth measurements")
parser.add_argument("--results_dir", type=str, default='', help="Results_dir")
parser.add_argument("--net_module", type=str, default='unet', help="Model to be used for DIP")
parser.add_argument("--initialization", type=str, default='random_normal', help="Initialization type")
args = parser.parse_known_args()[0]
print(args)

net_module = importlib.import_module(args.net_module)
parser = net_module.add_arguments(parser)
args = parser.parse_args()
print(args)

# Load the sampling mask
full = 0
if full:
    mask_lines = np.ones([dim])
else:
    #mask_lines = np.fromfile('/home/sayantan/NYU_data/generated_kspace_with_mask_4fold/mask_[4]fold_25.dat',dtype=np.int32)
    mask = np.load(f'../recon_data/{args.mask_type}.npy')
    mask = fft.ifftshift(mask)

# Load the ground truth and compute the measurements
ground_truth = np.load(args.gt_filename)
ground_truth = ground_truth.reshape(dim,dim)
# y = np.fft.fft2(ground_truth) / dim
# y = utils.add_noise(y, args.snr)
# y = y * mask

# Load the k-space
sgn = (-1)**np.arange(dim)
sgn = np.stack([sgn]*dim, axis=0)
sgn[1::2] = -sgn[1::2]
kspace = np.load(args.meas_filename)
y = np.fft.ifftshift(kspace) * sgn
y = y * mask


# Load the Z_init from performing CSGM
if args.initialization == 'random_normal':
    Z_init_np = np.random.RandomState(0).randn(1,dim,dim,1).astype(np.float32)
elif args.initialization == 'pinv':
    Z_init_np = np.fft.ifft2(y * dim).astype(np.float32)


# Directory to store results
model_name = f'est-mask_{args.mask_type}-SNR{args.snr}-lr{args.learning_rate:.1e}-regtv{args.reg_TV:.1e}-iter{args.T_in}'
# results_dir = '/shared/einstein/MRI/hallucination_map_data/tumorsim_hallucination/results_dip/sweep'
results_dir = args.results_dir
results_dir = os.path.join(results_dir, model_name)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

np.save(os.path.join(results_dir,'ymeas.npy'), y.astype(np.complex64))

# Copy details to txt file
# copy2('/shared/radon/MRI/INN_analysis/image_adaptive_gan_results/codes_for_IAGAN/IAGAN_TV_from_CSGM.py',results_dir+'/model_info.txt')

# CSGM training parameters and hyperparameters
# reg = 0
# T_in = 20000
# Initialize loss arrays
loss_recon = np.zeros([0])

with tf.Session() as sess:

    g = tf.placeholder(tf.complex64,shape=[dim,dim])

    net = net_module.Net(args)
    Z = tf.constant(Z_init_np, dtype=tf.float32, shape=[1,dim,dim,1])
    f_init = net(Z)

    # G_z = (1+f_init)/2
    G_z = f_init
    # G_z = tf.complex(G_z[...,0], G_z[...,1])
    G_z = tf.reshape(G_z,[dim,dim])
    G_z = tf.cast(G_z,dtype=tf.complex64)
    g_hat = tf.fft2d(G_z) / dim
    g_hat = g_hat * mask
    lsq_loss = tf.reduce_sum(tf.square(tf.abs(g-g_hat)))
    TV_loss = args.reg_TV*tf.reduce_sum(tf.image.total_variation(tf.expand_dims(G_z,2)))
    total_loss = lsq_loss + TV_loss

    print(net.trainable_variables)
    total_loss_solver = tf.train.AdamOptimizer(learning_rate=args.learning_rate,beta1=0.9).minimize(total_loss, var_list=[net.trainable_variables])
    #var_list = [var for var in tf.global_variables() if 'Momentum' in var.name]
    #Z_init = tf.variables_initializer(Z)
    var_list_opt = [var for var in tf.global_variables() if 'Adam' in var.name or 'beta1_power' in var.name or 'beta2_power' in var.name]
    var_list_init = tf.variables_initializer(var_list_opt)
    
    sess.run(tf.initialize_all_variables()) 
    sess.run(var_list_init) 

    txtfile = open(os.path.join(results_dir, 'loss_recon.txt'), 'a')
    for t_in in range(args.T_in):
        _, lsq_loss_curr, TV_loss_curr, total_loss_curr = sess.run([total_loss_solver, lsq_loss, TV_loss, total_loss], feed_dict={g:y})
        #if t_in>0:
        #	if abs(total_loss_curr-total_loss_prev)<1e-20:
        #		print('Stopped at iter '+str(t_in))
        #		break
        total_loss_prev = total_loss_curr
        if t_in % args.iter_intvl == 0:
            loss_recon = np.append(loss_recon,total_loss_curr)
            Gzest = sess.run(G_z) 
            recon_error = LA.norm(Gzest - ground_truth) / dim
            print('Iter = %d, Total loss = %g, Least squares loss = %g, TV_loss = %g, recon_Error = %g, '%(t_in,total_loss_curr,lsq_loss_curr, TV_loss_curr, recon_error))
            print('Iter = %d, Total loss = %g, Least squares loss = %g, TV_loss = %g, recon_Error = %g, '%(t_in,total_loss_curr,lsq_loss_curr, TV_loss_curr, recon_error), file=txtfile)
            # np.save(os.path.join(results_dir,'Z_'+str(t_in)+'.npy'), sess.run(Z))
            np.save(os.path.join(results_dir,'recon_'+str(t_in)+'.npy'), Gzest)
            Gzest = np.clip(Gzest*255, 0, 255).astype(np.uint8)
            io.imsave(os.path.join(results_dir,'recon_'+str(t_in)+'.png'), Gzest)
                    
print('Done!')




