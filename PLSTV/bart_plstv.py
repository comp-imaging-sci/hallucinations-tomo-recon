"""

Copyright (c) Computational Imaging Science Lab @ UIUC (2021)
Author      : Sayantan Bhadra
Email       : sayantanbhadra@wustl.edu

Script for performing PLS-TV reconstruction using the BART toolbox.
See intructions for installing the BART toolbox here: https://mrirecon.github.io/bart/
Recommended BART version >= 0.5.00

Before running this script, export the following variables to the current shell environment:
export TOOLBOX_PATH=/path/to/bart/
export PATH=$TOOLBOX_PATH:$PATH

"""
import os,sys
import numpy as np
path = os.environ["TOOLBOX_PATH"] + "/python/"
sys.path.append(path)
from bart import bart
import argparse
import cfl
from PIL import Image

# Function for converting float32 image array to uint8 array in the range [0,255]
def convert_to_uint(img):
    img = 255 * (img-img.min())/(img.max()-img.min())
    return img.astype(np.uint8)

parser = argparse.ArgumentParser()
parser.add_argument("--dist-type",choices=['ind','ood'],required=True,help="Type of data distribution")
parser.add_argument("--idx",type=int,default=0,required=True,help="Which image number")
args = parser.parse_args()
dist_type = args.dist_type
idx = args.idx

dim = 320 # Image dimensions: dimxdim
recon_dir = './recons_'+dist_type+'/' # Directory for saving the reconstructed images
if not os.path.exists(recon_dir):
    os.makedirs(recon_dir)

# Load the mask
mask = np.load('../recon_data/mask.npy')

# k-space directory
kspace_dir = '../recon_data/kspace_'+dist_type+'/'

# Regularization parameter
lmda = '0.02'

# Sensitivity map (all ones)
smap = np.ones([dim,dim],np.complex64)

# Perform PLS-TV reconstruction
kspace = np.load(kspace_dir+'kspace_'+str(idx)+'.npy')
kspace = kspace * dim
kspace = kspace * mask
#recon = bart(1,'pics -d2 -i200 -g G0 -S -R T:7:0:'+lmda,kspace,smap) # If running on GPU (BART must be compiled with CUDA during installation)
recon = bart(1,'pics -d2 -i200 -S -R T:7:0:'+lmda,kspace,smap) # If running on CPU
recon = np.abs(recon)/dim
np.save(recon_dir+'recon_'+str(idx)+'.npy',recon)
recon_im = Image.fromarray(convert_to_uint(recon))
recon_im.save(recon_dir+'recon_'+str(idx)+'.png')




