"""
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

parser = argparse.ArgumentParser()
parser.add_argument("--dist-type",type=str,help="Type of data distribution",default="ind")
args = parser.parse_args()

dim = 320 # Image dimensions: dimxdim
num_recons = 69 # Number of k-space measurements
recon_dir = './recons/' # Directory for saving the reconstructed images
if not os.path.exists(recon_dir):
    os.makedirs(recon_dir)

# Load the mask
mask = np.load('../mask.npy')

# k-space directory
kspace_dir = '../recon_data/kspace_'+dist_type+'/'

# Regularization parameter
lmda = '0.02'

# Sensitivity map (all ones)
smap = np.ones([dim,dim],np.complex64)

for idx in range(num_recons):
    print('Recon '+str(idx))
    kspace = np.load(kspace_dir+'kspace_'+str(idx)+'.npy')
    kspace = kspace * dim
    kspace = kspace * mask
    #recon = bart(1,'pics -d2 -i200 -g G0 -S -R T:7:0:'+lmda,kspace,smap) # If running on GPU
    recon = bart(1,'pics -d2 -i200 -S -R T:7:0:'+lmda,kspace,smap) # If running on CPU
    recon = np.abs(recon)/dim
    np.save(recon_dir+'recon_'+str(idx)+'.npy',recon)




