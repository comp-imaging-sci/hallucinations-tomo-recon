# Compute raw error map, measurement space hallucination map and null space hallucination map
# Note that for display and further analysis the absolute value is taken after computing each map
import numpy as np 
import os,sys
sys.path.append('../')
from utils import *
import argparse

# Function for computing the measurement space hallucination map
def meas_hm(recon,pinv_g):
    recon_meas,_ = f_meas_null(recon)
    h_map = recon_meas - pinv_g
    return h_map

# Function for computing the null space hallucination map
def null_hm(recon,gt):
    _, recon_null = f_meas_null(recon)
    _, gt_null = f_meas_null(gt)
    h_map = recon_null - gt_null
    h_map[recon_null==0]=0
    return h_map

parser = argparse.ArgumentParser()
parser.add_argument("--recon-type",choices=['UNET','PLSTV','DIP'],required=True,help="Type of reconstruction method")
parser.add_argument("--dist-type",choices=['ind,ood'],required=True,help="Type of data distribution")
parser.add_argument("--map-type",choices=['em,meas_hm,null_hm'],required=True,help="Type of map")
parser.add_argument("--idx",type=int,default=0,required=True,help="Which image number")
args = parser.parse_args()

recon_type = args.recon_type
dist_type = args.dist_type
map_type = args.map_type
idx = args.idx

# Load ground truth for error map and null space hallucination map
if map_type == 'em' or map_type == 'meas_hm':
    gt = np.load('../recon_data/gt_'+str(idx)+'.npy')

# Load the k-space data g for computing measurement space hallucination map
if map_type == 'meas_hm':
    kspace_dir = '../recon_data/kspace_'+dist_type+'/'
    g = np.load(kspace_dir+'kspace_'+str(idx))

# Load the reconstructed image
recon = '../'+recon_type+'/recons_'+dist_type+'/recon_'+str(idx)+'.npy'

# Directory for saving the raw map
map_dir = './'+recon_type+'_'+map_type+'_'+dist_type+'/'
if not os.path.exists(map_dir):
    os.makedirs(map_dir)

# Compute the map
if map_type == 'em':
    map = np.abs(recon-gt)
elif map_type == 'meas_hm':
    map = np.abs(meas_hm(recon,pinv_g))
else: # map_type == 'null_hm'
    map = np.abs(null_hm(recon,gt))

# Save the map
np.save(map_dir+'map_'+str(idx)+'.npy',map)










