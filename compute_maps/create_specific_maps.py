# Create specific map from raw map
import numpy as np 
from PIL import Image
from skimage import exposure
from skimage import morphology
from skimage.measure import label
from scipy.ndimage import gaussian_filter
import os
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--recon-type",choices=['UNET','PLSTV','DIP'],required=True,help="Type of reconstruction method")
parser.add_argument("--dist-type",choices=['ind,ood'],required=True,help="Type of data distribution")
parser.add_argument("--map-type",choices=['em,null_hm'],required=True,help="Type of map")
parser.add_argument("--idx",type=int,default=0,required=True,help="Which image number")
args = parser.parse_args()

recon_type = args.recon_type
dist_type = args.dist_type
map_type = args.map_type
idx = args.idx

def convert_to_uint(img):
    img = 255 * (img-img.min())/(img.max()-img.min())
    return img.astype(np.uint8)

def threshold(img,th_p):
    th_value = np.percentile(img,th_p)
    return (img>th_value).astype(np.uint8)

raw_map_dir = './'+recon_type+'_'+map_type+'_'+dist_type+'/'
seg_mask_dir = '../recon_data/seg_mask_'+dist_type+'/'
specific_map_dir = './'+recon_type+'_specific_'+map_type+'_'+dist_type+'/'
if not os.path.exists(specific_map_dir):
    os.makedirs(specific_map_dir)

# Specific map processing parameters
sigma = 7.0 # Gaussian blur width
th_p = 95 # Global threshold percentile value for intensity cut-off
small_tol = 100 # Remove detected regions for which total number of pixels is less than this value

# Create the binary specific map from the raw map and save as an image file
map = np.load(raw_map_dir+'map_'+str(idx)+'.npy')

# Step 1: Convert the raw map to uint8 datatype within the range [0,255]
map = convert_to_uint(map)

# Step 2: Load segmentation mask and apply the mask on the raw map
seg_mask = np.load(seg_mask_dir+'sm_'+str(idx)+'.npy')
map = map * seg_mask

# Step 3: Perform histogram equalization
map = exposure.equalize_hist(map)

# Step 4: Apply Gaussian blur
map = gaussian_filter(map,sigma=sigma)

# Step 5: Peform thresholding
map = threshold(map,th_p=th_p)

# Step 6: Remove very small detected regions and obtain the final specific map
map_labels = label(map,return_num=False)
map_labels = morphology.remove_small_objects(map_labels,small_tol)
map_labels[map_labels>0] = 1

# Save specific map as png
map_labels = np.uint8(255*map_labels)
specific_map_im = Image.fromarray(map_labels)
specific_map_im.save(specific_map_dir+'sp_map_'+str(idx)+'.png')






