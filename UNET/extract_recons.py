# Extract reconstructed test images from the U-Net method
import h5py
import os
import numpy as np 

model = 'h_map'
num_recons = 5

recon_dir = './experiments/'+model+'/reconstructions/'

ensemble_dir_ind = './recons_ind/'
if not os.path.exists(ensemble_dir_ind):
    os.makedirs(ensemble_dir_ind)
ensemble_dir_ood = './recons_ood/'
if not os.path.exists(ensemble_dir_ood):
    os.makedirs(ensemble_dir_ood)

# Ensemble of ind recons
print('Start saving ind recons\n')
filename = recon_dir+'file_ind.h5'
recon = h5py.File(filename,'r')['reconstruction']
recon = np.asarray(recon)
for idx in range(num_recons):
    recon_slice = recon[idx]
    np.save(ensemble_dir_ind+'recon_'+str(idx)+'.npy',recon_slice)

# Ensemble of ood recons
print('Start saving ood recons\n')
filename = recon_dir+'file_ood.h5'
recon = h5py.File(filename,'r')['reconstruction']
recon = np.asarray(recon)
for idx in range(num_recons):
    recon_slice = recon[idx]
    np.save(ensemble_dir_ood+'recon_'+str(idx)+'.npy',recon_slice)

        