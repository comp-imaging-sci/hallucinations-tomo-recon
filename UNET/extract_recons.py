# Extract reconstructed test images from the U-Net method
import h5py
import os
import numpy as np 

model = 'h_map'
num_recons = 69

recon_dir = './experiments/'+model+'/reconstructions/'

ensemble_dir_ind = './recons_ind/'
if not os.path.exists(ensemble_dir_ind):
    os.makedirs(ensemble_dir_ind)
ensemble_dir_ood = './recons_ood/'
if not os.path.exists(ensemble_dir_ood):
    os.makedirs(ensemble_dir_ood)

# Ensemble of ind recons
vol_count = 0
img_count = 0

print('Start saving ind recons\n')
while vol_count < 50:
    print('Vol = '+str(vol_count))
    filename = recon_dir+'file_ind_'+str(vol_count)+'_pn_uni_0.3.h5'
    recon = h5py.File(filename,'r')['reconstruction']
    recon = np.asarray(recon)
    for idx in range(min(10,recon.shape[0])):
        if img_count == num_recons:
            break
        recon_slice = recon[idx]
        np.save(ensemble_dir_ind+'recon_'+str(img_count)+'.npy',recon_slice)
        print('Saving ind recon '+str(img_count))
        img_count +=1
    else:
        vol_count +=1
        continue
    break

#Ensemble of ood recons
vol_count = 0
img_count = 0

print('Start saving ood recons\n')
while vol_count < 50:
    print('Vol = '+str(vol_count))
    filename = recon_dir+'file_ood_'+str(vol_count)+'_pn_uni_0.3.h5'
    recon = h5py.File(filename,'r')['reconstruction']
    recon = np.asarray(recon)
    for idx in range(recon.shape[0]):
        if img_count == num_recons:
            break
        recon_slice = recon[idx]
        np.save(ensemble_dir_ood+'recon_'+str(img_count)+'.npy',recon_slice)
        print('Saving ood recon '+str(img_count))
        img_count +=1
    else:
        vol_count +=1
        continue
    break