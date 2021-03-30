#!/bin/bash

Type=$1 # ind or ood
idx=$2  # index of the recon (0-4)

snr=20
learning_rate=0.01
date=210329
reg_TV=1e-02
T_in=1000

gt_filename=../recon_data/gt_${Type}/gt_${idx}.npy
meas_filename=../recon_data/kspace_${Type}/kspace_${idx}.npy
mkdir -p ./recons_${Type}
results_path=./recons_${Type}/recon_${idx}

python -u dip_main.py --snr $snr --learning_rate $learning_rate --reg_TV $reg_TV --T_in $T_in --date $date --gt_filename $gt_filename --meas_filename $meas_filename --results_path $results_path --net unet