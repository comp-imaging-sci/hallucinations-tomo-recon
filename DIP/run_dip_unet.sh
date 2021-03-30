#!/bin/bash

Type=$1 # ind or ood
idx=$2  # index of the recon (0-4)

snr=20
learning_rate=0.01
date=210329
reg_TV=1e-02

gt_filename=../recon_data/gt_${Type}/gt_${idx}.npy
meas_filename=../recon_data/kspace_${Type}/kspace_${idx}.npy
results_dir=$basedir/phase_noise_experiments/recons_dip2_${Type}_pn_${phasenoise}/recon_$rootpath
results_dir=./recons_${Type}/recon_${idx}

mkdir -p $results_dir

python -u dip_main.py --snr $snr --learning_rate $learning_rate --reg_TV $reg_TV --date $date --gt_filename $gt_filename --meas_filename $meas_filename --results_dir $results_dir --net unet