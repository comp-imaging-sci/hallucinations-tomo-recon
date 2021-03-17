#!/bin/bash
# Change the checkpoint directory and file according to your final checkpoint
python -u models/unet/test_unet.py --mode test --challenge singlecoil --data-path ./datasets/ --exp h_map --checkpoint ./experiments/h_map/version_0/checkpoints/epoch\=49.ckpt 

