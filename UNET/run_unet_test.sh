#!/bin/bash
python -u models/unet/test_unet.py --mode test --challenge singlecoil --gpus 1 --data-path ./ --exp h_map --checkpoint ./experiments/h_map/epoch\=49.ckpt 

