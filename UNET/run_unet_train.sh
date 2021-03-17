#!/bin/bash
python -u models/unet/train_unet.py --mode train --challenge singlecoil --data-path ./datasets/ --exp h_map --mask-type equispaced --gpus 1 --master_port=$RANDOM > train.log
