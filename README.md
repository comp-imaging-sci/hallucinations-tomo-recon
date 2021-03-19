# hallucinations-tomo-recon
Codes related to the paper "On hallucinations in tomographic image reconstruction" by Bhadra *et al.*: https://arxiv.org/pdf/2012.00646.pdf.

## Directory structure and usage
* `recon_data`: Contains 5 data samples each (indexed as 0-4) from in-distribution (ind) and out-of-distribution (ood) data. Each data sample contains the true object, segmentation mask and simulated k-space data.
* `UNET`: Contains codes used for reconstructing image using a pre-trained U-Net model.
* `PLSTV`: Contains code for reconstructing image by use of the PLS-TV method.
* `compute_maps`: Contains codes for computing hallucination maps and specific maps.

## U-Net
The U-Net model was trained using codes from https://github.com/facebookresearch/fastMRI which uses `pytorch` and `pytorch-lightning`. The true object images for training were axial brain MRI images collected from the NYU fastMRI Initiative database: https://arxiv.org/pdf/1811.08839.pdf. We have placed the pre-trained model used in our numerical studies as `UNET/experiments/h_map/epoch\=49.ckpt` which can be used to reconstruct images from the test dataset. The hyperparameters used during training can be found at `UNET/experiments/h_map/meta_tags.csv`.

#### Dependencies
The codes for reconstructing images using the trained U-Net model have been tested successfully using the `pytorch` virtual environment installed with `conda 4.5.12`. The relevant softwares which are pre-requisites and must be installed within the virtual environment have been listed in `UNET/requirements.txt`.

#### Instructions
1. Enter the `UNET` directory from root:
```
cd UNET
```
2. Run the following script to perform reconstruction using the saved U-Net model:
```
./run_unet_test.sh
```
3. Extract reconstructed images as numpy arrays from saved `.h5` files:
```
python extract_recons.py
```

## PLS-TV
The Berkeley Advanced Reconstruction Toolbox (BART) software is used for the PLS-TV method: https://mrirecon.github.io/bart/. Please install the BART software before running our code for PLS-TV. Our implementation was successfully tested with `bart-0.5.00`. 

#### Instructions
1. Enter the `PLSTV` directory from root:
```
cd PLSTV
```
2. Run the script that performs PLS-TV using BART given the distribution type and the corresponding data index. Example:
```
python bart_plstv.py --dist-type ind --idx 2
```

