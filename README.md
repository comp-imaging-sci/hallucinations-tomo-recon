# hallucinations-tomo-recon
Codes related to the paper "On hallucinations in tomographic image reconstruction" by Bhadra *et al.*: https://arxiv.org/pdf/2012.00646.pdf.

## Directory structure and usage
* `recon_data`: Contains 5 data samples each (indexed as 0-4) from in-distribution (ind) and out-of-distribution (ood) data. Each data sample contains the true object, segmentation mask and simulated k-space data.
* `UNET`: Contains codes used for reconstructing image using a pre-trained U-Net model.
* `PLSTV`: Contains code for reconstructing image by use of the PLS-TV method.
* `compute_maps`: Contains codes for computing hallucination maps and specific maps.

## U-Net
The U-Net model was trained using codes from https://github.com/facebookresearch/fastMRI which uses `pytorch` and `pytorch-lightning`. We have placed the pre-trained model used in our numerical studies as `UNET/experiments/h_map/epoch\=49.ckpt` which can be used to reconstruct images from the test dataset. The hyperparameters used during training can be found at `UNET/experiments/h_map/meta_tags.csv`.

#### Dependencies
The codes for reconstructing images using the trained U-Net model have been tested successfully using the `pytorch` virtual environment installed with `conda 4.5.12`. The relevant softwares which are pre-requisites and must be installed within the virtual environment have been listed in `UNET/requirements.txt`.

#### Instructions
1. Enter the `UNET` directory from root directory:
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
The reconstructed images will be saved in new subdirectories `recons_ind` and `recons_ood`.

## PLS-TV
The Berkeley Advanced Reconstruction Toolbox (BART) software is used for the PLS-TV method: https://mrirecon.github.io/bart/. Please install the BART software before running our code for PLS-TV. Our implementation was successfully tested with `bart-0.5.00`. 

#### Instructions
1. Enter the `PLSTV` directory from root directory:
```
cd PLSTV
```
2. Run the script that performs PLS-TV using BART given the distribution type and the corresponding data index. Example:
```
python bart_plstv.py --dist-type ind --idx 2
```
The reconstructed image will be saved in the subdirectory `recons_ind` or `recons_ood` depending on `dist-type`.

## Deep Image Prior (DIP)
DIP-based image reconstruction was performed using a randomly initialized network having the same architecture as the U-Net described above. The method was implemented in Tensorflow 1.14. The relevant hyperparameters used can be found in `DIP/run_dip_unet.sh` and `dip_main.py`. 


#### Dependencies
The codes for reconstructing images using DIP have been tested successfully using `tensorflow-gpu 1.14` installed with `conda`, but should also work with the 1.15 version.

#### Instructions
1. Enter the `DIP` folder:
```
cd DIP
```
2. Run the following script to reconstruct an image from kspace data with index `i` (`i` goes from 0 to 4) and type `Type` (type is either `ind` or `ood`) as follows:
```
bash run_dip_unet.sh $Type $i
```
3. The reconstructed images will be saved in new subdirectories `recons_ind` and `recons_ood` within the `DIP` folder.

## Computing hallucination maps
An error map or a hallucination map can be computed after an image has been reconstructed. The type of map is indicated by entering any of the following arguments:
* `em`: Error map
* `meas_hm`: Measurement space hallucination map
* `null_hm`: Null space hallucination map

#### Instructions
1. Enter the directory `compute_maps` from root directory:
```
cd compute_maps
```
2. Example of computing the null space hallucination map for an ood image reconstructed using the U-Net:
```
python compute_raw_maps.py --recon-type UNET --dist-type ood --map-type null_hm --idx 0
```
The error map or hallucination map will saved in the subdirectory `[recon_type]_[map_type]_[dist-type]`.

3. Compute the specific map (`em` or `null_hm`) after the corresponding raw map has been computed in Step 2. Example of computing specific null space hallucination map after performing the example in Step 2:
```
python compute_specific_maps.py --recon-type UNET --dist-type ood --map-type null_hm --idx 0
```
The specific map is saved as a `.png` file in the subdirectory `[recon_type]_specific_[map_type]_[dist-type]`.

**NOTE:** 

1. The true objects, reconstructed images and hallucination maps should be vertically flipped upside-down for display purposes.
2. The type of distribution and index corresponding to the figures shown in the paper for brain images are as follows:
  1. Fig. 1 (bottom row): `--dist-type ood --idx 0`
  2. Fig. 2: `--dist-type ind --idx 0`
  3. Fig. 3: `--dist-type ood --idx 1`

## Citation
```
@article{bhadra2020hallucinations,
  title={On hallucinations in tomographic image reconstruction},
  author={Bhadra, Sayantan and Kelkar, Varun A and Brooks, Frank J and Anastasio, Mark A},
  journal={arXiv preprint arXiv:2012.00646},
  year={2020}
}
```