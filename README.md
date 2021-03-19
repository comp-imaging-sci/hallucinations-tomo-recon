# hallucinations-tomo-recon
Codes related to the paper "On hallucinations in tomographic image reconstruction" by Bhadra *et al.*: https://arxiv.org/pdf/2012.00646.pdf.

## U-Net
The U-Net model was trained using codes from https://github.com/facebookresearch/fastMRI which uses `pytorch` and `pytorch-lightning`. The true object images for training were axial brain MRI images collected from the NYU fastMRI Initiative database: https://arxiv.org/pdf/1811.08839.pdf. We have placed the pre-trained model used in our numerical studies as `UNET/experiments/h_map/epoch\=49.ckpt` which can be used to reconstruct images from the test dataset.

#### Pre-requisites and environments
The codes for reconstructing images using the trained U-Net model have been tested successfully using the `pytorch` virtual environment installed with `conda 4.5.12`. The relevant softwares which are pre-requisites and must be installed within the virtual environment have been listed in `UNET/requirements.txt`.

#### Instructions

