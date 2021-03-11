# Helper functions related to the undersampled single-coil MRI imaging operator with Cartesian sampling
import numpy as np
import numpy.fft as fft

# Forward operator 
def forward(f,mask):
    return mask*fft.fftshift(fft.fft2(fft.ifftshift(f),norm='ortho'))

# Pseudoinverse operator
def pinv(g):
    return fft.fftshift(fft.ifft2(fft.ifftshift(g),norm='ortho'))

# Function for computing measurement and null components of an object
def f_meas_null(f,mask):
    H_f = forward(f,mask)
    f_meas = pinv(H_f)
    f_null = f - f_meas
    return f_meas, f_null

