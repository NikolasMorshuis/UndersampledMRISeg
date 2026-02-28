""" Undersampling Mask Creation Script """

from sigpy.mri.samp import poisson
import numpy as np

# The example mask at 8x acceleration provided by the K2S dataset has a ratio of 0.3 of the k-space sampled in center
# We keep this ratio across other acceleration factors.


mask_8x = poisson(
    (256,256),
    8,
    calib=(50,50),
    dtype=np.float32,
    seed=42,
    max_attempts=30,
)

def calculate_fullysampled_center_size(acc, img_shape, ratio=0.3):
    total_samples = img_shape[0] * img_shape[1] / acc
    center_samples = int(total_samples * ratio)
    center_size = int(np.sqrt(center_samples))
    return center_size

acceleration_factors = [16, 32, 64, 128]

masks = {}
for acc in acceleration_factors:
    center_size = calculate_fullysampled_center_size(acc, (256,256), ratio=0.3)
    mask = poisson(
        (256,256),
        acc,
        calib=(center_size,center_size),
        dtype=np.bool_,
        seed=42,
        max_attempts=30,
    )
    mask = mask.astype(np.bool_)
    masks[acc] = mask

print('ok')