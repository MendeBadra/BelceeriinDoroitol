from patchify import patchify, unpatchify
from tqdm import tqdm
import numpy as np
import cupy as cp
import os, gc
from skimage.filters import threshold_otsu
from multipledispatch import dispatch

from vegetation_indices import VegetationIndices


def calc_msavi(x):
    # Vegetation Class
    VI_CLASS = VegetationIndices()

    soil_vi_name = "MSAVI"
    soil_vi_func = VI_CLASS.get_vi_function(soil_vi_name)
    return soil_vi_func(x[0], x[1], x[2], x[3], x[4])


# This is my function
def patch_msavi_threshold(big_tiff_stack, step=256):
    # Vegetation Class
    # VI_CLASS = VegetationIndices()

    # soil_vi_name = "MSAVI"
    # soil_vi_func = VI_CLASS.get_vi_function(soil_vi_name)
    
    # @dispatch(tuple)
    # def calc_msavi(x):
    #     return soil_vi_func(x[0], x[1], x[2], x[3], x[4])
    # @dispatch(tuple)
    # def calc_msavi(x):
    #     # Vegetation Class
    #     VI_CLASS = VegetationIndices()

    #     soil_vi_name = "MSAVI"
    #     soil_vi_func = VI_CLASS.get_vi_function(soil_vi_name)
    #     return soil_vi_func(x[0], x[1], x[2], x[3], x[4])
    
    patch_tiff = patchify(big_tiff_stack, (step, step, big_tiff_stack.shape[2]), step=step)
    
    print("You have got an array of shape: ", big_tiff_stack.shape, " and it will be patched into: ", patch_tiff.shape, ".\nHow ever due to patchify"
          ,"package patches them into whole number, we have resulting shape after unpatchify: ",(patch_tiff.shape[0]*step, patch_tiff.shape[1]*step, 1))
    
    # May be find the lost pixels?
    test = np.zeros((patch_tiff.shape[0]*step, patch_tiff.shape[1]*step))
    patch_mask = patchify(test, (step, step), step=step)
    
    # for loop to calculate msavi, threshold and save it to patch_mask
    for i in tqdm(range(patch_tiff.shape[0])):
        for j in range(patch_tiff.shape[1]):
            single_patch_stack = patch_tiff[i,j,0,:,:,:]
            # multi-dim array to tuple and then msavi
            if np.count_nonzero(single_patch_stack) > 5:
                single_patch_tuple = tuple(cp.array(single_patch_stack[:,:,band])
                                       for band in range(patch_tiff.shape[5]))
                msavi = calc_msavi(single_patch_tuple)
                thresh = threshold_otsu(msavi)
                mask = cp.where(msavi > thresh, 1, 0)
                patch_mask[i,j,:,:] = mask.get()
                del single_patch_tuple, msavi, thresh, mask
                gc.collect()
            else:
                patch_mask[i,j,:,:] = np.zeros((256,256))
    
    print("Your patch processing have done!")            
    reconst_mask = unpatchify(patch_mask, test.shape)
    del test, patch_mask, patch_tiff
    gc.collect()
    return reconst_mask