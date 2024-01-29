import pandas as pd
import nibabel as nib
import io
import tempfile
import numpy as np
import copy
import streamlit as st

@st.cache_data
def load_csv_data(uploaded_file):
    data = pd.read_csv(uploaded_file)
    return data

@st.cache_data
def load_mri_data(uploaded_seg, uploaded_struct):
    struct_data = None
    seg_data = None

    seg_img = nib.load(uploaded_seg)

    seg_data = seg_img.__class__(
        seg_img.get_fdata(), seg_img.affine, seg_img.header
    )

    struct_img = nib.load(uploaded_struct)

    struct_data = struct_img.__class__(
        struct_img.get_fdata(), struct_img.affine, struct_img.header
    )

    return seg_data, struct_data

def squeeze_dim(arr, dim):
    if arr.shape[dim] == 1 and len(arr.shape) > 3:
        return np.squeeze(arr, axis=dim)
    return arr

@st.cache_data
def get_cropped_stack_based_on_mask(
    _image_ni, _mask_ni, uploaded_seg, boundary_i=15, boundary_j=15, boundary_k=0, unit="mm"
):
    """
    Crops the input image to the field of view given by the bounding box
    around its mask.
    Original code by Michael Ebner:
    https://github.com/gift-surg/NiftyMIC/blob/master/niftymic/base/stack.py

    Input
    -----
    image_ni:
        Nifti image
    mask_ni:
        Corresponding nifti mask
    uploaded_seg:
        Uploaded segmentation, only for caching purposes
    boundary_i:
    boundary_j:
    boundary_k:
    unit:
        The unit defining the dimension size in nifti

    Output
    ------
    image_cropped:
        Image cropped to the bounding box of mask_ni
    mask_cropped
        Mask cropped to its bounding box
    """

    image_ni = copy.deepcopy(_image_ni)
    mask_ni = copy.deepcopy(_mask_ni)

    image = squeeze_dim(image_ni.get_fdata(), -1)
    mask = squeeze_dim(mask_ni.get_fdata(), -1)

    assert all(
        [i >= m] for i, m in zip(image.shape, mask.shape)
    ), "For a correct cropping, the image should be larger or equal to the mask."

    # Get rectangular region surrounding the masked voxels
    [x_range, y_range, z_range] = get_rectangular_masked_region(mask)

    if np.array([x_range, y_range, z_range]).all() is None:
        print("Cropping to bounding box of mask led to an empty image.")
        return None

    if unit == "mm":
        spacing = image_ni.header.get_zooms()
        boundary_i = np.round(boundary_i / float(spacing[0]))
        boundary_j = np.round(boundary_j / float(spacing[1]))
        boundary_k = np.round(boundary_k / float(spacing[2]))

    shape = [min(im, m) for im, m in zip(image.shape, mask.shape)]
    x_range[0] = np.max([0, x_range[0] - boundary_i])
    x_range[1] = np.min([shape[0], x_range[1] + boundary_i])

    y_range[0] = np.max([0, y_range[0] - boundary_j])
    y_range[1] = np.min([shape[1], y_range[1] + boundary_j])

    z_range[0] = np.max([0, z_range[0] - boundary_k])
    z_range[1] = np.min([shape[2], z_range[1] + boundary_k])

    new_origin = list(
        nib.affines.apply_affine(
            mask_ni.affine, [x_range[0], y_range[0], z_range[0]]
        )
    ) + [1]

    new_affine = image_ni.affine
    new_affine[:, -1] = new_origin
    image_cropped = crop_image_to_region(image, x_range, y_range, z_range)
    mask_cropped = crop_image_to_region(mask, x_range, y_range, z_range)
    # image_cropped = nib.Nifti1Image(image_cropped, new_affine)
    return image_cropped, mask_cropped


@st.cache_data
def crop_image_to_region(
    image: np.ndarray,
    range_x: np.ndarray,
    range_y: np.ndarray,
    range_z: np.ndarray,
) -> np.ndarray:
    """
    Crop given image to region defined by voxel space ranges
    Original code by Michael Ebner:
    https://github.com/gift-surg/NiftyMIC/blob/master/niftymic/base/stack.py

    Input
    ------
    image: np.array
        image which will be cropped
    range_x: (int, int)
        pair defining x interval in voxel space for image cropping
    range_y: (int, int)
        pair defining y interval in voxel space for image cropping
    range_z: (int, int)
        pair defining z interval in voxel space for image cropping

    Output
    ------
    image_cropped:
        The image cropped to the given x-y-z region.
    """
    image_cropped = image[
        range_x[0] : range_x[1],
        range_y[0] : range_y[1],
        range_z[0] : range_z[1],
    ]
    return image_cropped


@st.cache_data
def get_rectangular_masked_region(
    mask: np.ndarray,
) -> tuple:
    """
    Computes the bounding box around the given mask
    Original code by Michael Ebner:
    https://github.com/gift-surg/NiftyMIC/blob/master/niftymic/base/stack.py

    Input
    -----
    mask: np.ndarray
        Input mask
    range_x:
        pair defining x interval of mask in voxel space
    range_y:
        pair defining y interval of mask in voxel space
    range_z:
        pair defining z interval of mask in voxel space
    """
    if np.sum(abs(mask)) == 0:
        return None, None, None
    shape = mask.shape
    # Compute sum of pixels of each slice along specified directions
    sum_xy = np.sum(mask, axis=(0, 1))  # sum within x-y-plane
    sum_xz = np.sum(mask, axis=(0, 2))  # sum within x-z-plane
    sum_yz = np.sum(mask, axis=(1, 2))  # sum within y-z-plane

    # Find masked regions (non-zero sum!)
    range_x = np.zeros(2)
    range_y = np.zeros(2)
    range_z = np.zeros(2)

    # Non-zero elements of numpy array nda defining x_range
    ran = np.nonzero(sum_yz)[0]
    range_x[0] = np.max([0, ran[0]])
    range_x[1] = np.min([shape[0], ran[-1] + 1])

    # Non-zero elements of numpy array nda defining y_range
    ran = np.nonzero(sum_xz)[0]
    range_y[0] = np.max([0, ran[0]])
    range_y[1] = np.min([shape[1], ran[-1] + 1])

    # Non-zero elements of numpy array nda defining z_range
    ran = np.nonzero(sum_xy)[0]
    range_z[0] = np.max([0, ran[0]])
    range_z[1] = np.min([shape[2], ran[-1] + 1])

    # Numpy reads the array as z,y,x coordinates! So swap them accordingly
    return (
        range_x.astype(int),
        range_y.astype(int),
        range_z.astype(int),
    )
