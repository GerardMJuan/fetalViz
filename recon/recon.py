"""
Functions containing all the steps of the reconstruction pipeline.
"""

import os
import json
import shutil
import time
import subprocess
import docker
import pandas as pd
import nibabel as nib
import copy
import numpy as np
import glob

DOCKER_NIFTYMIC = "gerardmartijuan/niftymic.multifact"
DOCKER_ANTS = "antsx/ants:master"

def apply_masks(list_fo_files, list_of_masks):
    """apply the corresponding mask to each file, overwriting it"""
    for file, mask in zip(list_fo_files, list_of_masks):
        subprocess.call(
            [
                "fslmaths",
                file,
                "-mas",
                mask,
                file,
            ]
        )

def denoise_image(in_path, out_path):
    """
    Use ANTs to denoise an image.

    Eventually, use docker to run the ANTs command.
    """

    # Run denoise image using a subprocess call
    subprocess.call(
        [
            "DenoiseImage",
            "-i",
            str(in_path),
            "-o",
            str(out_path),
            "-n",
            "Gaussian",
            "-s",
            "1",
        ]
    )


def squeeze_dim(arr, dim):
    if arr.shape[dim] == 1 and len(arr.shape) > 3:
        return np.squeeze(arr, axis=dim)
    return arr


def get_cropped_stack_based_on_mask(
    image_ni, mask_ni, boundary_i=15, boundary_j=15, boundary_k=0, unit="mm"
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

    image_ni = copy.deepcopy(image_ni)

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
    image_cropped = nib.Nifti1Image(image_cropped, new_affine)
    # image_cropped.header.set_xyzt_units(2)
    # image_cropped.header.set_qform(new_affine, code="aligned")
    # image_cropped.header.set_sform(new_affine, code="scanner")
    return image_cropped


def remove_directory_contents(path: str):
    """
    Removes all files and directories within the specified path.

    :param path: Path to the directory whose contents are to be removed.
    :raises FileNotFoundError: If the specified path does not exist.
    :raises PermissionError: If there are insufficient permissions to remove the contents.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"The path {path} does not exist.")

    for root, directories, files in os.walk(path, topdown=False):
        for file_name in files:
            os.remove(os.path.join(root, file_name))
        for dir_name in directories:
            os.rmdir(os.path.join(root, dir_name))

    shutil.rmtree(path)

def crop_images_and_masks(
    list_of_files,
    list_of_masks,
    list_of_cropped_files,
    list_of_cropped_masks,
):
    """Crop the images and masks to the bounding box of the masks

    Parameters
    ----------
    list_of_files : array
        Contains the paths to the files to be cropped
    list_of_masks : array
        Contains the paths to the masks to be cropped
    list_of_cropped_files : array
        Contains the paths to the output cropped files
    list_of_cropped_masks : array
        Contains the paths to the output cropped masks
    """

    for file, mask, cropped_file, cropped_mask in zip(
        list_of_files, list_of_masks, list_of_cropped_files, list_of_cropped_masks
    ):
        # load the image and mask
        import pdb; pdb.set_trace()
        image_ni = nib.load(file)
        mask_ni = nib.load(mask)

        # get the cropped image and mask
        image_cropped = get_cropped_stack_based_on_mask(image_ni, mask_ni)
        mask_cropped = get_cropped_stack_based_on_mask(mask_ni, mask_ni)

        # apply the mask to the image
        image_cropped_masked = (
            image_cropped.get_fdata()
            * mask_cropped.get_fdata()
        )

        # update the image with the cropped image
        image_cropped = nib.Nifti1Image(
            image_cropped_masked, image_cropped.affine
        )

        # save the cropped image and mask
        nib.save(image_cropped, cropped_file)
        nib.save(mask_cropped, cropped_mask)


def create_brain_masks(
    list_of_files,
    list_of_masks,
    gpu=False,
):
    """
    Creates the brain masks using NiftyMIC

    GPU disabled by default
    """
    # get base directory of the files (it will be the same for all)
    base_dir = os.path.dirname(list_of_files[0])
    # replace the base dir with docker dir /app/NiftyMIC/nifti/
    list_of_files = [
        str(x).replace(base_dir, "/app/NiftyMIC/nifti") for x in list_of_files
    ]
    base_dir_mask = os.path.dirname(list_of_masks[0])

    # create base_dir_mask if it does not exist
    if not os.path.exists(base_dir_mask):
        os.makedirs(base_dir_mask)

    list_of_masks = [
        str(x).replace(base_dir_mask, "/app/NiftyMIC/masks")
        for x in list_of_masks
    ]
    command = [
        "niftymic_segment_fetal_brains",
        "--filenames",
        " ".join(str(x) for x in sorted(list_of_files)),
        "--filenames-masks",
        " ".join(str(x) for x in sorted(list_of_masks)),
    ]
    command = " ".join(command)
    client = docker.from_env()

    # check if gpu is available
    if not gpu:
        dev_req = []
    else:
        dev_req = [
            docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])
        ]

    container = client.containers.run(
        DOCKER_NIFTYMIC,
        user=os.getuid(),
        command=command,
        volumes={
            base_dir: {"bind": "/app/NiftyMIC/nifti", "mode": "rw"},
            base_dir_mask: {"bind": "/app/NiftyMIC/masks", "mode": "rw"},
        },
        device_requests=dev_req,
        detach=True,
        stderr=True,
        stdout=True,
    )
    container.wait()

    # save the logs to the output folder
    with open(f"{base_dir}/log_masks.txt", "w") as f:
        f.write(container.logs().decode("utf-8"))


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
    # Return rectangular region surrounding masked region.
    #  \param[in] mask_sitk sitk.Image representing the mask
    #  \return range_x pair defining x interval of mask in voxel space
    #  \return range_y pair defining y interval of mask in voxel space
    #  \return range_z pair defining z interval of mask in voxel space


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


def reconstruct_volume(
    input_recon_dir, mask_recon_dir, recon_dir, algorithm
):
    """
    Reconstructs the volume using the specified algorithm
    And in the specified environment.


    """
    list_of_files = glob.glob(os.path.join(input_recon_dir, "*.nii.gz"))
    list_of_masks = glob.glob(os.path.join(mask_recon_dir, "*.nii.gz"))

    if algorithm == "niftymic":
        docker_image = DOCKER_NIFTYMIC
        docker_command = [
            "niftymic_run_reconstruction_pipeline",
            "--filenames",
            " ".join(
                os.path.join("/input", os.path.basename(x))
                for x in sorted(list_of_files)
            ),
            "--filenames-masks",
            " ".join(
                os.path.join("/masks", os.path.basename(x))
                for x in sorted(list_of_masks)
            ),
            "--dir-output",
            "/srr",
            "--isotropic-resolution",
            "0.8",
            "--suffix-mask",
            "_mask" "--alpha" "0.01",
            "--automatic-target-stack",
            "1",
            "--run-bias-field-correction",
            "1",
        ]
        docker_command = " ".join(docker_command)
        docker_volumes = {
            recon_dir: {"bind": "/srr", "mode": "rw"},
            input_recon_dir: {"bind": "/input", "mode": "rw"},
            mask_recon_dir: {"bind": "/masks", "mode": "rw"},
        }

    elif algorithm == "nesvor":
        print('nyi')
        # docker_image = DOCKER_NESVOR
        # docker_command = [
            # "nesvor",
            # "reconstruct",
            # "--input-stacks",
            # " ".join(str(x) for x in sorted(list_of_files)),
            # "--stack-masks",
            # " ".join(str(x) for x in sorted(list_of_masks)),
            # "--output-volume",
            # "/out/nesvor.nii.gz",
            # "--output-resolution",
            # "--bias-field-correction",
            # "0.8",
            # "--n-levels-bias",
            # "1",
            # "--batch-size",
            # "8192",
        # ]
        # docker_command = " ".join(docker_command)
        # docker_volumes = {
            # recon_dir: {"bind": "/out", "mode": "rw"},
            # input_recon_dir: {"bind": "/data", "mode": "rw"},
            # mask_recon_dir: {"bind": "/data", "mode": "rw"},
        # }
# 
    elif algorithm == "svrtk":
        print('nyi')
        # docker_image = DOCKER_SVRTK
        # docker_command = [
        #     "bash",
        #     "/home/auto-proc-svrtk/auto-brain-reconstruction.sh",
        #     "/home/data/input",
        #     "/home/data",
        # ]
        # docker_command = " ".join(docker_command)
        # docker_volumes = {
        #     recon_dir: {"bind": "/home/data", "mode": "rw"},
        # }

    # use the docker interface for python
    client = docker.from_env()

    if algorithm != "nesvor":
        # no need for gpu
        gpu = []
    else:
        gpu = [
            docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])
        ]

    container = client.containers.run(
        docker_image,
        user=os.getuid(),
        command=docker_command,
        volumes=docker_volumes,
        detach=True,
        device_requests=gpu,
        stderr=True,
        stdout=True,
    )
    container.wait()
    # save the logs to the output folder
    with open(f"{recon_dir}/log_{algorithm}.txt", "w") as f:
        f.write(container.logs().decode("utf-8"))

    container.remove()

    # for each algorithm, copy the result to the output folder
    # with the correct BIDS name
    if algorithm == "niftymic":
        result_img = f"{recon_dir}/recon_template_space/srr_template.nii.gz"
        result_mask = (
            f"{recon_dir}/recon_template_space/srr_template_mask.nii.gz"
        )
    elif algorithm == "nesvor":
        result_img = f"{recon_dir}/nesvor.nii.gz"
        result_mask = None
    elif algorithm == "svrtk":
        result_img = f"{recon_dir}/reo-SVR-output-brain.nii.gz"
        result_mask = None

    # Change result_img to recon.nii.gz and result_mask to recon_mask.nii.gz
    # and copy them to the base output folder
    shutil.copy(result_img, f"{recon_dir}/recon.nii.gz")
    if result_mask is not None:
        shutil.copy(result_mask, f"{recon_dir}/recon_mask.nii.gz")

    return result_img, result_mask
