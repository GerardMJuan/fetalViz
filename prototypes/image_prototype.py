"""
First prototype with streamlit.

The app would need to open a brain file (in nii.gz format) 
and visualize it in three axes (axial, sagittal and coronal).
The user should be able to navigate through the different slices 
using the mouse wheel when the mouse is over the corresponding axis,
and to zoom in/out with right click + move the mouse.
"""

import streamlit as st
import nibabel as nib
import tempfile
import os
import numpy as np
import matplotlib.pyplot as plt
from itkwidgets import view
import vtk
from ipywidgets import embed
import streamlit.components.v1 as components


def plot_brain_slice(nifti_image, view, slice_index):
    # Get the image data as a numpy array
    img_data = nifti_image.get_fdata()

    # Extract slices
    if view == "axial":
        img_slice = img_data[slice_index, :, :].T
        title = f"Axial (X={slice_index})"
    elif view == "sagittal":
        img_slice = img_data[:, slice_index, :].T
        title = f"Sagittal (Y={slice_index})"
    elif view == "coronal":
        img_slice = img_data[:, :, slice_index].T
        title = f"Coronal (Z={slice_index})"

    # Create transparent background image
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img_slice, cmap="gray", origin="lower")
    ax.set_title(title)

    # Set the background to transparent
    fig.patch.set_visible(False)
    ax.axis("off")

    return fig


def main():
    st.title("Brain Visualization App")

    folder_path = st.text_input(
        "Enter the folder path:",
        value="/media/gerard/HDD/MULTIFACT_DATA/cases_VM/fet-008",
    )

    if folder_path is not None:
        st.write("Visualizing brain image:")

        # Get the image data shape
        folder_path = folder_path.rstrip("/")
        subject = os.path.basename(folder_path)

        # Get the image data as a numpy array
        nifti_image = nib.load(os.path.join(folder_path, "N4", f"{subject}.nii.gz"))

        img_shape = nifti_image.get_fdata().shape

        # initialize the slices
        slice_x = img_shape[0] // 2
        slice_y = img_shape[1] // 2
        slice_z = img_shape[2] // 2

        # Plot slices with selected indices in a 2x2 grid
        col1, col2 = st.columns(2)
        col3, col4 = st.columns(2)

        # Create sliders for each axis in a 2x2 grid
        with col1:
            container_col1 = st.container()
            slice_x = st.slider(
                "Axial slice (X-axis)", 0, img_shape[0] - 1, img_shape[0] // 2
            )
        with col2:
            container_col2 = st.container()
            slice_y = st.slider(
                "Sagittal slice (Y-axis)",
                0,
                img_shape[1] - 1,
                img_shape[1] // 2,
            )

        with col3:
            st.write("")  # Leave this column empty
        with col4:
            container_col4 = st.container()
            slice_z = st.slider(
                "Coronal slice (Z-axis)",
                0,
                img_shape[2] - 1,
                img_shape[2] // 2,
            )

        # Plot the slices
        container_col1.pyplot(plot_brain_slice(nifti_image, "axial", slice_x))
        container_col2.pyplot(plot_brain_slice(nifti_image, "sagittal", slice_y))
        container_col4.pyplot(plot_brain_slice(nifti_image, "coronal", slice_z))


if __name__ == "__main__":
    main()
