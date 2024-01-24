import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from skimage.transform import resize


@st.cache_data
def resize_image(image, target_shape=(200, 200)):
    """
    Resize an image to a target shape while maintaining its aspect ratio.
    Fill any extra space with black pixels.
    :param image: Input image
    :param target_shape: Desired shape (height, width)
    :return: Resized image with the same aspect ratio
    """
    is_color = len(image.shape) == 3 and image.shape[2] == 4

    # Calculate the aspect ratio of the image
    aspect_ratio = image.shape[1] / image.shape[0]
    target_aspect_ratio = target_shape[1] / target_shape[0]

    if aspect_ratio > target_aspect_ratio:
        # Width is larger than height in proportion to target shape
        new_width = target_shape[1]
        new_height = int(new_width / aspect_ratio)
    else:
        # Height is larger than width in proportion to target shape
        new_height = target_shape[0]
        new_width = int(new_height * aspect_ratio)

    if is_color:
        # Resize the image with the new dimensions, considering the color dimension
        resized_image = resize(
            image,
            (new_height, new_width, 4),
            mode="constant",
            anti_aliasing=True,
        )

        # Create a black canvas with the target shape and 3 color  + 1 alpha channels
        final_image = np.zeros(
            (target_shape[0], target_shape[1], 4), dtype=resized_image.dtype
        )
        # add alpha channel to 1
        final_image[:, :, 3] = 1
    else:
        # Resize grayscale image
        resized_image = resize(
            image, (new_height, new_width), mode="constant", anti_aliasing=True
        )

        # Create a black canvas with the target shape
        final_image = np.zeros(
            (target_shape[0], target_shape[1]), dtype=resized_image.dtype
        )

    # Determine the center offset
    y_offset = (target_shape[0] - new_height) // 2
    x_offset = (target_shape[1] - new_width) // 2

    # Place the resized image onto the canvas
    if is_color:
        final_image[
            y_offset:y_offset + new_height,
            x_offset:x_offset + new_width,
            :,
        ] = resized_image
    else:
        final_image[
            y_offset:y_offset + new_height, x_offset:x_offset + new_width
        ] = resized_image

    return final_image


def plot_mri_slices(
    segmentation_data, structural_data, x_slider, y_slider, z_slider
):
    # Create Matplotlib figure and subplots for each view
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes = axes.flatten()

    # Sagittal
    axes[0].imshow(structural_data[x_slider, :, :], cmap="gray")
    axes[0].set_title("Sagittal")

    # Coronal
    axes[1].imshow(structural_data[:, y_slider, :], cmap="gray")
    axes[1].set_title("Coronal")

    # Axial
    axes[2].imshow(structural_data[:, :, z_slider], cmap="gray")
    axes[2].set_title("Axial")

    # Hide axes ticks and labels
    for ax in axes:
        ax.axis("off")

    # Render the Matplotlib figure in Streamlit
    st.pyplot(fig)


# Function to convert data slice to grayscale using matplotlib
@st.cache_data
def convert_to_grayscale(data_slice):
    # normalize data slice between 0 and 1
    data_slice = data_slice.astype(np.float32)
    data_slice -= np.min(data_slice)
    data_slice /= np.max(data_slice)

    return data_slice


@st.cache_data
def plot_legend(segmentation_colors, segmentation_labels, segmentation_long_names):
    """Create, with matplotlib, a legend for the segmentation colors.

    The legend should be a single column, with the segmentation color on the left, in a square, and the long label on the right. One line per segmentation label.

    Returns
    -------
    fig : matplotlib figure    
    """

    # Create Matplotlib figure and subplots for each view
    fig, ax = plt.subplots(1, 1)  # Increase the size of the legend
    fig.patch.set_alpha(0.0)  # Set the figure background to be transparent

    # Create a list of patches
    patches = []

    # Create a list of labels
    labels = []

    # Loop over the segmentation colors
    for label, color in segmentation_colors.items():
        # Create a patch for the color
        patch = plt.Rectangle((0, 0), 1, 1, fc=color, alpha=0.5)
        patches.append(patch)
        label_short = segmentation_labels[label]

        # Create a label for the color
        label = segmentation_long_names[label_short]
        labels.append(label)
    
    # Create the legend
    legend = ax.legend(patches, labels, loc='center', ncol=1, framealpha=0.0)
    for text in legend.get_texts():
        text.set_color("white")

    # Hide axes ticks and labels
    ax.axis("off")
    plt.tight_layout(pad=15.5)

    return fig



@st.cache_data
def overlay_images(image, segmentation, alpha, color_map):
    """
    Overlay segmentation on an image.
    :param image: The main image
    :param segmentation: Segmentation image
    :param alpha: Transparency level of the segmentation (0 <= alpha <= 1)
    :param color_map: Dictionary mapping segmentation values to RGBA colors
    :return: Overlay image
    """

    print(color_map)

    overlay = np.zeros((*image.shape, 4))  # 4 channels: RGB + Alpha
    for label, color in color_map.items():
        overlay[segmentation == int(label)] = color

    # Adjust the alpha channel based on the provided transparency level
    overlay[:, :, 3] *= alpha

    image_rgb = convert_to_grayscale(image)
    image_rgb = plt.cm.gray(image_rgb)[:, :, :3]

    image_rgba = np.dstack(
        [image_rgb, np.ones(image_rgb.shape[:2])]
    )  # Add alpha channel to MRI

    # Blend images
    blended = overlay * overlay[:, :, 3:4] + image_rgba * (
        1 - overlay[:, :, 3:4]
    )

    return blended

def plot_mri_slices_col(segmentation_data, structural_data, show_segmentation, transparency, color_map):
    # Extract the shape of the structural data
    x_max, y_max, z_max = structural_data.shape

    # Pre-compute grayscale images
    grayscale_images = np.array([convert_to_grayscale(slice) for slice in structural_data])

    # Pre-compute overlays
    overlays = np.array([overlay_images(slice, segmentation, transparency, color_map) for slice, segmentation in zip(structural_data, segmentation_data)])

    # Adjusting the layout using Streamlit columns
    # Create a 3-column layout
    col1, col2, col3 = st.columns(3)

    # Sagittal view with its slider
    with col1:
        st.markdown("<h3 style='text-align: center; color: white;'>Saggital</h3>", unsafe_allow_html=True)
        idx_x = st.slider("slider_x", 0, x_max - 1, int(x_max / 2), key="x_slider", label_visibility="hidden")
        displayed_image_x = overlays[idx_x, :, :] if show_segmentation else grayscale_images[idx_x, :, :]
        st.image(resize_image(displayed_image_x), use_column_width=True)

    # Coronal view with its slider
    with col2:
        st.markdown("<h3 style='text-align: center; color: white;'>Coronal</h3>", unsafe_allow_html=True)
        idx_y = st.slider("slider_y", 0, y_max - 1, int(y_max / 2), key="y_slider", label_visibility="hidden")
        displayed_image_y = overlays[:, idx_y, :] if show_segmentation else grayscale_images[:, idx_y, :]
        st.image(resize_image(displayed_image_y), use_column_width=True)

    # Axial view with its slider
    with col3:
        st.markdown("<h3 style='text-align: center; color: white;'>Axial</h3>", unsafe_allow_html=True)
        idx_z = st.slider("label_z", 0, z_max - 1, int(z_max / 2), key="z_slider", label_visibility="hidden")
        displayed_image_z = overlays[:, :, idx_z] if show_segmentation else grayscale_images[:, :, idx_z]
        st.image(resize_image(displayed_image_z), use_column_width=True)
