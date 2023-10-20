import streamlit as st
from src.data_processing import (
    load_csv_data,
    load_mri_data,
    get_cropped_stack_based_on_mask,
)  # Assume load_csv_data is defined in data_processing.py
from src.visualization import (
    plot_mri_slices,
    plot_mri_slices_col,
)  # Assume plot_mri_slices is defined in visualization.py
import ipywidgets as widgets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.title("Fetal MRI Biomarker Visualization")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data = load_csv_data(uploaded_file)
    st.write(data)  # This will display the dataframe in Streamlit

# Upload MRI files
uploaded_seg = st.file_uploader(
    "Upload a segmentation file (_dseg.nii.gz)", type=["nii.gz"]
)
uploaded_struct = st.file_uploader(
    "Upload a structural scan file (_T2w.nii.gz)", type=["nii.gz"]
)


if uploaded_seg is not None and uploaded_struct is not None:
    segmentation, structural = load_mri_data(uploaded_seg, uploaded_struct)

    # process MRI data to be displayed
    image_cropped = get_cropped_stack_based_on_mask(structural, segmentation)
    mask_cropped = get_cropped_stack_based_on_mask(segmentation, segmentation)

    # initialize session state
    st.session_state["seg_disabled"] = False

    col1, col2 = st.columns(2)

    def disable():
        st.session_state["seg_disabled"] = not st.session_state["seg_disabled"]

    with col1:
        show_segmentation = st.checkbox(
            "Segmentation", key="but_a", on_change=disable
        )

    with col2:
        transparency = st.slider(
            "Segmentation transparency",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01,
            disabled=st.session_state["seg_disabled"],
        )

    plot_mri_slices_col(
        mask_cropped.get_fdata(),
        image_cropped.get_fdata(),
        show_segmentation,
        transparency,
    )
    # Sample data
    np.random.seed(42)  # for reproducibility
    data = {
        "Age": np.random.randint(20, 40, 100),
        "Volume": np.random.randn(100) * 10 + 100,
        "Label": np.random.choice(
            ["Background", "Grey Matter", "White Matter"], 100
        ),
    }
    df = pd.DataFrame(data)

    # Dropdown for labels
    label = st.selectbox(
        "Choose a label", ["Background", "Grey Matter", "White Matter"]
    )

    # Filtering data based on label
    filtered_df = df[df["Label"] == label]

    # Creating the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    for lbl, group in df.groupby("Label"):
        if lbl == label:
            ax.scatter(
                group["Age"], group["Volume"], label=lbl, s=60, alpha=0.7
            )
        else:
            ax.scatter(
                group["Age"], group["Volume"], label=lbl, s=30, alpha=0.3
            )

    current_subject_age = 25  # for example
    current_subject_volume = 110  # for example
    ax.scatter(
        current_subject_age,
        current_subject_volume,
        color="red",
        s=100,
        marker="X",
        label="Current Subject",
    )
    ax.legend()
    ax.set_xlabel("Age (Weeks)")
    ax.set_ylabel("Volume")
    ax.set_title(f"Volume vs Age for {label}")

    # Displaying the plot
    st.pyplot(fig)
