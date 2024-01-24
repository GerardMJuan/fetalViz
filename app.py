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
from src.sidebar import create_sidebar
import ipywidgets as widgets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import glob 

if 'seg_enabled' not in st.session_state:
    st.session_state['seg_enabled'] = False

st.title("Fetal MRI Biomarker Visualization")

uploaded_seg, uploaded_struct, df_data, config_model = create_sidebar()

# Print current directory
# st.write(os.getcwd())


if uploaded_seg is not None and uploaded_struct is not None:
    segmentation, structural = load_mri_data(uploaded_seg, uploaded_struct)

    # process MRI data to be displayed
    image_cropped, mask_cropped = get_cropped_stack_based_on_mask(structural, segmentation)

    col1, col2 = st.columns(2)

    with col1:
        st.session_state["seg_enabled"] = st.checkbox(
            "Show segmentation",
            key="but_a",
        )
    with col2:
        transparency = st.slider(
            "Segmentation transparency",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01,
            disabled=not st.session_state["seg_enabled"],
        )

    plot_mri_slices_col(
        mask_cropped,
        image_cropped,
        st.session_state["seg_enabled"],
        transparency,
        config_model["segmentation"]["color_map"],
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

    current_subject_age = st.slider(
        "Select current subject age",
        min_value=20,
        max_value=40,
        value=25,
    )

    current_subject_volume = st.slider(
        "Select current subject volume",
        min_value=80,
        max_value=120,
        value=110,
    )

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
