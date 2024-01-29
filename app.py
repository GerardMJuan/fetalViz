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

uploaded_seg, uploaded_struct, df_data, config_model, subject = create_sidebar()

# Print current directory
# st.write(os.getcwd())

if uploaded_seg is not None and uploaded_struct is not None:
    segmentation, structural = load_mri_data(uploaded_seg, uploaded_struct)

    #TODO: add information about the patient in a tabular format

    # process MRI data to be displayed
    image_cropped, mask_cropped = get_cropped_stack_based_on_mask(structural, segmentation, uploaded_seg)

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

    # Dropdown for labels

    label_long = st.selectbox(
        "Choose a label", config_model["segmentation"]["labels_long"].values()
    )

    # invert the dict
    inv_dict = {v: k for k, v in config_model["segmentation"]["labels_long"].items()}

    # for the selected label, get the corresponding key
    label = inv_dict[label_long]

    # select only rows where pathology == Neurotypical
    df = df_data[df_data["pathology"] == "Neurotypical"]
    fig, ax = plt.subplots()
    ax.scatter(df["age"], df[label])
    ax.set_xlabel('Age')
    ax.set_ylabel('Label')
    ax.set_title('Scatter plot of Age vs Label for Neurotypical subjects')

    # extract current subject age and volume from the df_data
    current_subject_age = df_data[df_data["participant_id"] == subject]["age"].values[0]
    current_subject_volume = df_data[df_data["participant_id"] == subject][label].values[0]

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
    ax.set_ylabel(label)
    ax.set_title(f"Volume vs Age for {label_long}")

    # Displaying the plot
    st.pyplot(fig)
