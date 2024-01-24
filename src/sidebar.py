import streamlit as st
import json
import os
import glob 
import pandas as pd

@st.cache_data
def load_config_model(config_file):
    with open(f'data/{config_file}', 'r') as f:
        config_model = json.load(f)
    return config_model


def create_sidebar():
    # Create a sidebar
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose the app mode",
        ["Show instructions", "Run the app", "Show the source code"],
    )

    # Get all .json files in data/
    config_files = glob.glob('data/*.json')
    config_files = [os.path.basename(file) for file in config_files]

    # Add selectbox to sidebar
    config_file = st.sidebar.selectbox(
        "Choose the configuration model",
        config_files,
    )

    config_model = load_config_model(config_file)

    # load the csv file 
    df_subjects = pd.read_csv(config_model["general"]['csv_path'])

    # extract subjects
    subject_list = df_subjects[config_model["general"]['subject_key']].values

    # create a sidebar selectbox for each subject. Populate with subject list
    subject = st.sidebar.selectbox(
        "Choose the subject",
        subject_list,
        index=None,
    )

    # If no subject is selected, return None for seg_path and T2w_path
    if subject is None:
        seg_path = None
        T2w_path = None
    else:
        # When the user selects the subject, load the corresponding seg and struct
        # files
        # Combine subject path, subject, folder prefix
        subject_path = os.path.join(config_model["general"]['subject_path'], subject, config_model["general"]["folder_prefix"])
        print(subject_path)

        seg_path = os.path.join(subject_path, f"{subject}{config_model['general']['seg_suffix']}")
        T2w_path = os.path.join(subject_path, f"{subject}{config_model['general']['rec_suffix']}")

    return seg_path, T2w_path, df_subjects, config_model
