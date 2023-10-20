# Import necessary libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Load data
@st.cache_data  # This decorator helps to cache the data and prevents reload on every interaction
def load_data():
    df = pd.read_csv(
        "/media/gerard/HDD/MULTIFACT_DATA/dhcp_anat_pipeline/measurements/reports/pipeline_all_measures.csv"
    )  # Replace with your csv file path
    df["age at scan"] = (
        df["age at scan"].round(0).astype(int)
    )  # Round age values to nearest integer
    return df


df = load_data()

# Get subjects id
subj_list = df["subject ID"].values

# Sidebar
st.sidebar.header("Select Biomarker")
biomarker = st.sidebar.selectbox(
    "Biomarker", df.columns[3:]
)  # Replace df.columns with your specific biomarker columns if necessary

subj = st.sidebar.selectbox(
    "Subject selection", subj_list
)  # Replace df.columns with your specific biomarker columns if necessary

# Main
st.title("Patient Biomarker Analysis Over Time")
st.write(f"Plotting values for: {biomarker}")
st.write(f"For subject X: {biomarker}")

# Create figure
plt.figure(figsize=(10, 6))
sns.lineplot(
    x="age at scan", y=biomarker, data=df, ci="sd", linewidth=0.5
)  # Replace 'sd' with necessary CI type

sns.scatterplot(
    x="age at scan", y=biomarker, data=df[df["subject ID"] == subj], color="red"
)  # Replace 'sd' with necessary CI type

plt.title(f"Trajectory of {biomarker} over time")
plt.xlabel("Age at Scan")
plt.ylabel(f"{biomarker} Value")


# Display plot in streamlit
st.pyplot(plt.gcf())
