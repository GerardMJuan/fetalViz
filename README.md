# fetalViz
Fetal visualization prototyping. Various prototypes are available.

Code here is very WIP. Both projects are in early stages of development.

## Summary of app.py

The `app.py` file is a Streamlit application for visualizing fetal MRI biomarkers. Here's an overview of its main features:

1. Allows users to upload segmentation and structural MRI data.
2. Displays MRI slices with an option to show segmentation overlay.
3. Provides a slider to adjust segmentation transparency.
4. Includes a dropdown menu to select specific labels for analysis.
5. Generates a scatter plot comparing the selected label's volume against age for neurotypical subjects. (WIP)
6. Highlights the current subject's data point on the scatter plot. (WIP)
7. Utilizes configuration files for segmentation labels and color maps.
8. Integrates with custom modules for data processing and visualization.

The application aims to provide an interactive and informative interface for analyzing fetal MRI data and comparing individual subjects to neurotypical data.

## Summary of recon/app.py

The `recon/app.py` file is a PyQt5 application for visualizing and processing fetal MRI data using the NiftyMIC toolbox. Here's an overview of its main features:

1. Provides a graphical user interface for selecting and viewing multiple MRI files.
2. Displays MRI slices in axial, sagittal, and coronal views with interactive navigation.
3. Allows users to select an output folder for reconstruction results.
4. Offers customizable reconstruction parameters, including denoising, cropping, and resolution settings. (WIP)
5. Implements a multi-step reconstruction process:
   - Creating brain masks
   - Denoising images (currently commented out)
   - Cropping images and masks
   - Running the NiftyMIC reconstruction algorithm
6. Uses a separate thread for the reconstruction process to keep the UI responsive.
7. Displays progress and status messages during the reconstruction process. (WIP)
8. Integrates with Docker to run the NiftyMIC reconstruction tools. (WIP)

The application provides a user-friendly interface for fetal MRI reconstruction, allowing researchers and clinicians to process and visualize fetal brain images more easily.

## LICENSE

This project is licensed under the MIT License. See the LICENSE file for details.
