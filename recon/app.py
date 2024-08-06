import sys
import os
from pathlib import Path
import time
import shutil
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QFileDialog, QVBoxLayout, QHBoxLayout, 
                             QWidget, QLabel, QListWidget, QSlider, QGroupBox, QProgressBar, QTextEdit,
                             QCheckBox, QDoubleSpinBox, QFormLayout)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
from recon import create_brain_masks, denoise_image, crop_images_and_masks, reconstruct_volume

##### NO FA EL DENOISED
###################333333


class MRICanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=5, dpi=100):
        fig = plt.Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MRICanvas, self).__init__(fig)
        self.setParent(parent)
        self.mpl_connect('scroll_event', self.on_scroll)
        self.view_name = ""
        self.current_slice = 0
        self.total_slices = 0
        self.data = None
        self.orientation = None

    def on_scroll(self, event):
        if event.button == 'up':
            self.current_slice = min(self.current_slice + 1, self.total_slices - 1)
        else:
            self.current_slice = max(self.current_slice - 1, 0)
        self.update_slice()

    def update_slice(self):
        if self.data is None:
            return
        axis = self.get_axis()
        
        self.axes.clear()
        if axis == 0:
            self.axes.imshow(np.rot90(self.data[self.current_slice, :, :]), cmap="gray")
        elif axis == 1:
            self.axes.imshow(np.rot90(self.data[:, self.current_slice, :]), cmap="gray")
        else:
            self.axes.imshow(np.rot90(self.data[:, :, self.current_slice]), cmap="gray")
        
        self.axes.set_title(f"{self.view_name} - Slice {self.current_slice + 1}/{self.total_slices}")
        self.axes.axis('off')
        self.draw()

    def get_axis(self):
        if self.orientation:
            try:
                return self.orientation.index(self.view_name[0].upper())
            except ValueError:
                pass
        return {"Axial": 2, "Sagittal": 0, "Coronal": 1}[self.view_name]


class ReconstructionThread(QThread):
    update_progress = pyqtSignal(int)
    update_message = pyqtSignal(str)

    def __init__(self, output_folder, loaded_files, params):
        super().__init__()
        self.output_folder = output_folder
        self.params = params
        self.loaded_files = loaded_files

    def run(self):
        """
        Run the actual reconstruction process

        The process is similar to the one used in run_recon, from multifact-clinic-pipeline github
        1. Create masks
        2. Denoise
        3. Crop the images and masks
        4. Run the reconstruction
        """

        # copy the list of files to a new folder in the output
        # folder named "input"
        input_folder = self.output_folder / "input"
        input_folder.mkdir(exist_ok=True)
        input_list =  []
        for file in self.loaded_files:
            # copy with shutil
            shutil.copy(file, input_folder)
            input_list.append(input_folder / os.path.basename(file))
        
        # prepare output mask names
        mask_folder = self.output_folder / "masks"
        mask_folder.mkdir(exist_ok=True)
        list_of_masks = []
        for file in input_list:
            tmp_mask = file.with_name(file.name.replace(".nii.gz", "_mask.nii.gz"))
            # change parent directory
            tmp_mask = mask_folder / os.path.basename(tmp_mask)
            list_of_masks.append(tmp_mask)
        
        # 1 create masks
        self.update_message.emit(f"Creating brain masks for each file...")
        create_brain_masks(
            input_list,
            list_of_masks,
        )

        # 2 Denoise
        denoise_list = []
        denoise_folder = self.output_folder / "denoised"
        denoise_folder.mkdir(exist_ok=True)

        for file in input_list:
            tmp_denoised = file.with_name(file.name.replace(".nii.gz", "_denoised.nii.gz"))
            tmp_denoised = denoise_folder / os.path.basename(tmp_denoised)
            denoise_list.append(tmp_denoised)
    
        self.update_progress.emit(20)
        self.update_message.emit(f"Denoising images...")
        # for file, denoised in zip(input_list, denoise_list):
            # denoise the image
        #     denoise_image(file, denoised)

        # 3 Crop the images and masks
        cropped_folder = self.output_folder / "cropped"
        cropped_folder_masks = self.output_folder / "cropped_masks"
        cropped_folder.mkdir(exist_ok=True)
        cropped_folder_masks.mkdir(exist_ok=True)
        cropped_images = []
        cropped_masks = []
        for file, mask in zip(denoise_list, list_of_masks):
            tmp_cropped = cropped_folder / os.path.basename(file)
            cropped_images.append(tmp_cropped)

            tmp_cropped_mask = cropped_folder_masks / os.path.basename(mask)
            cropped_masks.append(tmp_cropped_mask)

        self.update_progress.emit(40)
        self.update_message.emit(f"Cropping images and masks...")
        crop_images_and_masks(
            denoise_list,
            list_of_masks,
            cropped_images,
            cropped_masks
        )

        # 4 Run the reconstruction
        self.update_message.emit(f"Running the reconstruction...")
        self.update_progress.emit(60)
        reconstruct_volume(
            cropped_folder,
            cropped_folder_masks,
            self.output_folder,
            "niftymic"
        )

        self.update_progress.emit(100)
        self.update_message.emit(f"Reconstruction process completed.")


class FetalMRIApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fetal MRI Reconstruction")
        self.setGeometry(100, 100, 1400, 400)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QHBoxLayout(self.central_widget)

        self.left_panel = QWidget()
        self.left_layout = QVBoxLayout(self.left_panel)
        self.layout.addWidget(self.left_panel, 1)

        self.file_select_button = QPushButton("Select MRI Files")
        self.file_select_button.clicked.connect(self.select_files)
        self.left_layout.addWidget(self.file_select_button)

        self.file_list = QListWidget()
        self.file_list.itemClicked.connect(self.on_file_selected)
        self.left_layout.addWidget(self.file_list)

        self.output_folder_button = QPushButton("Select Output Folder")
        self.output_folder_button.clicked.connect(self.select_output_folder)
        self.left_layout.addWidget(self.output_folder_button)

        self.output_folder_label = QLabel("No output folder selected")
        self.left_layout.addWidget(self.output_folder_label)

        # Add parameter controls
        self.param_group = QGroupBox("Reconstruction Parameters")
        self.param_layout = QFormLayout()
        self.param_group.setLayout(self.param_layout)

        self.denoising_checkbox = QCheckBox()
        self.param_layout.addRow("Denoising:", self.denoising_checkbox)

        self.cropping_checkbox = QCheckBox()
        self.param_layout.addRow("Cropping:", self.cropping_checkbox)

        self.resolution_spinbox = QDoubleSpinBox()
        self.resolution_spinbox.setRange(0.1, 2.0)
        self.resolution_spinbox.setSingleStep(0.1)
        self.resolution_spinbox.setValue(0.8)
        self.param_layout.addRow("Resolution:", self.resolution_spinbox)

        self.left_layout.addWidget(self.param_group)

        # Add parameter description area
        self.param_description = QTextEdit()
        self.param_description.setReadOnly(True)
        self.param_description.setPlaceholderText("Hover over a parameter for description")
        self.left_layout.addWidget(self.param_description)

        self.run_button = QPushButton("Run Reconstruction")
        self.run_button.clicked.connect(self.run_reconstruction)
        self.run_button.setEnabled(False)
        self.left_layout.addWidget(self.run_button)

        self.progress_bar = QProgressBar()
        self.left_layout.addWidget(self.progress_bar)

        self.message_console = QTextEdit()
        self.message_console.setReadOnly(True)
        self.left_layout.addWidget(self.message_console)

        self.right_panel = QWidget()
        self.right_layout = QVBoxLayout(self.right_panel)
        self.layout.addWidget(self.right_panel, 3)

        self.view_layout = QHBoxLayout()
        self.right_layout.addLayout(self.view_layout)

        self.axial_group = self.create_view_group("Axial")
        self.sagittal_group = self.create_view_group("Sagittal")
        self.coronal_group = self.create_view_group("Coronal")

        self.view_layout.addWidget(self.axial_group)
        self.view_layout.addWidget(self.sagittal_group)
        self.view_layout.addWidget(self.coronal_group)

        self.loaded_files = {}
        self.current_file = None
        self.output_folder = None

        self.setup_parameter_descriptions()

    def create_view_group(self, title):
        group = QGroupBox(title)
        layout = QVBoxLayout(group)
        canvas = MRICanvas(self, width=5, height=5)
        canvas.view_name = title
        layout.addWidget(canvas)
        toolbar = NavigationToolbar(canvas, self)
        layout.addWidget(toolbar)
        return group

    def setup_parameter_descriptions(self):
        descriptions = {
            "Denoising": "Reduces noise in the MRI images, potentially improving image quality.",
            "Cropping": "Automatically removes non-brain tissue from the images.",
            "Resolution": "Sets the voxel size (in mm) for the reconstructed image. Lower values give higher resolution but may increase processing time."
        }

        for param, widget in [("Denoising", self.denoising_checkbox),
                            ("Cropping", self.cropping_checkbox),
                            ("Resolution", self.resolution_spinbox)]:
            widget.setToolTip(descriptions[param])
            widget.enterEvent = lambda event, text=descriptions[param]: self.show_param_description(text)
            widget.leaveEvent = lambda event: self.clear_param_description()

    def show_param_description(self, text):
        self.param_description.setText(text)

    def clear_param_description(self):
        self.param_description.clear()

    def select_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select MRI Files", "", "NIfTI Files (*.nii.gz)")
        for file in files:
            if file not in self.loaded_files:
                self.loaded_files[file] = nib.load(file)
                self.file_list.addItem(os.path.basename(file))
        self.check_run_button_state()  # Check button state after adding files

    def on_file_selected(self, item):
        file_path = next(f for f in self.loaded_files.keys() if os.path.basename(f) == item.text())
        self.current_file = file_path
        self.load_and_display_image(file_path)

    def load_and_display_image(self, file_path):
        img = self.loaded_files[file_path]
        data = img.get_fdata()
        try:
            orientation = nib.aff2axcodes(img.affine)
        except:
            orientation = None
        
        for view, group in zip(["Axial", "Sagittal", "Coronal"], 
                               [self.axial_group, self.sagittal_group, self.coronal_group]):
            canvas = group.findChild(MRICanvas)
            canvas.data = data
            canvas.orientation = orientation
            canvas.total_slices = data.shape[canvas.get_axis()]
            canvas.current_slice = canvas.total_slices // 2
            canvas.update_slice()

    def run_reconstruction(self):
        if not self.loaded_files or self.output_folder is None:
            self.message_console.append("Please select input files and output folder")
       
        params = {
            "denoising": self.denoising_checkbox.isChecked(),
            "cropping": self.cropping_checkbox.isChecked(),
            "resolution": self.resolution_spinbox.value()
        }

        self.message_console.append(f"Starting reconstruction...\nOutput folder: {self.output_folder}")
        self.message_console.append(f"Parameters: {params}")
        self.run_button.setEnabled(False)

        self.reconstruction_thread = ReconstructionThread(self.output_folder, self.loaded_files.keys(), params)
        self.reconstruction_thread.update_progress.connect(self.update_progress_bar)
        self.reconstruction_thread.update_message.connect(self.update_message_console)
        self.reconstruction_thread.finished.connect(self.reconstruction_finished)
        self.reconstruction_thread.start()

    def select_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.output_folder = Path(folder)
            self.output_folder_label.setText(f"Output folder: {self.output_folder}")
            self.check_run_button_state()

    def check_run_button_state(self):
        self.run_button.setEnabled(bool(self.loaded_files) and self.output_folder is not None)

    def update_progress_bar(self, value):
        self.progress_bar.setValue(value)

    def update_message_console(self, message):
        self.message_console.append(message)

    def reconstruction_finished(self):
        self.check_run_button_state()  # Re-enable the button if conditions are met
        self.message_console.append(f"Output files are stored in: {self.output_folder}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FetalMRIApp()
    window.show()
    sys.exit(app.exec_())