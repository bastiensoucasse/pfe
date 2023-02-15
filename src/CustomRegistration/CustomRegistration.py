"""
The Custom Registration module for Slicer provides the features for 3D images registration, based on the ITK library.
"""

import datetime

import numpy as np
import SimpleITK as sitk
import vtk
from ctk import ctkComboBox
from qt import QLabel, QMessageBox, QPushButton, QSlider, QSpinBox
from slicer import mrmlScene, util, vtkMRMLScalarVolumeNode, vtkMRMLScene
from slicer.ScriptedLoadableModule import ScriptedLoadableModule, ScriptedLoadableModuleLogic, ScriptedLoadableModuleWidget


class CustomRegistration(ScriptedLoadableModule):
    """
    Main class for the Custom Registration module used to define the module's metadata.
    """

    def __init__(self, parent) -> None:
        ScriptedLoadableModule.__init__(self, parent)

        # Set the module metadata.
        self.set_metadata()

    def set_metadata(self) -> None:
        """
        Sets the metadata for the module, i.e., the title, categories, contributors, help text, and acknowledgement text.
        """

        self.parent.title = "CustomRegistration"
        self.parent.categories = ["PFE"]
        self.parent.dependencies = []
        self.parent.contributors = ["Wissam Boussella (Université de Bordeaux)", "Iantsa Provost (Université de Bordeaux)", "Bastien Soucasse (Université de Bordeaux)", "Tony Wolff (Université de Bordeaux)"]
        self.parent.helpText = "This module provides features for 3D images registration, based on the ITK library."
        self.parent.acknowledgementText = "This project is supported and supervised by the reseacher and professor Fabien Baldacci (Université de Bordeaux)."


class CustomRegistrationLogic(ScriptedLoadableModuleLogic):
    """
    Logic class for the Custom Registration module used to define the module's algorithms.
    """

    def __init__(self) -> None:
        ScriptedLoadableModuleLogic.__init__(self)

    #
    # ROI SELECTION
    #

    def select_roi(self, image: sitk.Image, threshold: int) -> sitk.Image:
        """
        Selects as ROI the largest connected component after a threshold.

        Parameters:
            image: The SimpleITK image.
            threshold: The threshold value.

        Returns:
            The ROI SimpleITK image.
        """

        # Apply threshold filter.
        binary_image = sitk.BinaryThreshold(image, lowerThreshold=threshold, upperThreshold=1000000)

        # Apply connected component filter.
        label_map = sitk.ConnectedComponent(binary_image)

        # Find largest connected component.
        label_shape_stats = sitk.LabelShapeStatisticsImageFilter()
        label_shape_stats.Execute(label_map)

        largest_label = 1
        max_volume = 0
        for label in range(1, label_shape_stats.GetNumberOfLabels() + 1):
            volume = label_shape_stats.GetPhysicalSize(label)
            if volume > max_volume:
                max_volume = volume
                largest_label = label

        # Use binary image of largest connected component as ROI.
        roi_binary = sitk.BinaryThreshold(label_map, lowerThreshold=largest_label, upperThreshold=largest_label)
        roi = sitk.Mask(image, roi_binary)
        return roi

    #
    # CROPPING
    #

    def crop(self, image: sitk.Image, start_val, size) -> sitk.Image:
        """
        Crops a volume using the selected algorithm.

        Parameters:
            image: The SimpleITK image to be cropped.
            start_val: The start index of the cropping region.
            size: The size of the cropping region.

        Returns:
            The cropped SimpleITK image.
        """

        crop_filter = sitk.ExtractImageFilter()
        crop_filter.SetSize(size)
        crop_filter.SetIndex(start_val)
        cropped_image = crop_filter.Execute(image)
        return cropped_image

    #
    # UTILITIES
    #

    def run(self, script_file: str, function_name: str, *args, **kwargs):
        """
        Loads and runs a script file.

        Parameters:
            script_file: The script file.
            function_name: The name of the function to run.
            args: Positional arguments to pass.
            kwargs: Keyword arguments to pass.

        Returns:
            The result returned.
        """

        # Load the script file.
        with open(script_file, "r") as f:
            code = f.read()
            exec(code, globals())

        # Retrieve the function.
        function = globals()[function_name]

        # Run the function.
        return function(*args, **kwargs)

    def cleanup(self) -> None:
        """
        Cleans up the module.

        :IDK: Could change class.
        """

        # Remove the observer.
        mrmlScene.RemoveObserver(self.observerTag)


class CustomRegistrationWidget(ScriptedLoadableModuleWidget):
    """
    Widget class for the Custom Registration module used to define the module's panel interface.

    :TODO: Homogenise the names of variables (image, node, volume)
    """

    def __init__(self, parent=None) -> None:
        ScriptedLoadableModuleWidget.__init__(self, parent)

    def setup(self) -> None:
        """
        Sets up the widget for the module by adding a welcome message to the layout.
        """

        # Initialize the widget.
        ScriptedLoadableModuleWidget.setup(self)

        # Initialize the logic.
        self.logic = CustomRegistrationLogic()

        # Load the UI.
        self.panel_ui = util.loadUI(self.resourcePath("UI/Panel.ui"))
        self.layout.addWidget(self.panel_ui)

        # Setup the preprocessing.
        self.preprocessing_setup()
        self.roi_selection_setup()
        self.cropping_setup()
        self.resampling_setup()

    #
    # PREPROCESSING
    #

    def preprocessing_setup(self):
        """
        Sets up the preprocessing widget by retrieving the volume selection widget and initializing it.
        """

        # Retrieve the volumes in memory.
        self.volumes = mrmlScene.GetNodesByClass("vtkMRMLScalarVolumeNode")

        # Initialize the preprocessing selected volume.
        self.currentVolume = None
        self.currentVolumeIndex = -1

        # Get the selected volume combo box.
        self.volumeComboBox = self.panel_ui.findChild(ctkComboBox, "SelectedVolumeComboBox")
        assert self.volumeComboBox

        # Connect the selected volume combo box to its "on cahnged" function.
        self.volumeComboBox.activated.connect(self.on_volume_activated)

        # Fill the selected volume combo box with the available volumes and utility options.
        for i in range(self.volumes.GetNumberOfItems()):
            volume = self.volumes.GetItemAsObject(i)
            self.volumeComboBox.addItem(volume.GetName())
        self.volumeComboBox.addItem("Rename current volume…")
        self.volumeComboBox.addItem("Delete current volume…")
        self.volumeComboBox.setCurrentIndex(-1)

        # Add observer to update combobox when new volume is added to MRML Scene.
        self.observerTag = mrmlScene.AddObserver(vtkMRMLScene.NodeAddedEvent, self.update_volume_list)

    #
    # ROI SELECTION
    #

    def roi_selection_setup(self):
        """
        Sets up the ROI selection widget.
        """

        # Initialize the ROI.
        self.selected_volume_roi = None

        # Get the ROI selection threshold slider.
        self.roi_selection_threshold_slider = self.panel_ui.findChild(QSlider, "ROISelectionThresholdSlider")
        assert self.roi_selection_threshold_slider

        # Get the ROI selection threshold value label.
        self.roi_selection_threshold_value_label = self.panel_ui.findChild(QLabel, "ROISelectionThresholdValueLabel")
        assert self.roi_selection_threshold_value_label

        # Connect the ROI selection threshold slider to its "on changed" function.
        self.roi_selection_threshold_slider.valueChanged.connect(self.on_roi_selection_threshold_slider_value_changed)

        # Get the ROI selection button.
        self.roi_selection_button = self.panel_ui.findChild(QPushButton, "ROISelectionButton")
        assert self.roi_selection_button

        # Connect the ROI selection button to the algorithm.
        self.roi_selection_button.clicked.connect(self.select_roi)

    def on_roi_selection_threshold_slider_value_changed(self):
        """
        Updates the ROI selection threshold value to match the slider.

        Called when the ROI selection threshold slider value is changed.
        """

        # Retrieve the threshold value.
        threshold = self.roi_selection_threshold_slider.value

        # Update the label accordingly.
        self.roi_selection_threshold_value_label.setText(str(threshold))

    def reset_roi(self) -> None:
        """
        Resets the ROI selection.
        """

        # Reset the ROI selection.
        self.selected_volume_roi = None

        # Update the label accordingly.
        self.roi_selection_threshold_slider.setValue(0)
        self.roi_selection_threshold_value_label.setText("0")

    def select_roi(self):
        """
        Selects the ROI.
        """

        # Ensure that a volume is selected.
        if not self.currentVolume:
            self.error_message("Please select a volume to select a ROI from.")
            return

        # Retrieve the threshold value.
        threshold = self.roi_selection_threshold_slider.value

        # Call the ROI selection algorithm.
        self.selected_volume_roi = self.logic.select_roi(self.vtk_to_sitk(self.currentVolume), threshold)

        # Log the ROI selection.
        print(f'ROI has been selected with a threshold value of {threshold} in "{self.currentVolume.GetName()}".')

    #
    # CROPPING
    #
    # :TODO: Sort functions and move utilities to the end.
    #

    def cropping_setup(self):
        """
        Sets up the cropping widget by retrieving the crop button and coordinates input widgets.
        """

        # :COMMENT: Get the crop button widget.
        self.cropButton = self.panel_ui.findChild(QPushButton, "crop_button")
        self.cropButton.clicked.connect(self.cropping)

        # :COMMENT: Get the coordinates entered by user in spinbox widgets.
        self.start = []
        self.end = []
        for i in ["x", "y", "z"]:
            self.start.append(self.panel_ui.findChild(QSpinBox, "s" + i))
            self.end.append(self.panel_ui.findChild(QSpinBox, "e" + i))

    def on_volume_activated(self, index):
        """
        Called when an item in the volume combobox is selected.
        Handles selection of a volume and updates the cropping parameters accordingly, and edition of the volume combobox.
        """

        options = ["Delete current volume…", "Rename current volume…"]

        name = self.volumeComboBox.currentText
        if name in options:
            # :TODO: Add renaming and deleting features.
            if name == "Delete current volume…":
                print("[DEBUG]", name, "option not implemented yet!\n")

                # print(
                #     f"\"{self.currentVolume.GetName()}\" has been deleted."
                # )

            if name == "Rename current volume…":
                print("[DEBUG]", name, "option not implemented yet!\n")

                # old_name = self.currentVolume.GetName()
                # # ...
                # print(
                #     f"\"{old_name}\" has been renamed as {self.currentVolume.GetName()}."
                # )

            self.currentVolumeIndex = -1

        else:
            self.currentVolumeIndex = index
            self.currentVolume = self.volumes.GetItemAsObject(index)
            volume_data = self.currentVolume.GetImageData()
            self.currentVolumeDim = volume_data.GetDimensions()
            self.update_dimensions_display()

            for i in range(3):
                self.start[i].setMaximum(self.currentVolumeDim[i])
                self.end[i].setMaximum(self.currentVolumeDim[i])

            # :COMMENT: Remove the previously computed ROI.
            self.reset_roi()

            # :COMMENT: Update the available target volumes list.
            self.update_resampling_available_targets()

            # :COMMENT: Log the selected volume change.
            print(f'"{self.currentVolume.GetName()}" has been selected.')

    def vtk_to_sitk(self, vtk_image):
        """
        Converts a VTK Volume Node to a SimpleITK image.
        :param vtk_image: VTK Volume Node to be converted.
        :return: SimpleITK image.
        """

        vtk_image_data = vtk_image.GetImageData()
        np_array = vtk.util.numpy_support.vtk_to_numpy(
            vtk_image_data.GetPointData().GetScalars()
        )
        np_array = np.reshape(np_array, (vtk_image_data.GetDimensions()[::-1]))
        sitk_image = sitk.GetImageFromArray(np_array)
        return sitk_image

    def sitk_to_vtk(self, sitk_image):
        """
        Converts a SimpleITK image to a VTK Volume Node.

        Parameters:
            sitk_image: SimpleITK image to be converted.

        Returns:
            VTK Volume Node.
        """

        np_array = sitk.GetArrayFromImage(sitk_image)
        vtk_image_data = vtk.vtkImageData()
        vtk_image_data.SetDimensions(np_array.shape[::-1])
        vtk_image_data.AllocateScalars(vtk.VTK_FLOAT, 1)
        vtk_array = vtk.util.numpy_support.numpy_to_vtk(np_array.flatten())
        vtk_image_data.GetPointData().SetScalars(vtk_array)
        vtk_image = vtkMRMLScalarVolumeNode()
        vtk_image.SetAndObserveImageData(vtk_image_data)
        return vtk_image

    # :GLITCH: [VTK] Warning: In /Volumes/D/S/S-0/Modules/Loadable/VolumeRendering/Logic/vtkSlicerVolumeRenderingLogic.cxx, line 674
    #          [VTK] vtkSlicerVolumeRenderingLogic (0x600002bb0420): CopyDisplayToVolumeRenderingDisplayNode: No display node to copy.
    # This error appears when the cropped volume is selected in the volume rendering module but it is displayed though.
    def cropping(self):
        """
        Crops a volume using the selected algorithm.
        """

        # :COMMENT: Do not do anything if crop button clicked but no volume selected.
        # :TODO: Solve, knowing that it works for the "Select a Volume" case.
        if self.currentVolumeIndex < 0:
            return

        # :COMMENT: Ensure that a volume is selected.
        assert self.currentVolume

        # :COMMENT: Retrieve coordinates input.
        start_val = []
        end_val = []
        for i in range(3):
            start_val.append(self.start[i].value)
            end_val.append(self.end[i].value)

        # :COMMENT: Check that coordinates are valid.
        if any(end_val[i] < start_val[i] for i in range(3)):
            self.error_message(
                "End values must be greater than or equal to start values."
            )
            return

        # :COMMENT: Save selected volume's data, and convert the volume to a SimpleITK image.
        data_backup = [
            self.currentVolume.GetSpacing(),
            self.currentVolume.GetOrigin(),
            vtk.vtkMatrix4x4(),
        ]
        self.currentVolume.GetIJKToRASDirectionMatrix(data_backup[2])
        sitk_image = self.vtk_to_sitk(self.currentVolume)

        # :COMMENT: Get the size of the crop region.
        size = [end_val[i] - start_val[i] + 1 for i in range(3)]

        # :COMMENT: Crop the image.
        cropped_image = self.logic.crop(sitk_image, start_val, size)

        # :COMMENT: Convert the cropped SimpleITK image back to a VTK Volume Node.
        vtk_image = self.sitk_to_vtk(cropped_image)

        # :COMMENT: Set the new volume's data with the original volume's data.
        vtk_image.SetSpacing(data_backup[0])
        vtk_image.SetOrigin(data_backup[1])
        vtk_image.SetIJKToRASDirectionMatrix(data_backup[2])

        # :COMMENT: Add the VTK Volume Node to the scene.
        self.add_new_volume(vtk_image, "cropped")

        new_size = vtk_image.GetImageData().GetDimensions()
        print(
            f'"{self.currentVolume.GetName()}" has been cropped to size ({new_size[0]}x{new_size[1]}x{new_size[2]}) as "{vtk_image.GetName()}".'
        )

    def update_volume_list(self, caller, event):
        """
        Updates the list of volumes in the volume combobox when a change is detected in the MRML Scene.

        Parameters:
            caller: The widget calling this method.
            event: The event that triggered this method.
        """

        node = caller.GetNthNode(caller.GetNumberOfNodes() - 1)
        numberOfNonNodes = 2  # number of non-node options in the combobox
        if node.IsA("vtkMRMLVolumeNode"):
            newNodeIndex = self.volumeComboBox.count - numberOfNonNodes
            self.volumeComboBox.insertItem(newNodeIndex, node.GetName())
            self.volumes.AddItem(node)

    def update_dimensions_display(self):
        """
        Updates the display of the dimensions according to the selected volume.
        """

        dim_label = self.panel_ui.findChild(QLabel, "SelectedVolumeDimensionsValueLabel")
        dim_label.setText(
            "{} x {} x {}".format(
                self.currentVolumeDim[0],
                self.currentVolumeDim[1],
                self.currentVolumeDim[2],
            )
        )

    def add_new_volume(self, volume, name: str):
        """
        Adds a new volume to the scene.

        Parameters:
            volume: VTK Volume Node to be added.
        """

        # :COMMENT: Ensure that a volume is selected.
        assert self.currentVolume

        # :COMMENT: Generate and assign a unique name to the volume.
        current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        new_name = f"{self.currentVolume.GetName()}_{name}_{current_time}"
        volume.SetName(new_name)

        # :COMMENT: Update the MRML scene.
        mrmlScene.AddNode(volume)

        # :COMMENT: Useful for exporting.
        # sitk.WriteImage(cropped_image, '/Users/iantsaprovost/Desktop/test.nrrd')

    def error_message(self, message):
        """
        Displays an error message.

        Parameters:
            message: Message to be displayed.
        """

        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setText(message)
        msg.setWindowTitle("Error")
        msg.exec_()

    # :BUG: None of these versions are working.
    # def rename_volume(self):
    #     volume = self.volumes.GetItemAsObject(self.currentVolumeIndex)
    #     new_name, ok = QInputDialog.getText(
    #         self,
    #         "Rename Volume",
    #         "Enter the new name for the volume:",
    #         text=volume.GetName(),
    #     )
    #     if ok and new_name != "":
    #         volume.SetName(new_name)

    # def rename_volume(self):
    #     print(self.currentVolumeIndex)
    #     self.input_dialog = QInputDialog(None)
    #     self.input_dialog.setWindowTitle('Rename Volume')
    #     self.input_dialog.setLabelText('Enter new name:')
    #     self.input_dialog.setModal(True)
    #     self.input_dialog.finished.connect(self.handleRenameVolume)
    #     self.input_dialog.show()

    # def handleRenameVolume(self, result):
    #     print(self.currentVolumeIndex)
    #     if result == QDialog.Accepted:
    #         new_name = self.input_dialog.textValue()
    #         print(self.currentVolumeIndex)
    #         selected_volume = self.volumes.GetItemAsObject(self.currentVolumeIndex)
    #         print(self.currentVolumeIndex)
    #         print(selected_volume)
    #         selected_volume.SetName(new_name)

    # def rename_volume(self):
    # new_name, ok = QInputDialog.getText(
    #     self, "Rename Volume", "Enter new volume name:", QLineEdit.Normal
    # )
    # if ok:
    #     selected_index = self.volumeComboBox.currentIndex()
    #     if selected_index > 0:
    #         selected_volume = self.volumes.GetItemAsObject(selected_index - 1)
    #         selected_volume.SetName(new_name)
    #         self.volumeComboBox.setItemText(selected_index, new_name)
    #         self.currentVolumeIndex = selected_index
    # selected_index = self.volumeComboBox.currentIndex
    # selected_volume = self.volumes.GetItemAsObject(selected_index)
    # print(selected_volume)

    #
    # RESAMPLING
    #

    def resampling_setup(self):
        """
        Sets up the resampling widget by linking the UI to the scene and algorithm.
        """

        # Initialize the resampling target volume.
        self.resampling_target_volume = None

        # Get the resampling target volume combo box.
        self.resampling_target_volume_combo_box = self.panel_ui.findChild(ctkComboBox, "ResamplingTargetVolumeComboBox")
        assert self.resampling_target_volume_combo_box

        # Connect the resampling target volume combo box to its "on changed" function.
        self.resampling_target_volume_combo_box.activated.connect(self.on_resampling_target_volume_changed)

        # Fill the resampling target volume combo box with the available resampling targets.
        self.update_resampling_available_targets()

        # Get the resampling button.
        self.resampling_button = self.panel_ui.findChild(QPushButton, "ResamplingButton")
        assert self.resampling_button

        # Connect the resampling button to the algorithm.
        self.resampling_button.clicked.connect(self.resample)

    def on_resampling_target_volume_changed(self):
        """
        Updates the resampling target volumes in memory.

        Called when the resampling target volumes combo box is changed.
        """

        # Retrieve the resampling target volume name in the combo box.
        name = self.resampling_target_volume_combo_box.currentText

        # Update the resampling target volume.
        self.resampling_target_volume = self.get_volume_by_name(name)

        # Log the resampling target volume change.
        print(f'Resampling Target Volume: "{name}."')

    def update_resampling_available_targets(self):
        """
        Updates the list of available resampling targets in the resampling target volume combo box.
        """

        # Reset the resampling target volume.
        self.resampling_target_volume = None

        # Clear the resampling target volume combo box.
        self.resampling_target_volume_combo_box.clear()

        # Add the available volumes to the resampling target volume combo box.
        for i in range(self.volumes.GetNumberOfItems()):
            name = self.volumes.GetItemAsObject(i).GetName()
            if not self.currentVolume or self.currentVolume.GetName() != name:
                self.resampling_target_volume_combo_box.addItem(name)
        self.resampling_target_volume_combo_box.setCurrentIndex(-1)

    def resample(self):
        """
        Retrieves the selected and target volumes and runs the resampling algorithm.
        """

        # Ensure that a volume is selected as well as a resampling target volume.
        if not self.currentVolume:
            self.error_message("Please select a volume to resample.")
            return
        if not self.resampling_target_volume:
            self.error_message("Please select a resampling target volume.")
            return

        # Call the resampling algorithm.
        resampled_volume = self.sitk_to_vtk(self.logic.run(
            self.resourcePath("Scripts/Resampling.py"),
            "resample",
            self.vtk_to_sitk(self.currentVolume),
            self.vtk_to_sitk(self.resampling_target_volume),
        ))

        # Transfer the volume metadata.
        self.transfer_volume_metadata(self.currentVolume, resampled_volume)

        # Save the resampled volume.
        self.add_new_volume(resampled_volume, "resampled")

        # Log the resampling.
        print(f'"{self.currentVolume.GetName()}" has been resampled to match "{self.resampling_target_volume.GetName()}" as "{resampled_volume.GetName()}".')

    #
    # UTILITIES
    #

    def get_volume_by_name(self, name: str):
        """
        Retrieves a volume by its name.

        Parameters:
            name: The name of the volume to retrieve.

        Returns:
            The VTK volume.
        """

        # Search for the volume by its name.
        for i in range(self.volumes.GetNumberOfItems()):
            volume = self.volumes.GetItemAsObject(i)
            if volume.GetName() == name:
                return volume

        # Return None if the volume was not found.
        return None

    def transfer_volume_metadata(self, source_volume: vtkMRMLScalarVolumeNode, target_volume: vtkMRMLScalarVolumeNode) -> None:
        """
        Copies the metadata from the source volume to the target volume.

        Parameters:
            source_volume: The volume to copy the metadata from.
            target_volume: The volume to copy the metadata to.
        """

        # Retrieve the metadata from the source volume.
        spacing = source_volume.GetSpacing()
        origin = source_volume.GetOrigin()
        ijk_to_ras_direction_matrix = vtk.vtkMatrix4x4()
        source_volume.GetIJKToRASDirectionMatrix(ijk_to_ras_direction_matrix)

        # Apply the metadata to the target volume.
        target_volume.SetSpacing(spacing)
        target_volume.SetOrigin(origin)
        target_volume.SetIJKToRASDirectionMatrix(ijk_to_ras_direction_matrix)
