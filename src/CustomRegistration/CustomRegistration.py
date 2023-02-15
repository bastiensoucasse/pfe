"""
The Custom Registration module for Slicer provides the features for 3D images registration, based on the ITK library.
"""

import datetime

import numpy as np
import SimpleITK as sitk
import vtk
from ctk import ctkComboBox
from qt import (
    QDialog,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
)
from slicer import mrmlScene, util, vtkMRMLScalarVolumeNode, vtkMRMLScene
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleWidget,
)


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
        self.parent.contributors = [
            "Wissam Boussella (Université de Bordeaux)",
            "Iantsa Provost (Université de Bordeaux)",
            "Bastien Soucasse (Université de Bordeaux)",
            "Tony Wolff (Université de Bordeaux)",
        ]
        self.parent.helpText = "This module provides features for 3D images registration, based on the ITK library."
        self.parent.acknowledgementText = "This project is supported and supervised by the reseacher and professor Fabien Baldacci (Université de Bordeaux)."


class CustomRegistrationLogic(ScriptedLoadableModuleLogic):
    """
    Logic class for the Custom Registration module used to define the module's algorithms.
    """

    def __init__(self) -> None:
        ScriptedLoadableModuleLogic.__init__(self)

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
        binary_image = sitk.BinaryThreshold(
            image, lowerThreshold=threshold, upperThreshold=1000000
        )

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
        roi_binary = sitk.BinaryThreshold(
            label_map, lowerThreshold=largest_label, upperThreshold=largest_label
        )
        roi = sitk.Mask(image, roi_binary)
        return roi

    def crop(self, image: sitk.Image, index, size) -> sitk.Image:
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
        crop_filter.SetIndex(index)
        crop_filter.SetSize(size)
        cropped_image = crop_filter.Execute(image)
        return cropped_image

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


class CustomRegistrationWidget(ScriptedLoadableModuleWidget):
    """
    Widget class for the Custom Registration module used to define the module's panel interface.
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
        assert self.logic

        # Load the UI.
        self.panel = util.loadUI(self.resourcePath("UI/Panel.ui"))
        assert self.panel
        self.layout.addWidget(self.panel)

        # Setup the preprocessing.
        self.volume_selection_setup()
        self.roi_selection_setup()
        self.cropping_setup()
        self.resampling_setup()

    #
    # VOLUME SELECTION
    #

    def volume_selection_setup(self) -> None:
        """
        Sets up the preprocessing widget by retrieving the volume selection widget and initializing it.
        """

        # Retrieve the volumes in memory.
        self.volumes = mrmlScene.GetNodesByClass("vtkMRMLScalarVolumeNode")
        assert self.volumes

        # Initialize the preprocessing selected volume.
        self.selected_volume = None
        self.selection_is_option = False

        # Get the selected volume combo box.
        self.selected_volume_combo_box = self.panel.findChild(
            ctkComboBox, "SelectedVolumeComboBox"
        )
        assert self.selected_volume_combo_box

        # Connect the selected volume combo box to its "on changed" function.
        self.selected_volume_combo_box.activated.connect(
            self.on_selected_volume_combo_box_changed
        )

        # Fill the selected volume combo box with the available volumes and utility options.
        for i in range(self.volumes.GetNumberOfItems()):
            volume = self.volumes.GetItemAsObject(i)
            self.selected_volume_combo_box.addItem(volume.GetName())
        self.selected_volume_combo_box.addItem("Rename current volume…")
        self.selected_volume_combo_box.addItem("Delete current volume…")
        self.selected_volume_combo_box.setCurrentIndex(-1)

        # Add observer to update combobox when new volume is added to MRML Scene.
        self.observer_tag = mrmlScene.AddObserver(
            vtkMRMLScene.NodeAddedEvent, self.update_volume_list
        )

        # :TODO:Iantsa: Fix/remove commented code.
        # self.observer_tag_3 = mrmlScene.AddObserver(
        #     vtkMRMLScene.NodeRemovedEvent, self.update_volume_list
        # )

    def on_selected_volume_combo_box_changed(self, index: int) -> None:
        """
        Handles selection of a volume and updates the cropping parameters accordingly, and edition of the volume combobox.

        Called when an item in the volume combobox is selected.

        Parameters:
            index: The selected volume index.
        """

        OPTIONS = ["Delete current volume…", "Rename current volume…"]

        # :COMMENT: Retrieve the selection text.
        name = self.selected_volume_combo_box.currentText

        # :COMMENT: Handle the different options.
        if name in OPTIONS:
            self.selection_is_option = True

            if self.volumes.GetNumberOfItems() < 1:
                self.display_error_message("No volumes imported.")
                self.update_selected_volume_combo_box()
                return

            if not self.selected_volume:
                self.display_error_message("Please select a volume first.")
                self.update_selected_volume_combo_box()
                return

            if name == "Rename current volume…":
                self.rename_volume()

            if name == "Delete current volume…":
                self.delete_volume()

        # :COMMENT: Select a new volume.
        else:
            self.selection_is_option = False

            self.selected_volume_index = index
            self.selected_volume = self.volumes.GetItemAsObject(index)
            assert self.selected_volume
            volume_data = self.selected_volume.GetImageData()
            self.selected_volume_dimensions = volume_data.GetDimensions()
            assert self.selected_volume_dimensions
            self.update_dimensions_display()

            for i in range(3):
                self.cropping_start[i].setMaximum(self.selected_volume_dimensions[i])
                self.cropping_end[i].setMaximum(self.selected_volume_dimensions[i])

            # :COMMENT: Remove the previously computed ROI.
            self.reset_roi()

            # :COMMENT: Reset the cropping parameters.
            self.reset_cropping_values()

            # :COMMENT: Update the available target volumes list.
            self.update_resampling_available_targets()

            # :TODO:Bastien: Update the view (upper left visualization).

            # :COMMENT: Log the selected volume change.
            print(f'"{self.selected_volume.GetName()}" has been selected.')

    def rename_volume(self) -> None:
        """
        Loads the renaming feature with a minimal window.
        """

        assert self.selected_volume
        self.renaming_old_name = self.selected_volume.GetName()
        self.renaming_input_dialog = QInputDialog(None)
        self.renaming_input_dialog.setWindowTitle("Rename Volume")
        self.renaming_input_dialog.setLabelText("Enter the new name:")
        self.renaming_input_dialog.setModal(True)
        self.renaming_input_dialog.setTextValue(self.selected_volume.GetName())
        self.renaming_input_dialog.finished.connect(self.handle_rename_volume)
        self.renaming_input_dialog.show()

    def handle_rename_volume(self, result) -> None:
        """
        Applies the renaming of the selected volume.

        Parameters:
            result: The result of the input dialog.
        """

        if result == QDialog.Accepted:
            assert self.selected_volume
            new_name = self.renaming_input_dialog.textValue()
            self.selected_volume.SetName(new_name)
            print(
                f'"{self.renaming_old_name}" has been renamed to "{self.selected_volume.GetName()}".'
            )
        self.update_selected_volume_combo_box()

    def update_selected_volume_combo_box(self) -> None:
        """
        Updates the selected volume combo box.
        """

        if self.selected_volume and self.selected_volume_index >= 0:
            self.selected_volume_combo_box.setItemText(
                self.selected_volume_index, self.selected_volume.GetName()
            )
        else:
            self.selected_volume_combo_box.setCurrentIndex(-1)
            self.selected_volume_dimensions.setText("…")

    def delete_volume(self) -> None:
        """
        Deletes the currently selected volume.
        """

        # :TODO:Iantsa: Implement the deletion of the currently selected volume.

        raise NotImplementedError

    def update_dimensions_display(self) -> None:
        """
        Updates the display of the dimensions according to the selected volume.
        """

        dim_label = self.panel.findChild(QLabel, "SelectedVolumeDimensionsValueLabel")
        dim_label.setText(
            "{} x {} x {}".format(
                self.selected_volume_dimensions[0],
                self.selected_volume_dimensions[1],
                self.selected_volume_dimensions[2],
            )
        )

    def update_volume_list(self, caller, event) -> None:
        """
        Updates the list of volumes in the volume combobox when a change is detected in the MRML Scene.

        Parameters:
            caller: The widget calling this method.
            event: The event that triggered this method.
        """

        # :TODO:Iantsa: Fix the unused second parameter.

        node = caller.GetNthNode(caller.GetNumberOfNodes() - 1)
        numberOfNonNodes = 2  # number of non-node options in the combobox
        if node.IsA("vtkMRMLVolumeNode"):
            newNodeIndex = self.selected_volume_combo_box.count - numberOfNonNodes
            self.selected_volume_combo_box.insertItem(newNodeIndex, node.GetName())
            self.volumes.AddItem(node)

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
        self.roi_selection_threshold_slider = self.panel.findChild(
            QSlider, "ROISelectionThresholdSlider"
        )
        assert self.roi_selection_threshold_slider

        # Get the ROI selection threshold value label.
        self.roi_selection_threshold_value_label = self.panel.findChild(
            QLabel, "ROISelectionThresholdValueLabel"
        )
        assert self.roi_selection_threshold_value_label

        # Connect the ROI selection threshold slider to its "on changed" function.
        self.roi_selection_threshold_slider.valueChanged.connect(
            self.on_roi_selection_threshold_slider_value_changed
        )

        # Get the ROI selection button.
        self.roi_selection_button = self.panel.findChild(
            QPushButton, "ROISelectionButton"
        )
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
        if not self.selected_volume:
            self.display_error_message("Please select a volume to select a ROI from.")
            return

        # Retrieve the threshold value.
        threshold = self.roi_selection_threshold_slider.value

        # Call the ROI selection algorithm.
        self.selected_volume_roi = self.logic.select_roi(
            self.vtk_to_sitk(self.selected_volume), threshold
        )

        # Log the ROI selection.
        print(
            f'ROI has been selected with a threshold value of {threshold} in "{self.selected_volume.GetName()}".'
        )

    #
    # CROPPING
    #

    def cropping_setup(self) -> None:
        """
        Sets up the cropping widget by retrieving the crop button and coordinates input widgets.
        """

        # :COMMENT: Get the crop button widget.
        self.cropping_button = self.panel.findChild(QPushButton, "crop_button")
        assert self.cropping_button
        self.cropping_button.clicked.connect(self.crop)

        # :COMMENT: Get the coordinates entered by user in spinbox widgets.
        self.cropping_start = []
        self.cropping_end = []
        for i in ["x", "y", "z"]:
            self.cropping_start.append(self.panel.findChild(QSpinBox, "s" + i))
            self.cropping_end.append(self.panel.findChild(QSpinBox, "e" + i))

    def crop(self) -> None:
        """
        Crops a volume using the selected algorithm.
        """

        # :GLITCH: An error appears when the cropped volume is selected in the volume rendering module but it is displayed though.
        # [VTK] Warning: In /Volumes/D/S/S-0/Modules/Loadable/VolumeRendering/Logic/vtkSlicerVolumeRenderingLogic.cxx, line 674
        # [VTK] vtkSlicerVolumeRenderingLogic (0x600002bb0420): CopyDisplayToVolumeRenderingDisplayNode: No display node to copy.

        # :COMMENT: Ensure that a volume is selected.
        if not self.selected_volume:
            self.display_error_message("Please select a volume to crop.")
            return

        # :COMMENT: Retrieve coordinates input.
        start_val = []
        end_val = []
        for i in range(3):
            start_val.append(self.cropping_start[i].value)
            end_val.append(self.cropping_end[i].value)

        # :COMMENT: Check that coordinates are valid.
        if any(end_val[i] < start_val[i] for i in range(3)):
            self.display_error_message(
                "End values must be greater than or equal to start values."
            )
            return

        # :COMMENT: Save selected volume's data, and convert the volume to a SimpleITK image.
        data_backup = [
            self.selected_volume.GetSpacing(),
            self.selected_volume.GetOrigin(),
            vtk.vtkMatrix4x4(),
        ]
        self.selected_volume.GetIJKToRASDirectionMatrix(data_backup[2])
        sitk_image = self.vtk_to_sitk(self.selected_volume)

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

        # :COMMENT: Log the cropping.
        new_size = vtk_image.GetImageData().GetDimensions()
        print(
            f'"{self.selected_volume.GetName()}" has been cropped to size ({new_size[0]}x{new_size[1]}x{new_size[2]}) as "{vtk_image.GetName()}".'
        )

    def reset_cropping_values(self) -> None:
        """
        Reset the cropping parameters.
        """

        # Set all values to 0.
        for i in range(3):
            self.cropping_start[i].value = 0
            self.cropping_end[i].value = 0

    #
    # RESAMPLING
    #

    def resampling_setup(self) -> None:
        """
        Sets up the resampling widget by linking the UI to the scene and algorithm.
        """

        # Initialize the resampling target volume.
        self.resampling_target_volume = None

        # Get the resampling target volume combo box.
        self.resampling_target_volume_combo_box = self.panel.findChild(
            ctkComboBox, "ResamplingTargetVolumeComboBox"
        )
        assert self.resampling_target_volume_combo_box

        # Connect the resampling target volume combo box to its "on changed" function.
        self.resampling_target_volume_combo_box.activated.connect(
            self.on_resampling_target_volume_changed
        )

        # Fill the resampling target volume combo box with the available resampling targets.
        self.update_resampling_available_targets()

        # Get the resampling button.
        self.resampling_button = self.panel.findChild(QPushButton, "ResamplingButton")
        assert self.resampling_button

        # Connect the resampling button to the algorithm.
        self.resampling_button.clicked.connect(self.resample)

    def on_resampling_target_volume_changed(self) -> None:
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

    def update_resampling_available_targets(self) -> None:
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
            if not self.selected_volume or self.selected_volume.GetName() != name:
                self.resampling_target_volume_combo_box.addItem(name)
        self.resampling_target_volume_combo_box.setCurrentIndex(-1)

    def resample(self) -> None:
        """
        Retrieves the selected and target volumes and runs the resampling algorithm.
        """

        # Ensure that a volume is selected as well as a resampling target volume.
        if not self.selected_volume:
            self.display_error_message("Please select a volume to resample.")
            return
        if not self.resampling_target_volume:
            self.display_error_message("Please select a resampling target volume.")
            return

        # Call the resampling algorithm.
        resampled_volume = self.sitk_to_vtk(
            self.logic.run(
                self.resourcePath("Scripts/Resampling.py"),
                "resample",
                self.vtk_to_sitk(self.selected_volume),
                self.vtk_to_sitk(self.resampling_target_volume),
            )
        )

        # Transfer the volume metadata.
        self.transfer_volume_metadata(self.selected_volume, resampled_volume)

        # Save the resampled volume.
        self.add_new_volume(resampled_volume, "resampled")

        # Log the resampling.
        print(
            f'"{self.selected_volume.GetName()}" has been resampled to match "{self.resampling_target_volume.GetName()}" as "{resampled_volume.GetName()}".'
        )

    #
    # UTILITIES
    #

    def display_error_message(self, message: str) -> None:
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

    def transfer_volume_metadata(
        self,
        source_volume: vtkMRMLScalarVolumeNode,
        target_volume: vtkMRMLScalarVolumeNode,
    ) -> None:
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

    def add_new_volume(self, volume, name: str) -> None:
        """
        Adds a new volume to the scene.

        Parameters:
            volume: VTK Volume Node to be added.
        """

        # :COMMENT: Ensure that a volume is selected.
        assert self.selected_volume

        # :COMMENT: Generate and assign a unique name to the volume.
        current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        new_name = f"{self.selected_volume.GetName()}_{name}_{current_time}"
        volume.SetName(new_name)

        # :COMMENT: Update the MRML scene.
        mrmlScene.AddNode(volume)

        # :COMMENT: Useful for exporting.
        # :TODO:Iantsa: Fix/remove commented code.
        # sitk.WriteImage(cropped_image, '/Users/iantsaprovost/Desktop/test.nrrd')

    def vtk_to_sitk(self, volume: vtkMRMLScalarVolumeNode) -> sitk.Image:
        """
        Converts a VTK volume into a SimpleITK image.

        Parameters:
            volume: The VTK volume to convert.

        Returns:
            The SimpleITK image.
        """

        volume_image_data = volume.GetImageData()
        np_array = vtk.util.numpy_support.vtk_to_numpy(
            volume_image_data.GetPointData().GetScalars()
        )
        np_array = np.reshape(np_array, volume_image_data.GetDimensions()[::-1])
        image = sitk.GetImageFromArray(np_array)
        return image

    def sitk_to_vtk(self, image: sitk.Image) -> vtkMRMLScalarVolumeNode:
        """
        Converts a SimpleITK image to a VTK volume.

        Parameters:
            image: The SimpleITK image to convert.

        Returns:
            The VTK volume.
        """

        np_array = sitk.GetArrayFromImage(image)
        volume_image_data = vtk.vtkImageData()
        volume_image_data.SetDimensions(np_array.shape[::-1])
        volume_image_data.AllocateScalars(vtk.VTK_FLOAT, 1)
        vtk_array = vtk.util.numpy_support.numpy_to_vtk(np_array.flatten())
        volume_image_data.GetPointData().SetScalars(vtk_array)
        volume = vtkMRMLScalarVolumeNode()
        volume.SetAndObserveImageData(volume_image_data)
        return volume

    def destroy_observers(self) -> None:
        """
        Cleans up the module from any observer.
        """

        # Remove the observers.
        mrmlScene.RemoveObserver(self.observer_tag)
        # :TODO:Iantsa: Fix/remove commented code.
        # mrmlScene.RemoveObserver(self.observerTag2)
