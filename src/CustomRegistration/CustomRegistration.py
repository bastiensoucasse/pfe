"""
The Custom Registration module for Slicer provides the features for 3D images registration, based on the ITK library.
"""

import datetime
import os
import pickle
import unittest
from math import pi

import Elastix
import numpy as np
import scipy
import SimpleITK as sitk
import sitkUtils as su
import vtk
from ctk import ctkCollapsibleButton, ctkCollapsibleGroupBox, ctkComboBox
from Processes import Process, ProcessesLogic
from qt import (
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QElapsedTimer,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QTimer,
    QVBoxLayout,
)
from slicer import (
    app,
    modules,
    mrmlScene,
    util,
    vtkMRMLScalarVolumeDisplayNode,
    vtkMRMLScalarVolumeNode,
    vtkMRMLScene,
)
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleTest,
    ScriptedLoadableModuleWidget,
)

AXIS_MAP = {"x": 0, "y": 1, "z": 2}


class CustomRegistration(ScriptedLoadableModule):
    """
    Main class for the Custom Registration module used to define the module's metadata.
    """

    def __init__(self, parent) -> None:
        ScriptedLoadableModule.__init__(self, parent)

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

    #
    # ROI SELECTION
    #

    def create_mask(
        self, volume: vtkMRMLScalarVolumeNode, threshold: int
    ) -> vtkMRMLScalarVolumeNode:
        """
        Creates a binary mask that selects the voxel of the image that are greater than the threshold.

        Parameters:
            volume: The VTK volume.
            threshold: The threshold value.

        Returns:
            The ROI binary mask as a VTK volume.
        """

        mask = self.sitk_to_vtk(
            sitk.BinaryThreshold(
                self.vtk_to_sitk(volume),
                lowerThreshold=threshold,
                upperThreshold=1000000,
            )
        )
        self.transfer_volume_metadata(volume, mask)
        return mask

    def select_roi(
        self, volume: vtkMRMLScalarVolumeNode, mask: vtkMRMLScalarVolumeNode
    ) -> vtkMRMLScalarVolumeNode:
        """
        Selects as ROI the largest connected component after a threshold.

        Parameters:
            volume: The VTK volume to select the ROI from.
            mask_volume: The ROI binary mask as a SimpleITK image.
            threshold: The threshold value.

        Returns:
            The VTK volume representing the ROI of the input volume.
        """

        # :COMMENT: Apply connected component filter.
        label_map = sitk.ConnectedComponent(self.vtk_to_sitk(mask))

        # :COMMENT: Find largest connected component.
        label_shape_stats = sitk.LabelShapeStatisticsImageFilter()
        label_shape_stats.Execute(label_map)
        max_size = 0
        largest_label = 1
        for label in range(1, label_shape_stats.GetNumberOfLabels() + 1):
            size = label_shape_stats.GetPhysicalSize(label)
            if size > max_size:
                max_size = size
                largest_label = label

        # :COMMENT: Use binary image of largest connected component as ROI.
        binary = sitk.BinaryThreshold(
            label_map, lowerThreshold=largest_label, upperThreshold=largest_label
        )

        roi = self.sitk_to_vtk(binary)
        self.transfer_volume_metadata(volume, roi)

        return roi

    #
    # CROPPING
    #

    def crop(
        self, volume: vtkMRMLScalarVolumeNode, start, end
    ) -> vtkMRMLScalarVolumeNode:
        """
        Crops a volume using the selected algorithm.

        Parameters:
            volume: The VTK volume to be cropped.
            start: The start point of the cropping region.
            end: The end point of the cropping region.

        Returns:
            The cropped VTK volume.
        """

        # :TODO:Iantsa: Delete former implementation when done with report.
        # crop_filter = sitk.ExtractImageFilter()
        # crop_filter.SetIndex(index)
        # crop_filter.SetSize(size)
        # cropped_image = crop_filter.Execute(image)
        # return cropped_image

        # :COMMENT: Ensure that the size is valid.
        crop_size = [end[i] - start[i] for i in range(len(start))]
        if not all(crop_size):
            return None

        # :COMMENT: Handles the autocropping offset too large.
        dims = volume.GetImageData().GetDimensions()
        for axis, i in AXIS_MAP.items():
            if start[i] < 0 or end[i] > dims[i]:
                raise AutocroppingValueError(
                    f"Offset {axis} too large.",
                    axis,
                )

        # :COMMENT: Convert the volume to a SimpleITK image.
        image = self.vtk_to_sitk(volume)

        # :COMMENT: Get the size of the original image.
        size = image.GetSize()

        # :COMMENT: Create a cropping filter.
        crop_filter = sitk.CropImageFilter()

        # :COMMENT: Set the lower and upper cropping indices
        crop_filter.SetLowerBoundaryCropSize(start)
        crop_filter.SetUpperBoundaryCropSize([size[i] - end[i] for i in range(3)])

        # :COMMENT: Crop the image.
        cropped_image = crop_filter.Execute(image)

        # :COMMENT: Convert the cropped SimpleITK image back to a VTK volume.
        cropped_volume = self.sitk_to_vtk(cropped_image)

        # :COMMENT: Transfer the initial volume metadata.
        self.transfer_volume_metadata(volume, cropped_volume)

        return cropped_volume

    def automatic_crop(
        self,
        volume: vtkMRMLScalarVolumeNode,
        roi: vtkMRMLScalarVolumeNode,
        margins,
    ) -> vtkMRMLScalarVolumeNode:
        """
        Crops a volume using the selected algorithm.

        Parameters:
            volume: The VTK volume to be cropped.
            roi: The VTK volume representing the ROI selection of the input volume.
            margins: The margin to add around the ROI selection.

        Returns:
            A tuple representing
                - The cropped VTK volume.
                - The start index of the cropping region.
                - The end index of the cropping region.
        """

        # :COMMENT: Get the pixel data array and dimensions of the image data.
        roi_image_data = roi.GetImageData()
        dims = roi_image_data.GetDimensions()

        # :COMMENT: Convert the pixel data array into a numpy array.
        pixel_data_array = vtk.util.numpy_support.vtk_to_numpy(
            roi_image_data.GetPointData().GetScalars()
        )
        # :TODO:Iantsa: Make sure the order is the right one for better optimization.
        pixel_data_array = pixel_data_array.reshape(dims, order="F")

        # :COMMENT: Ensure the selection is not empty.
        if not np.any(pixel_data_array):
            return None

        # :COMMENT: Get the bounds of the ROI.
        xyz = np.argwhere(pixel_data_array).T
        start = [int(np.min(i)) for i in xyz]
        end = [int(np.max(i)) for i in xyz]

        # :COMMENT Apply the margins along the ROI.
        for i in range(3):
            start[i] -= margins[i]
            end[i] += margins[i]

        return self.crop(volume, start, end), start, end

    #
    # RESAMPLING
    #

    def resample(
        self,
        input_volume: vtkMRMLScalarVolumeNode,
        target_volume: vtkMRMLScalarVolumeNode,
    ) -> vtkMRMLScalarVolumeNode:
        """
        Resamples a volume selected as input to march a volume selected as target (dimensions, pixel size).

        Parameters:
            input_volume: The VTK volume selected as input volume.
            target_volume: The VTK volume selected as target volume.

        Returns:
            The resampled version of the input volume, matchin the target volume.
        """

        # :COMMENT: Use the default transform.
        transform = sitk.Transform()

        # :COMMENT: Use the default interpolation.
        interpolator = sitk.sitkLinear

        # :COMMENT: Resample the input image to match the reference image's size, spacing, and origin.
        resampled_volume = self.sitk_to_vtk(
            sitk.Resample(
                self.vtk_to_sitk(input_volume),
                self.vtk_to_sitk(target_volume),
                transform,
                interpolator,
            )
        )
        self.transfer_volume_metadata(input_volume, resampled_volume)
        resampled_volume.SetSpacing(target_volume.GetSpacing())
        return resampled_volume

    #
    # DIFFERENCE MAP
    #

    def difference_map(self, imageData1, imageData2, name, mode, sigma):
        dims1 = imageData1.GetImageData().GetDimensions()
        dims2 = imageData2.GetImageData().GetDimensions()
        if dims1 != dims2:
            raise ValueError("Images must have the same dimensions")

        volume_image_data1 = imageData1.GetImageData()
        np_array1 = vtk.util.numpy_support.vtk_to_numpy(
            volume_image_data1.GetPointData().GetScalars()
        )
        np_array1 = np.reshape(
            np_array1, volume_image_data1.GetDimensions()[::-1]
        ).astype(
            np.float  # type: ignore
        )

        volume_image_data2 = imageData2.GetImageData()
        np_array2 = vtk.util.numpy_support.vtk_to_numpy(
            volume_image_data2.GetPointData().GetScalars()
        )
        np_array2 = np.reshape(
            np_array2, volume_image_data2.GetDimensions()[::-1]
        ).astype(
            np.float  # type: ignore
        )

        # outputNode = outputNode.astype(np.float)

        if mode == "gradient":
            #:COMMENT: Compute the gradient of each image
            grad_x1, grad_y1, grad_z1 = np.gradient(np_array1)
            grad_x2, grad_y2, grad_z2 = np.gradient(np_array2)

            #:COMMENT:  Compute the difference of the gradient
            outputNode = (
                np.abs(grad_x1 - grad_x2)
                + np.abs(grad_y1 - grad_y2)
                + np.abs(grad_z1 - grad_z2)
            )
        elif mode == "absolute":
            outputNode = np.abs(np_array1 - np_array2).astype(np.float)  # type: ignore
        else:
            raise ValueError("Invalid mode. Mode must be 'gradient' or 'absolute'.")

        if sigma >= 3:
            #:COMMENT: Apply a sigma*sigma*sigma blur with zero padding to outputNode
            kernel = np.ones((sigma, sigma, sigma)) / (
                np.sum(np.ones((sigma, sigma, sigma))) + 1e-8
            )
            outputNode = scipy.ndimage.convolve(
                outputNode, kernel, mode="constant", cval=0.0
            )
        # #:COMMENT:  Normalize the difference
        outputNode = outputNode / np.linalg.norm(outputNode)

        #:COMMENT:  Create a new volume node for the output
        volume_image_data = vtk.vtkImageData()
        volume_image_data.SetDimensions(outputNode.shape[::-1])
        volume_image_data.AllocateScalars(vtk.VTK_FLOAT, 1)
        vtk_array = vtk.util.numpy_support.numpy_to_vtk(outputNode.flatten())
        volume_image_data.GetPointData().SetScalars(vtk_array)
        volume = vtkMRMLScalarVolumeNode()

        #:COMMENT:  Transfer metadata from the first input image data to the output volume node
        self.transfer_volume_metadata(imageData1, volume)
        volume.SetName(name)
        volume.SetAndObserveImageData(volume_image_data)

        #:COMMENT: Compute the mean of the difference
        mean_difference = np.mean(outputNode)

        return volume, mean_difference

    #
    # UTILITIES
    #

    def vtk_to_sitk(self, volume: vtkMRMLScalarVolumeNode) -> sitk.Image:
        """
        Converts a VTK volume into a SimpleITK image.

        Parameters:
            volume: The VTK volume to convert.

        Returns:
            The SimpleITK image.
        """

        volume_image_data = volume.GetImageData()
        np_array = vtk.util.numpy_support.vtk_to_numpy(volume_image_data.GetPointData().GetScalars())  # type: ignore
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
        vtk_array = vtk.util.numpy_support.numpy_to_vtk(np_array.flatten())  # type: ignore
        volume_image_data.GetPointData().SetScalars(vtk_array)
        volume = vtkMRMLScalarVolumeNode()
        volume.SetAndObserveImageData(volume_image_data)
        return volume

    def transfer_volume_metadata(
        self,
        source_volume: vtkMRMLScalarVolumeNode,
        target_volume: vtkMRMLScalarVolumeNode,
    ) -> None:
        """
        Copies the metadata from the source volume to the target volume.

        Parameters:
            source_volume: The VTK volume to copy the metadata from.
            target_volume: The VTK volume to copy the metadata to.
        """

        # :COMMENT: Retrieve the metadata from the source volume.
        spacing = source_volume.GetSpacing()
        origin = source_volume.GetOrigin()
        ijk_to_ras_direction_matrix = vtk.vtkMatrix4x4()
        source_volume.GetIJKToRASDirectionMatrix(ijk_to_ras_direction_matrix)

        # :COMMENT: Apply the metadata to the target volume.
        target_volume.SetSpacing(spacing)
        target_volume.SetOrigin(origin)
        target_volume.SetIJKToRASDirectionMatrix(ijk_to_ras_direction_matrix)


class CustomRegistrationWidget(ScriptedLoadableModuleWidget):
    """
    Widget class for the Custom Registration module used to define the module's panel interface.
    """

    def __init__(self, parent=None) -> None:
        ScriptedLoadableModuleWidget.__init__(self, parent)

    def setup(self) -> None:
        """
        Sets up the widget for the module.
        """

        # :COMMENT: Initialize the widget.
        ScriptedLoadableModuleWidget.setup(self)

        # :COMMENT: Initialize the logic.
        self.logic = CustomRegistrationLogic()
        assert self.logic

        # :COMMENT: Hide the useless widgets.
        util.setApplicationLogoVisible(False)
        util.setModulePanelTitleVisible(False)
        util.setModuleHelpSectionVisible(False)

        # :COMMENT: Load the panel UI.
        self.panel = util.loadUI(self.resourcePath("UI/Panel.ui"))
        assert self.panel

        # :COMMENT: Apply the color palette to the panel.
        self.panel.setPalette(util.mainWindow().palette)

        # :COMMENT: Insert the panel UI into the layout.
        self.layout.addWidget(self.panel)

        # :COMMENT: Collapse all the collapsible buttons.
        collapsible_buttons = self.panel.findChildren(ctkCollapsibleButton)
        for collapsible_widget in collapsible_buttons:
            collapsible_widget.collapsed = True

        # :COMMENT: Initialize the volume list.
        self.volumes = []

        # :COMMENT: Set up the input/target volume architecture.
        self.setup_input_volume()
        self.setup_target_volume()
        self.difference_map = None

        # :COMMENT: Set up the module regions.
        self.setup_view()
        self.setup_pascal_only_mode()
        self.setup_roi_selection()
        self.setup_cropping()
        self.setup_automatic_cropping()
        self.setup_resampling()
        self.setup_registration()
        self.setup_difference_map()
        self.setup_plugin_loading()

        # :COMMENT: Add observer to update combobox when new volume is added to MRML Scene.
        self.scene_observers = [
            mrmlScene.AddObserver(vtkMRMLScene.NodeAddedEvent, self.update),
            mrmlScene.AddObserver(vtkMRMLScene.NodeRemovedEvent, self.update),
            # :TODO:Bastien: Add observer for volume modified event.
        ]

        # :COMMENT: Launch the first volume list update.
        self.update_allowed = True
        self.update()

    def cleanup(self) -> None:
        """
        Cleans up the module from any observer.
        """

        for observer in self.scene_observers:
            mrmlScene.RemoveObserver(observer)

        self.reset()

    def reset(self) -> None:
        """
        Resets all parameters to their default values.
        """

        # Reset the input/target volume architecture.
        self.reset_input_volume()
        self.reset_target_volume()

        # Reset the module regions.
        self.reset_view()
        self.reset_pascal_only_mode()
        self.reset_roi_selection()
        self.reset_cropping()
        self.reset_resampling()
        self.reset_registration()
        self.reset_difference_map()
        self.reset_plugin_loading()

    def update(self, caller=None, event=None) -> None:
        """
        Updates the list of volumes, the view, and the panel, as well as the different module regions accordingly.

        Parameters:
            caller: The widget calling this method.
            event: The event that triggered this method.
        """

        if not self.update_allowed:
            return

        def update_volume_list() -> None:
            """
            Updates the list of volumes by retrieving the volumes that are not ROI masks in the scene.
            """

            self.volumes = [
                volume
                for volume in mrmlScene.GetNodesByClass("vtkMRMLScalarVolumeNode")
                if not volume.GetName().endswith("ROI Mask")
                and not volume.GetName().endswith("ROI Volume")
                and not volume.GetName().endswith("Difference Map")
            ]

        def update_panel(variation: str = "all") -> None:
            """
            Updates the elements from the panel such as the volume combo boxes and their information labels (dimensions and spacing).

            Parameters:
                variation: Either "input", "target", or "all".
            """

            # :COMMENT: Ensure the variation is valid.
            assert variation in ["input", "target", "all"]

            # :COMMENT: Handle the "all" variation.
            if variation == "all":
                update_panel("input")
                update_panel("target")
                return

            # :COMMENT: Define the combo boxes.
            volume_combo_box = self.get_ui(
                ctkComboBox, f"{variation.capitalize()}VolumeComboBox"
            )
            volume_dimensions_value_label = self.get_ui(
                QLabel, f"{variation.capitalize()}VolumeDimensionsValueLabel"
            )
            volume_spacing_value_label = self.get_ui(
                QLabel, f"{variation.capitalize()}VolumeSpacingValueLabel"
            )

            # :COMMENT: Reset the volume combo box.
            volume_combo_box.clear()

            # :COMMENT: Add the available volumes to the combo box.
            for i in range(len(self.volumes)):
                volume = self.volumes[i]
                volume_combo_box.addItem(volume.GetName())

            # :COMMENT: Add the utility options to the combo box.
            volume_combo_box.addItem("Rename current volume…")
            volume_combo_box.addItem("Delete current volume…")

            # :COMMENT: Retrieve the volume and its index.
            if variation == "input":
                volume = self.input_volume
                volume_index = self.input_volume_index
            else:
                volume = self.target_volume
                volume_index = self.target_volume_index

            # :COMMENT: Reset the combo box if volume is None.
            if not volume:
                volume_combo_box.setCurrentIndex(-1)
                volume_dimensions_value_label.setText("…")
                return

            # :COMMENT: Set the combo box position.
            volume_combo_box.setCurrentIndex(volume_index)

            # :COMMENT: Display the dimensions.
            volume_dimensions = volume.GetImageData().GetDimensions()
            volume_dimensions_value_label.setText(
                "{} x {} x {}".format(
                    volume_dimensions[0],
                    volume_dimensions[1],
                    volume_dimensions[2],
                )
            )

            # :COMMENT: Display the spacing.
            volume_spacing = volume.GetSpacing()
            volume_spacing_value_label.setText(
                "{:.1f} x {:.1f} x {:.1f}".format(
                    volume_spacing[0],
                    volume_spacing[1],
                    volume_spacing[2],
                )
            )

        # :COMMENT: Update the in-memory volume list and the associated panel elements.
        update_volume_list()
        update_panel()

        # :COMMENT: Update the module regions.
        self.update_view()
        self.update_pascal_only_mode()
        self.update_roi_selection()
        self.update_cropping("manual")
        self.update_resampling()
        self.update_plugin_loading()

    #
    # INPUT VOLUME
    #

    def setup_input_volume(self) -> None:
        """
        Sets up the input volume architecture by initializing the data and retrieving the UI widgets.
        """

        # :COMMENT: Retreive the input volume combo box.
        self.input_volume_combo_box = self.get_ui(ctkComboBox, "InputVolumeComboBox")

        # :COMMENT: Initialize the input volume.
        self.input_volume = None
        self.input_volume_index = None

        def on_input_volume_combo_box_changed(index: int) -> None:
            """
            Handles change of input volume with options.

            Called when an item in an input volume combobox is selected.

            Parameters:
                index: The input volume index.
            """

            OPTIONS = ["Delete current volume…", "Rename current volume…"]

            # :COMMENT: Retrieve the selection text.
            name = self.input_volume_combo_box.currentText

            # :COMMENT: Handle the different options.
            if name in OPTIONS:
                # :COMMENT: Ensure that there is at least one volume imported.
                if len(self.volumes) < 1:
                    self.update()
                    self.display_error_message("No volumes imported.")
                    return

                # :COMMENT: Ensure that a volume is selected as input.
                if not self.input_volume:
                    self.update()
                    self.display_error_message("Please select a volume first.")
                    return

                if name == "Rename current volume…":
                    self.rename_input_volume()
                    return

                if name == "Delete current volume…":
                    self.delete_input_volume()
                    return

            # :COMMENT: Select the volume at specified index otherwise.
            self.choose_input_volume(index)

        # :COMMENT: Connect the input volume combo box.
        self.input_volume_combo_box.activated.connect(on_input_volume_combo_box_changed)

    def reset_input_volume(self) -> None:
        """
        Resets the input volume to None.
        """

        # :COMMENT: Reset the input volume.
        self.choose_input_volume()

    def choose_input_volume(self, index: int = -1) -> None:
        """
        Selects an input volume.
        """

        if index < 0:
            self.input_volume = None
            self.input_volume_index = None
            assert not self.input_volume
        else:
            self.input_volume = self.volumes[index]
            self.input_volume_index = index
            assert self.input_volume
        self.update()

    def rename_input_volume(self) -> None:
        """
        Loads the renaming feature with a minimal window.
        """

        # :COMMENT: Define the handler.
        def rename_volume_handler(result) -> None:
            """
            Applies the renaming of the input volume.

            Parameters:
                result: The result of the input dialog.
            """

            if result == QDialog.Accepted:
                # :COMMENT: Ensure that a volume is input.
                assert self.input_volume

                # :COMMENT: Retrieve the new name and apply it.
                new_name = self.renaming_input_dialog.textValue()
                self.input_volume.SetName(new_name)

                # :COMMENT: Log the renaming.
                print(
                    f'"{self.renaming_old_name}" has been renamed to "{self.input_volume.GetName()}".'
                )

            # :DIRTY:Bastien: Find a way to add modified node event observer in the setup.
            self.update()

        # :COMMENT: Ensure that a volume is selected as input.
        assert self.input_volume

        # :COMMENT: Save the old name for logging.
        self.renaming_old_name = self.input_volume.GetName()

        # :COMMENT: Open an input dialog for the new name.
        self.renaming_input_dialog = QInputDialog(None)
        self.renaming_input_dialog.setWindowTitle("Rename Volume")
        self.renaming_input_dialog.setLabelText("Enter the new name:")
        self.renaming_input_dialog.setModal(True)
        self.renaming_input_dialog.setTextValue(self.input_volume.GetName())
        self.renaming_input_dialog.finished.connect(rename_volume_handler)
        self.renaming_input_dialog.show()

    def delete_input_volume(self) -> None:
        """
        Deletes the current input volume.
        """

        assert self.input_volume
        volume = self.input_volume
        if self.input_volume_index == self.target_volume_index:
            self.reset_target_volume()
        self.reset_input_volume()
        self.update_allowed = False
        mrmlScene.RemoveNode(volume)
        self.update_allowed = True
        print(f'"{volume.GetName()}" has been deleted.')
        volume = None

    #
    # TARGET VOLUME
    #

    def setup_target_volume(self) -> None:
        """
        Sets up the target volume architecture by initializing the data and retrieving the UI widgets.
        """

        # :COMMENT: Retreive the target volume combo box.
        self.target_volume_combo_box = self.get_ui(ctkComboBox, "TargetVolumeComboBox")

        # :COMMENT: Initialize the target volume.
        self.target_volume = None
        self.target_volume_index = None

        def on_target_volume_combo_box_changed(index: int) -> None:
            """
            Handles change of target volume with options.

            Called when an item in a target volume combobox is selected.

            Parameters:
                index: The target volume index.
            """

            OPTIONS = ["Delete current volume…", "Rename current volume…"]

            # :COMMENT: Retrieve the selection text.
            name = self.target_volume_combo_box.currentText

            # :COMMENT: Handle the different options.
            if name in OPTIONS:
                # :COMMENT: Ensure that there is at least one volume imported.
                if len(self.volumes) < 1:
                    self.update()
                    self.display_error_message("No volumes imported.")
                    return

                # :COMMENT: Ensure that a volume is selected as target.
                if not self.target_volume:
                    self.update()
                    self.display_error_message("Please select a volume first.")
                    return

                if name == "Rename current volume…":
                    self.rename_target_volume()
                    return

                if name == "Delete current volume…":
                    self.delete_target_volume()
                    return

            # :COMMENT: Select the volume at specified index otherwise.
            self.choose_target_volume(index)

        # :COMMENT: Connect the target volume combo box.
        self.target_volume_combo_box.activated.connect(
            on_target_volume_combo_box_changed
        )

    def reset_target_volume(self) -> None:
        """
        Resets the target volume to None.
        """

        # :COMMENT: Clear the view (top visualization).
        self.choose_target_volume()

    def choose_target_volume(self, index: int = -1) -> None:
        """
        Selects an target volume.
        """

        if index < 0:
            self.target_volume = None
            self.target_volume_index = None
            assert not self.target_volume
        else:
            self.target_volume = self.volumes[index]
            self.target_volume_index = index
            assert self.target_volume
        self.update()

    def rename_target_volume(self) -> None:
        """
        Loads the renaming feature with a minimal window.
        """

        # :COMMENT: Define the handler.
        def rename_volume_handler(result) -> None:
            """
            Applies the renaming of the target volume.

            Parameters:
                result: The result of the target dialog.
            """

            if result == QDialog.Accepted:
                # :COMMENT: Ensure that a volume is target.
                assert self.target_volume

                # :COMMENT: Retrieve the new name and apply it.
                new_name = self.renaming_target_dialog.textValue()
                self.target_volume.SetName(new_name)

                # :COMMENT: Log the renaming.
                print(
                    f'"{self.renaming_old_name}" has been renamed to "{self.target_volume.GetName()}".'
                )

            # :DIRTY:Bastien: Find a way to add modified node event observer in the setup.
            self.update()

        # :COMMENT: Ensure that a volume is selected as target.
        assert self.target_volume

        # :COMMENT: Save the old name for logging.
        self.renaming_old_name = self.target_volume.GetName()

        # :COMMENT: Open an target dialog for the new name.
        self.renaming_target_dialog = QInputDialog(None)
        self.renaming_target_dialog.setWindowTitle("Rename Volume")
        self.renaming_target_dialog.setLabelText("Enter the new name:")
        self.renaming_target_dialog.setModal(True)
        self.renaming_target_dialog.setTextValue(self.target_volume.GetName())
        self.renaming_target_dialog.finished.connect(rename_volume_handler)
        self.renaming_target_dialog.show()

    def delete_target_volume(self) -> None:
        """
        Deletes the current target volume.
        """

        assert self.target_volume
        volume = self.target_volume
        if self.target_volume_index == self.input_volume_index:
            self.reset_input_volume()
        self.reset_target_volume()
        self.update_allowed = False
        mrmlScene.RemoveNode(volume)
        self.update_allowed = True
        print(f'"{volume.GetName()}" has been deleted.')
        volume = None

    #
    # VIEW INTERFACE
    #

    def setup_view(self) -> None:
        """
        Sets up the view by retrieving the 2D views.
        """

        VIEWS = ["Red", "Green", "Yellow"]

        # :COMMENT: List of vtkMRMLSliceCompositeNode objects that provide an interface for manipulating the properties of a slice composite node.
        self.slice_composite_nodes = []

        # :COMMENT: List of vtkMRMLSliceLogic objects that provide logic for manipulating a slice view of a volume.
        self.slice_logic = []

        # :COMMENT: Retrieve the objects for each view.
        for i in range(len(VIEWS)):
            self.slice_composite_nodes.append(
                mrmlScene.GetNodeByID(f"vtkMRMLSliceCompositeNode{VIEWS[i]}")
            )
            assert self.slice_composite_nodes[i]

            self.slice_logic.append(
                app.layoutManager().sliceWidget(VIEWS[i]).sliceLogic()
            )
            assert self.slice_logic[i]

        # :COMMENT: Initialize the view.
        self.reset_view()

    def reset_view(self) -> None:
        """
        Resets the view by clearing the 2D views.
        """

        # :COMMENT: Clear each of the slice composite nodes.
        for i in range(len(self.slice_composite_nodes)):
            slice_node = self.slice_logic[i].GetSliceNode()
            slice_node.SetOrientationToAxial()

        # :COMMENT: Update the slice composite nodes.
        self.update_view()

    def update_view(self) -> None:
        """
        Updates the view by setting all the 2D views to their volume, or to blank if there is no volume assigned.
        """

        # :COMMENT: Update the input view.
        if self.input_volume:
            self.update_specific_view(
                0,
                self.input_volume,
                mask=self.input_foreground_volume,
            )
        else:
            self.update_specific_view(0, None)

        # :COMMENT: Update the target view.
        if self.target_volume:
            self.update_specific_view(
                1,
                self.target_volume,
                mask=self.target_foreground_volume,
            )
        else:
            self.update_specific_view(1, None)

        # :COMMENT: Update the difference map view.
        if self.difference_map:
            self.update_specific_view(
                2,
                self.difference_map,
            )
        else:
            self.update_specific_view(2, None)

    def update_specific_view(
        self,
        view_id: int,
        volume: vtkMRMLScalarVolumeNode,
        mask: vtkMRMLScalarVolumeNode = None,
    ) -> None:
        """
        Updates a given 2D view with the given volume and mask if needed.

        Parameters:
            view_id: The 2D view ID.
            volume: The given volume.
            mask: The given mask.
        """

        # :COMMENT: Set to blank if no volume.
        if not volume:
            self.slice_composite_nodes[view_id].SetBackgroundVolumeID("")
            self.slice_composite_nodes[view_id].SetForegroundVolumeID("")
            return

        # :COMMENT: Display the selected volume.
        self.slice_composite_nodes[view_id].SetBackgroundVolumeID(volume.GetID())
        if mask:
            self.slice_composite_nodes[view_id].SetForegroundVolumeID(mask.GetID())
            self.slice_composite_nodes[view_id].SetForegroundOpacity(0.5)

        # :COMMENT: Scale the view to the volumes.
        self.slice_logic[view_id].FitSliceToAll()

    #
    # PASCAL-ONLY MODE
    #

    def setup_pascal_only_mode(self) -> None:
        """
        …
        """

        # :COMMENT: Retrieve the Pascal-only mode checkbox.
        self.pascal_only_mode_checkbox = self.get_ui(
            QCheckBox, "PascalOnlyModeCheckBox"
        )
        self.pascal_only_mode_checkbox.clicked.connect(self.update_pascal_only_mode)

        # :COMMENT: Retrieve the 3D widget.
        self.three_dimension_widget = app.layoutManager().threeDWidget(0)
        assert self.three_dimension_widget

        # :COMMENT: Retrieve the rendering logic.
        self.rendering_logic = modules.volumerendering.logic()
        assert self.rendering_logic

        # :COMMENT: Initialize the rendering dispplay node.
        self.rendering_display_node = None

        # :COMMENT: Initialize the Pascal-only mode.
        self.reset_pascal_only_mode()

    def reset_pascal_only_mode(self) -> None:
        """
        …
        """

        # :COMMENT: Uncheck the Pascal-only mode checkbox and update the Pascal-only mode.
        self.pascal_only_mode_checkbox.setChecked(False)
        self.update_pascal_only_mode()

    def update_pascal_only_mode(self) -> None:
        """
        …
        """

        # :COMMENT: Remove any volume from the 3D view.
        if self.rendering_display_node:
            self.update_allowed = False
            mrmlScene.RemoveNode(self.rendering_display_node)
            self.rendering_display_node = None
            self.update_allowed = True

        # :COMMENT: If Pascal-only mode is disabled, hide the 3D widget and return.
        if not self.pascal_only_mode_checkbox.isChecked():
            self.three_dimension_widget.setVisible(False)
            return

        # :COMMENT: Display the 3D widget.
        self.three_dimension_widget.setVisible(True)

        # :COMMENT: Display the input volume into the 3D view.
        if self.input_volume:
            self.update_allowed = False
            self.rendering_display_node = (
                self.rendering_logic.CreateVolumeRenderingDisplayNode()
            )
            self.rendering_display_node.UnRegister(self.rendering_logic)
            self.rendering_display_node.SetName(
                "CustomRegistrationRenderingDisplayNode"
            )
            mrmlScene.AddNode(self.rendering_display_node)
            self.input_volume.AddAndObserveDisplayNodeID(
                self.rendering_display_node.GetID()
            )
            self.rendering_logic.UpdateDisplayNodeFromVolumeNode(
                self.rendering_display_node, self.input_volume
            )
            self.update_allowed = True

    #
    # ROI SELECTION
    #

    def setup_roi_selection(self) -> None:
        """
        …
        """

        # :COMMENT: Create a color table node that assigns the ROI mask to red.
        self.red_color_table_node = mrmlScene.AddNewNodeByClass("vtkMRMLColorTableNode")
        self.red_color_table_node.SetTypeToUser()
        self.red_color_table_node.SetNumberOfColors(2)
        self.red_color_table_node.SetColor(0, 0.0, 0.0, 0.0, 0.0)
        self.red_color_table_node.SetColor(1, 1.0, 0.0, 0.0, 1.0)

        # :COMMENT: Create a color table node that assigns the ROI volume to green.
        self.green_color_table_node = mrmlScene.AddNewNodeByClass(
            "vtkMRMLColorTableNode"
        )
        self.green_color_table_node.SetTypeToUser()
        self.green_color_table_node.SetNumberOfColors(2)
        self.green_color_table_node.SetColor(0, 0.0, 0.0, 0.0, 0.0)
        self.green_color_table_node.SetColor(1, 0.0, 1.0, 0.0, 1.0)

        # :COMMENT: Get the collapsible button.
        self.roi_selection_collapsible_button = self.get_ui(
            ctkCollapsibleButton, "ROISelectionCollapsibleWidget"
        )
        self.roi_selection_collapsible_button.clicked.connect(
            lambda: self.update_roi_selection()
        )

        # :COMMENT: Connect the input ROI selection threshold slider.
        self.input_mask_selection_threshold_slider = self.get_ui(
            QSlider, "InputROISelectionThresholdSlider"
        )
        self.input_mask_selection_threshold_slider.valueChanged.connect(
            lambda: self.update_roi_selection("input")
        )
        self.input_mask_selection_threshold_value_label = self.get_ui(
            QLabel, "InputROISelectionThresholdValueLabel"
        )

        # :COMMENT: Connect the target ROI selection threshold slider.
        self.target_mask_selection_threshold_slider = self.get_ui(
            QSlider, "TargetROISelectionThresholdSlider"
        )
        self.target_mask_selection_threshold_slider.valueChanged.connect(
            lambda: self.update_roi_selection("target")
        )
        self.target_mask_selection_threshold_value_label = self.get_ui(
            QLabel, "TargetROISelectionThresholdValueLabel"
        )

        # :COMMENT: Get the ROI selection button.
        self.roi_selection_button = self.get_ui(QPushButton, "ROISelectionButton")
        self.roi_selection_button.clicked.connect(self.select_roi)

        # :COMMENT: Create an empty dictionary in which each ROI mask will be store for a specific volume.
        self.volume_roi_map = {}

        # :COMMENT: Initialize the ROI selection.
        self.input_foreground_volume = None
        self.target_foreground_volume = None
        self.roi_selection_preview_allowed = True
        self.reset_roi_selection()

    def reset_roi_selection(self) -> None:
        """
        …
        """

        # :COMMENT: Reset the ROI selection threshold values.
        self.roi_selection_preview_allowed = False
        self.input_mask_selection_threshold_slider.setValue(0)
        self.target_mask_selection_threshold_slider.setValue(0)
        self.cropping_preview_allowed = True
        self.roi_selection_collapsible_button.collapsed = True

        # :COMMENT: Update the ROI selection.
        self.update_roi_selection()

    def update_roi_selection(self, variation: str = "all") -> None:
        """
        …
        """

        assert variation in ["input", "target", "all"]

        if variation == "all":
            self.update_roi_selection("input")
            self.update_roi_selection("target")

        # :COMMENT: Initialize the threshold value.
        threshold = 0

        # :COMMENT: Update the input threshold value.
        if variation == "input":
            self.slice_composite_nodes[0].SetForegroundVolumeID("")

            if self.input_volume:
                range = self.input_volume.GetImageData().GetScalarRange()
            else:
                range = (0, 255)
            self.roi_selection_preview_allowed = False
            self.input_mask_selection_threshold_slider.setMinimum(range[0])
            self.input_mask_selection_threshold_slider.setMaximum(range[1])
            self.roi_selection_preview_allowed = True

            threshold = self.input_mask_selection_threshold_slider.value
            self.input_mask_selection_threshold_value_label.setText(int(threshold))

        # :COMMENT: Update the target threshold value.
        if variation == "target":
            self.slice_composite_nodes[1].SetForegroundVolumeID("")

            if self.target_volume:
                range = self.target_volume.GetImageData().GetScalarRange()
            else:
                range = (0, 255)
            self.roi_selection_preview_allowed = False
            self.target_mask_selection_threshold_slider.setMinimum(range[0])
            self.target_mask_selection_threshold_slider.setMaximum(range[1])
            self.roi_selection_preview_allowed = True

            threshold = self.target_mask_selection_threshold_slider.value
            self.target_mask_selection_threshold_value_label.setText(int(threshold))

        if self.roi_selection_preview_allowed:
            if self.roi_selection_collapsible_button.isChecked():
                self.preview_roi_selection(variation)
            else:
                self.display_roi(variation)

    def preview_roi_selection(self, variation: str = "all") -> None:
        """
        …
        """

        assert variation in ["input", "target", "all"]

        if variation == "all":
            self.preview_roi_selection("input")
            self.preview_roi_selection("target")

        # :COMMENT: Set the update rule to blocked.
        self.update_allowed = False

        if variation == "input":
            # :COMMENT: Remove the previous input mask if needed.
            if self.input_foreground_volume:
                mrmlScene.RemoveNode(self.input_foreground_volume)
                self.input_foreground_volume = None

            # :COMMENT: Ensure the input volume is not None.
            if not self.input_volume:
                self.update_allowed = True
                return

            # :COMMENT: Call the ROI selection algorithm.
            threshold = self.input_mask_selection_threshold_slider.value
            self.input_foreground_volume = self.logic.create_mask(
                self.input_volume, threshold
            )

            # :COMMENT: Get or create the mask display node.
            mask_display_node = self.input_foreground_volume.GetDisplayNode()
            if not mask_display_node:
                mask_display_node = vtkMRMLScalarVolumeDisplayNode()
                mrmlScene.AddNode(mask_display_node)
                self.input_foreground_volume.SetAndObserveDisplayNodeID(
                    mask_display_node.GetID()
                )

            # :COMMENT: Assign the color map to the mask display.
            mask_display_node.SetAndObserveColorNodeID(
                self.red_color_table_node.GetID()
            )

            # :COMMENT: Add the mask to the scene to visualize it.
            self.input_foreground_volume.SetName(
                f"{self.input_volume.GetName()} ROI Mask"
            )
            mrmlScene.AddNode(self.input_foreground_volume)

            # :COMMENT: Set the update rule to allowed.
            self.update_specific_view(
                0, self.input_volume, self.input_foreground_volume
            )

        if variation == "target":
            # :COMMENT: Remove the previous target mask if needed.
            if self.target_foreground_volume:
                mrmlScene.RemoveNode(self.target_foreground_volume)
                self.target_foreground_volume = None

            # :COMMENT: Ensure the input volume is not None.
            if not self.target_volume:
                self.update_allowed = True
                return

            # :COMMENT: Retrieve the threshold value.
            threshold = self.target_mask_selection_threshold_slider.value

            # :COMMENT: Call the ROI selection algorithm.
            self.target_foreground_volume = self.logic.create_mask(
                self.target_volume, threshold
            )

            # :COMMENT: Get or create the mask display node.
            mask_display_node = self.target_foreground_volume.GetDisplayNode()
            if not mask_display_node:
                mask_display_node = vtkMRMLScalarVolumeDisplayNode()
                mrmlScene.AddNode(mask_display_node)
                self.target_foreground_volume.SetAndObserveDisplayNodeID(
                    mask_display_node.GetID()
                )

            # :COMMENT: Assign the color map to the mask display.
            mask_display_node.SetAndObserveColorNodeID(
                self.red_color_table_node.GetID()
            )

            # :COMMENT: Add the mask to the scene to visualize it.
            self.target_foreground_volume.SetName(
                f"{self.target_volume.GetName()} ROI Mask"
            )
            mrmlScene.AddNode(self.target_foreground_volume)

            # :COMMENT: Set the update rule to allowed.
            self.update_specific_view(
                1, self.target_volume, self.target_foreground_volume
            )

        # :COMMENT: Set the update rule to allowed.
        self.update_allowed = True

    def display_roi(self, variation: str = "all") -> None:
        """
        Displays the ROI volume in green.
        """

        assert variation in ["input", "target", "all"]

        if variation == "all":
            self.display_roi("input")
            self.display_roi("target")

        # :COMMENT: Set the update rule to blocked.
        self.update_allowed = False

        if variation == "input":
            # :COMMENT: Remove the previous input ROI if needed.
            if self.input_foreground_volume:
                mrmlScene.RemoveNode(self.input_foreground_volume)
                self.input_foreground_volume = None

            # :COMMENT: Ensure the target volume is not None.
            if not self.input_volume:
                self.update_allowed = True
                return

            # :COMMENT: Ensure the volume has ROI.
            if self.input_volume.GetName() not in self.volume_roi_map:
                self.update_allowed = True
                return

            # :COMMENT: Retrieve the ROI volume.
            self.input_foreground_volume = self.volume_roi_map[
                self.input_volume.GetName()
            ]

            # :COMMENT: Get or create the ROI display node.
            roi_display_node = self.input_foreground_volume.GetDisplayNode()
            if not roi_display_node:
                roi_display_node = vtkMRMLScalarVolumeDisplayNode()
                mrmlScene.AddNode(roi_display_node)
                self.input_foreground_volume.SetAndObserveDisplayNodeID(
                    roi_display_node.GetID()
                )

            # :COMMENT: Assign the color map to the ROI display.
            roi_display_node.SetAndObserveColorNodeID(
                self.green_color_table_node.GetID()
            )

            # :COMMENT: Add the ROI to the scene to visualize it.
            self.input_foreground_volume.SetName(
                f"{self.input_volume.GetName()} ROI Volume"
            )
            mrmlScene.AddNode(self.input_foreground_volume)

            # :COMMENT: Set the update rule to allowed.
            self.update_specific_view(
                0, self.input_volume, self.input_foreground_volume
            )

        if variation == "target":
            # :COMMENT: Remove the previous target ROI if needed.
            if self.target_foreground_volume:
                mrmlScene.RemoveNode(self.target_foreground_volume)
                self.target_foreground_volume = None

            # :COMMENT: Ensure the target volume is not None.
            if not self.target_volume:
                self.update_allowed = True
                return

            # :COMMENT: Ensure the volume has ROI.
            if self.target_volume.GetName() not in self.volume_roi_map:
                self.update_allowed = True
                return

            # :COMMENT: Retrieve the ROI volume.
            self.target_foreground_volume = self.volume_roi_map[
                self.target_volume.GetName()
            ]

            # :COMMENT: Get or create the ROI display node.
            roi_display_node = self.target_foreground_volume.GetDisplayNode()
            if not roi_display_node:
                roi_display_node = vtkMRMLScalarVolumeDisplayNode()
                mrmlScene.AddNode(roi_display_node)
                self.target_foreground_volume.SetAndObserveDisplayNodeID(
                    roi_display_node.GetID()
                )

            # :COMMENT: Assign the color map to the ROI display.
            roi_display_node.SetAndObserveColorNodeID(
                self.green_color_table_node.GetID()
            )

            # :COMMENT: Add the ROI to the scene to visualize it.
            self.target_foreground_volume.SetName(
                f"{self.target_volume.GetName()} ROI Volume"
            )
            mrmlScene.AddNode(self.target_foreground_volume)

            # :COMMENT: Set the update rule to allowed.
            self.update_specific_view(
                1, self.target_volume, self.target_foreground_volume
            )

        # :COMMENT: Set the update rule to allowed.
        self.update_allowed = True

    def select_roi(self) -> None:
        """
        …
        """

        # :COMMENT: Ensure that a volume is selected.
        if not self.input_volume and not self.target_volume:
            self.display_error_message(
                "Please select a volume (input, target, or both) to select a ROI from."
            )
            return

        if self.input_volume:
            # :COMMENT: Retrieve the name of the selected input volume.
            name = self.input_volume.GetName()

            # :COMMENT: Compute missing mask if needed.
            if not self.input_foreground_volume:
                self.input_foreground_volume = self.logic.create_mask(
                    self.input_volume, self.input_mask_selection_threshold_slider.value
                )

            # :COMMENT: Compute and save the ROI using the mask.
            roi = self.logic.select_roi(self.input_volume, self.input_foreground_volume)
            self.volume_roi_map[name] = roi

            # :COMMENT: Log the ROI selection.
            print(
                f'ROI has been selected with a threshold value of {self.input_mask_selection_threshold_slider.value} in "{name}".'
            )

        if self.target_volume:
            # :COMMENT: Retrieve the name of the selected target volume.
            name = self.target_volume.GetName()

            # :COMMENT: Compute missing mask if needed.
            if not self.target_foreground_volume:
                self.target_foreground_volume = self.logic.create_mask(
                    self.target_volume,
                    self.target_mask_selection_threshold_slider.value,
                )

            # :COMMENT: Compute and save the ROI using the mask.
            roi = self.logic.select_roi(
                self.target_volume, self.target_foreground_volume
            )
            self.volume_roi_map[name] = roi

            # :COMMENT: Log the ROI selection.
            print(
                f'ROI has been selected with a threshold value of {self.target_mask_selection_threshold_slider.value} in "{name}".'
            )

        # :COMMENT: Reset the ROI selection data.
        self.reset_roi_selection()

    #
    # CROPPING
    #

    def setup_cropping(self) -> None:
        """
        …
        """

        # :COMMENT: Get the collapsible button.
        self.cropping_collapsible_button = self.get_ui(
            ctkCollapsibleButton, "CroppingCollapsibleWidget"
        )
        self.cropping_collapsible_button.clicked.connect(
            lambda: self.update_cropping("manual")
        )
        # :TODO:Iantsa: Change "manual" right above to trigger the correct switch (manual vs. automatic).

        # :COMMENT: Get the crop button widget.
        self.cropping_button = self.get_ui(QPushButton, "CroppingButton")
        self.cropping_button.clicked.connect(self.crop)

        # :COMMENT: Get the coordinates spinbox widgets.
        self.cropping_start = []
        self.cropping_end = []
        for axis, i in AXIS_MAP.items():
            self.cropping_start.append(self.get_ui(QSpinBox, "s" + axis))
            self.cropping_end.append(self.get_ui(QSpinBox, "e" + axis))

            # :COMMENT: Connect the spinbox widgets to their "on changed" function that displays the cropping preview.
            self.cropping_start[i].valueChanged.connect(
                lambda: self.update_cropping("manual")
            )
            self.cropping_end[i].valueChanged.connect(
                lambda: self.update_cropping("manual")
            )

        # :COMMENT: Initialize the cropping preview.
        self.cropped_volume = None
        self.cropping_box = None
        self.cropping_preview_allowed = True
        self.reset_cropping()

    def reset_cropping(self) -> None:
        """
        …
        """

        # :COMMENT: Reset the cropping values.
        self.cropping_preview_allowed = False
        for i in range(3):
            self.cropping_start[i].value = 0
            self.cropping_end[i].value = 0
        self.cropping_preview_allowed = True
        self.cropping_collapsible_button.collapsed = True

        # :COMMENT: Update the cropping.
        self.update_cropping("manual")

    def update_cropping(self, variation: str) -> None:
        """
        …
        """

        assert variation in ["manual", "automatic"]

        # :COMMENT: Hide the cropping box by default.
        if self.cropping_box and self.cropping_box.GetDisplayNode():
            self.cropping_box.GetDisplayNode().SetVisibility(False)

        if not self.input_volume:
            return

        # :COMMENT: Reset the cropping value ranges.
        input_volume_image_data = self.input_volume.GetImageData()
        input_volume_dimensions = input_volume_image_data.GetDimensions()
        self.cropping_preview_allowed = False
        for i in range(3):
            self.cropping_start[i].setMaximum(input_volume_dimensions[i])
            self.cropping_end[i].setMaximum(input_volume_dimensions[i])
        self.cropping_preview_allowed = True

        if (
            variation == "manual"
            and self.cropping_collapsible_button.isChecked()
            and self.cropping_preview_allowed
        ):
            self.preview_cropping()

        if (
            variation == "automatic"
            and self.cropping_collapsible_button.isChecked()
            and self.cropping_preview_allowed
        ):
            self.preview_automatic_cropping()

    def preview_cropping(self) -> None:
        """
        …
        """

        # :DIRTY/TRICKY:Iantsa: Volume cropped each time a parameter is changed by user, even if the volume is not cropped in the end.

        # :COMMENT: Set the update rule to blocked.
        self.update_allowed = False

        # :COMMENT: Remove the previous cropping box if needed.
        if self.cropping_box:
            mrmlScene.RemoveNode(self.cropping_box)
            self.cropping_box = None

        # :COMMENT: Ensure that the input volume is not None.
        if not self.input_volume:
            self.cropped_volume = None
            self.cropping_box = None
            self.update_allowed = True
            return

        # :COMMENT: Retrieve coordinates input.
        start_val = []
        end_val = []
        for i in range(3):
            start_val.append(self.cropping_start[i].value)
            end_val.append(self.cropping_end[i].value)

        # :COMMENT: Check that coordinates are valid.
        if any(end_val[i] <= start_val[i] for i in range(3)):
            self.cropped_volume = None
            self.cropping_box = None
            self.update_allowed = True
            return

        # :COMMENT: Save the temporary cropped volume.
        self.cropped_volume = self.logic.crop(self.input_volume, start_val, end_val)

        # :COMMENT: Create a new cropping box.
        self.cropping_box = mrmlScene.AddNewNodeByClass(
            "vtkMRMLMarkupsROINode", "Cropping Preview"
        )
        self.cropping_box.SetLocked(True)

        # :COMMENT: Display cropping box only in red view.
        self.cropping_box.GetDisplayNode().SetViewNodeIDs(["vtkMRMLSliceNodeRed"])

        # :COMMENT: Get the bounds of the volume.
        bounds = [0, 0, 0, 0, 0, 0]
        self.cropped_volume.GetBounds(bounds)

        # :COMMENT: Get the size of the crop region.
        size = [end_val[i] - start_val[i] for i in range(3)]

        # :COMMENT: Calculate the center and radius of the volume.
        center = [(bounds[i] + bounds[i + 1]) / 2 for i in range(0, 5, 2)]
        radius = [size[i] / 2 for i in range(3)]

        # :COMMENT: Transform the center and radius according to the volume's orientation and spacing.
        matrix = vtk.vtkMatrix4x4()
        self.input_volume.GetIJKToRASDirectionMatrix(matrix)
        transform_matrix = np.array(
            [[matrix.GetElement(i, j) for j in range(3)] for i in range(3)]
        )
        transformed_center = np.array(center) + np.matmul(transform_matrix, start_val)
        transformed_radius = np.matmul(
            transform_matrix,
            np.array(self.cropped_volume.GetSpacing()) * np.array(radius),
        )

        # :COMMENT: Set the center and radius of the cropping box to the transformed center and radius.
        self.cropping_box.SetXYZ(transformed_center)
        self.cropping_box.SetRadiusXYZ(transformed_radius)

        # :COMMENT: Show the automatic cropping preview.
        self.cropping_box.GetDisplayNode().SetVisibility(True)

        # :COMMENT: Set the update rule to allowed.
        self.update_allowed = True

        # :END_DIRTY/TRICKY:

    def crop(self) -> None:
        """
        …
        """

        # :COMMENT: Ensure that a volume is selected.
        if not self.input_volume:
            self.display_error_message("Please select a volume to crop.")
            return

        # :BUG:Iantsa: Not handled yet (can be non existent if crop button clicked without changing the default parameters)
        if not self.cropped_volume:  # and not self.cropping_box:
            return

        # :COMMENT: Retrieve coordinates input.
        # :DIRTY:Iantsa: This is done twice: in preview and in crop.
        start_val = []
        end_val = []
        for i in range(3):
            start_val.append(self.cropping_start[i].value)
            end_val.append(self.cropping_end[i].value)

        # :COMMENT: Check that coordinates are valid.
        if any(end_val[i] <= start_val[i] for i in range(3)):
            self.display_error_message("End values must be greater than start values.")
            return

        # :COMMENT: Delete the cropping box (should exist if cropped_volume also exists)
        assert self.cropping_box
        self.update_allowed = False
        mrmlScene.RemoveNode(self.cropping_box)
        self.cropping_box = None
        self.update_allowed = True

        # :COMMENT: Add the cropped volume to the scene.
        self.add_new_volume(self.cropped_volume, "cropped")

        # :COMMENT: Log the cropping.
        new_size = self.cropped_volume.GetImageData().GetDimensions()
        print(
            f'"{self.input_volume.GetName()}" has been cropped to size ({new_size[0]}x{new_size[1]}x{new_size[2]}) as "{self.cropped_volume.GetName()}".'
        )

        # :COMMENT: Reset the cropping.
        self.reset_cropping()

        # :COMMENT: Select the cropped volume.
        self.choose_input_volume(len(self.volumes) - 1)

        # :COMMENT: Delete the temporary cropped volume.
        self.cropped_volume = None
        self.cropping_box = None

    #
    # AUTOMATIC CROPPING
    #

    def setup_automatic_cropping(self) -> None:
        """
        Sets up the automatic cropping widget by linking the UI to the scene and algorithm.
        """

        # :COMMENT: Collapsible button and cropping data handled by manual cropping setup.

        # :COMMENT: Get the automatic crop button widget.
        self.automatic_cropping_button = self.get_ui(
            QPushButton, "AutomaticCroppingButton"
        )
        self.automatic_cropping_button.clicked.connect(self.automatic_crop)

        # :COMMENT: Get the coordinates spinbox widgets.
        self.automatic_cropping_margins = []
        for axis, i in AXIS_MAP.items():
            self.automatic_cropping_margins.append(
                self.get_ui(QSpinBox, axis + "Margin")
            )
            # :DIRTY: Magic number.
            self.automatic_cropping_margins[i].setMaximum(2000)

            # :COMMENT: Connect the spinbox widgets to their "on changed" function that displays the automatic cropping preview.
            self.automatic_cropping_margins[i].valueChanged.connect(
                lambda: self.update_cropping("automatic")
            )

        self.reset_automatic_cropping()

        # :TODO:Iantsa: Initialize the cropping volume/box with 0 margin instead of None.

    def reset_automatic_cropping(self) -> None:
        """
        Reset the automatic cropping parameters.
        """

        # :COMMENT: Reset the cropping values.
        self.cropping_preview_allowed = False
        for i in range(3):
            self.automatic_cropping_margins[i].value = 0
        self.cropping_preview_allowed = True
        self.cropping_collapsible_button.collapsed = True

        # :COMMENT: Update the cropping.
        self.update_cropping("automatic")

    def preview_automatic_cropping(self) -> None:
        """
        Generates a bounding box to preview the automatic cropping.

        Called when a margin parameter spin box value is changed.
        """

        # :DIRTY/TRICKY:Iantsa: Volume cropped each time a parameter is changed by user, even if the volume is not cropped in the end.

        # :COMMENT: Set the update rule to blocked.
        self.update_allowed = False

        # :COMMENT: Remove the previous cropping box if needed.
        if self.cropping_box:
            mrmlScene.RemoveNode(self.cropping_box)
            self.cropping_box = None

        # :COMMENT: Ensure that a volume is selected.
        if not self.input_volume:
            self.cropped_volume = None
            self.cropping_box = None
            self.update_allowed = True
            return

        # :COMMENT: Retrieve the name of the selected input volume.
        name = self.input_volume.GetName()

        # :COMMENT: Ensure that a ROI has been selected.
        if name not in self.volume_roi_map:
            self.cropped_volume = None
            self.cropping_box = None
            self.display_error_message("Please select a ROI.")
            return
        roi = self.volume_roi_map[name]  # type: ignore

        # :COMMENT: Retrieve margins input.
        margins = []
        for i in range(3):
            margins.append(self.automatic_cropping_margins[i].value)

        # :COMMENT: Save the temporary cropped volume.
        try:
            self.cropped_volume, start_val, end_val = self.logic.automatic_crop(
                self.input_volume, roi, margins
            )
        except AutocroppingValueError as e:
            self.cropping_preview_allowed = False
            spin_box = self.automatic_cropping_margins[AXIS_MAP[e.axis]]
            spin_box.setValue(0)
            self.cropping_preview_allowed = True
            self.cropped_volume = None
            self.cropping_box = None
            self.display_error_message(str(e))
            return

        if not self.cropped_volume:
            self.cropping_box = None
            self.display_error_message(
                "Empty ROI selection. Please select a ROI again."
            )
            return

        # :COMMENT: Create a new cropping box.
        self.cropping_box = mrmlScene.AddNewNodeByClass(
            "vtkMRMLMarkupsROINode", "Automatic Cropping Preview"
        )
        self.cropping_box.SetLocked(True)

        # :COMMENT: Display cropping box only in red view.
        self.cropping_box.GetDisplayNode().SetViewNodeIDs(["vtkMRMLSliceNodeRed"])

        # :COMMENT: Get the bounds of the volume.
        bounds = [0, 0, 0, 0, 0, 0]
        self.cropped_volume.GetBounds(bounds)

        # :COMMENT: Compute the size of ROI.
        size = [end_val[i] - start_val[i] for i in range(3)]

        # :COMMENT: Calculate the center and radius of the ROI.
        center = [(bounds[i] + bounds[i + 1]) / 2 for i in range(0, 5, 2)]
        radius = [size[i] / 2 for i in range(3)]

        # :COMMENT: Transform the center and radius according to the volume's orientation and spacing.
        matrix = vtk.vtkMatrix4x4()
        self.input_volume.GetIJKToRASDirectionMatrix(matrix)
        transform_matrix = np.array(
            [[matrix.GetElement(i, j) for j in range(3)] for i in range(3)]
        )
        transformed_center = np.array(center) + np.matmul(transform_matrix, start_val)
        transformed_radius = np.matmul(
            transform_matrix,
            np.array(self.input_volume.GetSpacing()) * np.array(radius),
        )

        # :COMMENT: Set the center and radius of the cropping box to the transformed center and radius.
        self.cropping_box.SetXYZ(transformed_center)
        self.cropping_box.SetRadiusXYZ(transformed_radius)

        # :COMMENT: Show the automatic cropping preview.
        self.cropping_box.GetDisplayNode().SetVisibility(True)

        # :COMMENT: Set the update rule to allowed.
        self.update_allowed = True

        # :END_DIRTY/TRICKY:

    def automatic_crop(self) -> None:
        """
        Crops a volume using the selected ROI and margins.
        """

        # :COMMENT: Ensure that a volume is selected.
        if not self.input_volume:
            self.display_error_message("Please select a volume to automatically crop.")
            return

        # :BUG:Iantsa: Not handled yet (can be non existent if crop button clicked without changing the default parameters)
        if not self.cropped_volume:  # and not self.cropping_box:
            self.display_error_message(
                "Margin out of bound. Please enter valid parameters."
            )
            return

        # :BUG:Iantsa: Cannot ensure that coordinates are valid.

        # :COMMENT: Delete the cropping box (should exist if cropped_volume also exists)
        assert self.cropping_box
        self.update_allowed = False
        mrmlScene.RemoveNode(self.cropping_box)
        self.cropping_box = None
        self.update_allowed = True

        volume = self.cropped_volume

        # :COMMENT: Add the VTK volume to the scene.
        self.add_new_volume(self.cropped_volume, "cropped")

        # :COMMENT: Log the automatic cropping.
        new_size = volume.GetImageData().GetDimensions()
        print(
            f'"{self.input_volume.GetName()}" has been cropped to size ({new_size[0]}x{new_size[1]}x{new_size[2]}) as "{volume.GetName()}".'
        )

        # :COMMENT: Reset the cropping.
        self.reset_automatic_cropping()

        # :COMMENT: Select the cropped volume.
        self.choose_input_volume(len(self.volumes) - 1)

        # :COMMENT: Delete the temporary cropped volume.
        self.cropped_volume = None
        self.cropping_box = None

    #
    # RESAMPLING
    #

    def setup_resampling(self) -> None:
        """
        …
        """

        # :COMMENT: Get the resampling button.
        self.resampling_button = self.get_ui(QPushButton, "ResamplingButton")

        # :COMMENT: Connect the resampling button to the algorithm.
        self.resampling_button.clicked.connect(self.resample)

        # :COMMENT: Initialize the resampling.
        self.reset_resampling()

    def reset_resampling(self) -> None:
        """
        …
        """

        self.get_ui(
            ctkCollapsibleButton, "ResamplingCollapsibleWidget"
        ).collapsed = True

        # :COMMENT: Update the resampling.
        self.update_resampling()

    def update_resampling(self) -> None:
        """
        …
        """

        # :COMMENT: Nothing to update.

    def resample(self) -> None:
        """
        …
        """

        # :COMMENT: Ensure that the input and target volumes are selected.
        if not self.input_volume:
            self.display_error_message("Please select a volume to resample.")
            return
        if not self.target_volume:
            self.display_error_message("Please choose a target volume.")
            return

        # :COMMENT: Call the resampling algorithm.
        resampled_volume = self.logic.resample(self.input_volume, self.target_volume)

        # :COMMENT: Save the resampled volume.
        self.add_new_volume(resampled_volume, "resampled")

        # :COMMENT: Log the resampling.
        print(
            f'"{self.input_volume.GetName()}" has been resampled to match "{self.target_volume.GetName()}" as "{resampled_volume.GetName()}".'
        )

        # :COMMENT: Reset the resampling.
        self.reset_resampling()

        # :COMMENT: Select the resampled volume.
        self.choose_input_volume(len(self.volumes) - 1)

    #
    # REGISTRATION
    #

    def setup_registration(self) -> None:
        """
        Set up registration UI elements and connect to code and functions.
        """

        # :COMMENT: Link settings UI and code
        self.metrics_combo_box = self.get_ui(ctkComboBox, "ComboMetrics")
        self.interpolator_combo_box = self.get_ui(ctkComboBox, "comboBoxInterpolator")
        self.optimizers_combo_box = self.get_ui(ctkComboBox, "ComboOptimizers")
        self.metric_related_value_spin_box = self.get_ui(
            QDoubleSpinBox, "spinBoxMetricValue"
        )
        self.metric_related_value_label = self.get_ui(QLabel, "RegistrationMetricLabel")
        self.sampling_strat_combo_box = self.get_ui(
            ctkComboBox, "comboBoxSamplingStrat"
        )
        self.sampling_perc_spin_box = self.get_ui(
            QDoubleSpinBox, "doubleSpinBoxSamplingPerc"
        )

        # :COMMENT: registration types
        self.sitk_combo_box = self.get_ui(ctkComboBox, "ComboBoxSitk")
        self.sitk_combo_box.addItems(
            [
                "Rigid (6DOF)",
                "Affine",
                "Non Rigid Bspline (>27DOF)",
                "Demons",
                "Diffeomorphic Demons",
                "Fast Symmetric Forces Demons",
                "Symmetric Forces Demons",
            ]
        )

        self.elastix_combo_box = self.get_ui(ctkComboBox, "ComboBoxElastix")
        self.elastix_logic = Elastix.ElastixLogic()
        for preset in self.elastix_logic.getRegistrationPresets():
            self.elastix_combo_box.addItem(
                "{0} ({1})".format(
                    preset[Elastix.RegistrationPresets_Modality],
                    preset[Elastix.RegistrationPresets_Content],
                )
            )

        self.settings_registration = self.get_ui(
            ctkCollapsibleButton, "RegistrationSettingsCollapsibleButton"
        )

        # :COMMENT: Bspline only
        self.bspline_group_box = self.get_ui(QGroupBox, "groupBoxNonRigidBspline")
        self.transform_domain_mesh_size = self.get_ui(
            QLineEdit, "lineEditTransformDomainMeshSize"
        )
        self.transform_domain_mesh_size.editingFinished.connect(
            self.verify_transform_domain_ms
        )
        self.scale_factor = self.get_ui(QLineEdit, "lineEditScaleFactor")
        self.scale_factor.editingFinished.connect(self.verify_scale_factor)
        self.shrink_factor = self.get_ui(QLineEdit, "lineEditShrinkFactor")
        self.shrink_factor.editingFinished.connect(
            lambda: self.verify_shrink_factor(self.shrink_factor)
        )
        self.smoothing_sigmas = self.get_ui(QLineEdit, "lineEditSmoothingFactor")
        self.smoothing_sigmas.editingFinished.connect(
            lambda: self.verify_shrink_factor(self.smoothing_sigmas)
        )

        # :COMMENT: Demons only
        self.demons_group_box = self.get_ui(QGroupBox, "groupBoxDemons")
        self.demons_nb_iter = self.get_ui(QLineEdit, "lineEditDemonsNbIter")
        self.demons_std_deviation = self.get_ui(QLineEdit, "lineEditDemonsStdDeviation")
        self.demons_std_deviation.editingFinished.connect(
            self.verify_demons_std_deviation
        )

        # :COMMENT: Gradients parameters
        self.gradients_box = self.get_ui(
            ctkCollapsibleGroupBox, "CollapsibleGroupBoxGradient"
        )
        self.learning_rate_spin_box = self.get_ui(
            QDoubleSpinBox, "doubleSpinBoxLearningR"
        )
        self.nb_of_iter_spin_box = self.get_ui(QSpinBox, "spinBoxNbIter")
        self.conv_min_val_edit = self.get_ui(QLineEdit, "lineEditConvMinVal")
        self.conv_win_size_spin_box = self.get_ui(QSpinBox, "spinBoxConvWinSize")
        self.conv_min_val_edit.editingFinished.connect(self.verify_convergence_min_val)

        # :COMMENT: exhaustive parameters
        self.exhaustive_box = self.get_ui(
            ctkCollapsibleGroupBox, "CollapsibleGroupBoxExhaustive"
        )
        self.step_length_edit = self.get_ui(QLineEdit, "lineEditLength")
        self.step_length_edit.editingFinished.connect(self.verify_step_length)
        self.nb_steps_edit = self.get_ui(QLineEdit, "lineEditSteps")
        self.nb_steps_edit.editingFinished.connect(self.verify_nb_steps)
        self.opti_scale_edit = self.get_ui(QLineEdit, "lineEditScale")
        self.opti_scale_edit.editingFinished.connect(self.verify_opti_scale_edit)

        # :COMMENT: LBFGSB parameters
        self.lbfgs2_box = self.get_ui(
            ctkCollapsibleGroupBox, "CollapsibleGroupBoxLBFGS2"
        )
        self.gradient_conv_tol_edit = self.get_ui(QLineEdit, "lineEditGradientConvTol")
        self.gradient_conv_tol_edit.editingFinished.connect(
            self.verify_gradient_conv_tol
        )
        self.nb_iter_lbfgs2 = self.get_ui(QSpinBox, "spinBoxNbIterLBFGS2")
        self.max_nb_correction_spin_box = self.get_ui(
            QSpinBox, "spinBoxMaxNbCorrection"
        )
        self.max_nb_func_eval_spin_box = self.get_ui(QSpinBox, "spinBoxMaxNbFuncEval")
        self.step_length_edit = self.get_ui(QLineEdit, "lineEditLength")
        self.nb_steps_edit = self.get_ui(QLineEdit, "lineEditSteps")
        self.opti_scale_edit = self.get_ui(QLineEdit, "lineEditScale")

        # :COMMENT: Fill them combo boxes.
        self.metrics_combo_box.addItems(
            [
                "Mean Squares",
                "Mattes Mutual Information",
                "Joint Histogram Mutual Information",
                "Correlation",
            ]
        )
        self.optimizers_combo_box.addItems(["Gradient Descent", "Exhaustive", "LBFGSB"])
        self.interpolator_combo_box.addItems(
            [
                "Linear",
                "Nearest Neighbor",
                "BSpline1",
                "BSpline2",
                "BSpline3",
                "Gaussian",
            ]
        )
        self.sampling_strat_combo_box.addItems(["None", "Regular", "Random"])

        # :COMMENT: handle button
        self.button_registration = self.get_ui(QPushButton, "PushButtonRegistration")
        self.button_registration.clicked.connect(self.register)

        self.button_cancel = self.get_ui(QPushButton, "pushButtonCancel")
        self.button_cancel.clicked.connect(self.cancel_registration_process)
        self.button_cancel.setEnabled(False)

        self.progressBar = self.get_ui(QProgressBar, "progressBar")
        self.progressBar.hide()
        self.label_status = self.get_ui(QLabel, "label_status")
        self.label_status.hide()

        self.optimizers_combo_box.currentIndexChanged.connect(
            self.update_registration_optimizer
        )
        self.metrics_combo_box.currentIndexChanged.connect(self.update_metrics_value)
        self.sitk_combo_box.activated.connect(
            lambda: self.update_registration(True, self.sitk_combo_box.currentIndex)
        )
        self.elastix_combo_box.activated.connect(
            lambda: self.update_registration(False, self.elastix_combo_box.currentIndex)
        )
        # :COMMENT: Initialize the registration.
        self.reset_registration()

    def reset_registration(self) -> None:
        """
        Reset all user parameters, disable all parameters until user selects an algorithm
        """
        self.metrics_combo_box.setCurrentIndex(-1)
        self.optimizers_combo_box.setCurrentIndex(-1)
        self.interpolator_combo_box.setCurrentIndex(-1)
        self.sitk_combo_box.setCurrentIndex(-1)
        self.elastix_combo_box.setCurrentIndex(-1)
        self.sampling_strat_combo_box.setCurrentIndex(2)
        self.bspline_group_box.setEnabled(False)
        self.demons_group_box.setEnabled(False)
        self.reset_optimizers_box()
        self.metrics_combo_box.setEnabled(False)
        self.interpolator_combo_box.setEnabled(False)
        self.optimizers_combo_box.setEnabled(False)
        self.settings_registration.setEnabled(False)
        self.bspline_group_box.setEnabled(False)
        self.demons_group_box.setEnabled(False)
        self.metric_related_value_spin_box.setEnabled(False)
        self.sampling_strat_combo_box.setEnabled(False)
        self.sampling_perc_spin_box.setEnabled(False)
        self.exhaustive_box.setEnabled(False)
        self.gradients_box.setEnabled(False)
        self.lbfgs2_box.setEnabled(False)

        self.metric_related_value_label.text = ""
        self.metric_related_value_spin_box.setValue(0)

    def reset_optimizers_box(self) -> None:
        self.gradients_box.setEnabled(False)
        self.gradients_box.collapsed = 1
        self.exhaustive_box.setEnabled(False)
        self.exhaustive_box.collapsed = 1
        self.lbfgs2_box.setEnabled(False)
        self.lbfgs2_box.collapsed = 1
        self.scale_factor.text = "1, 2, 4"
        self.scale_factor.setEnabled(True)


    def update_registration_optimizer(self) -> None:
        """
        Update the UI according to selected optimizer by user.
        """
        self.reset_optimizers_box()
        if self.optimizers_combo_box.currentText == "Gradient Descent":
            self.gradients_box.setEnabled(True)
            self.gradients_box.collapsed = 0
        elif self.optimizers_combo_box.currentText == "Exhaustive":
            self.exhaustive_box.setEnabled(True)
            self.exhaustive_box.collapsed = 0
        elif self.optimizers_combo_box.currentText == "LBFGSB":
            self.lbfgs2_box.setEnabled(True)
            self.lbfgs2_box.collapsed = 0
            self.scale_factor.text = "1, 1, 1"
            self.scale_factor.setEnabled(False)

    def update_metrics_value(self) -> None:
        """
        update the displayed value for metrics depending on the choice of the metrics
        """
        if "Mattes" in self.metrics_combo_box.currentText:
            self.metric_related_value_label.text = "number Of Histogram Bins"
            self.metric_related_value_spin_box.setValue(50)
            self.metric_related_value_spin_box.setEnabled(True)
        elif "Joint" in self.metrics_combo_box.currentText:
            self.metric_related_value_label.text = "number Of Histogram Bins"
            self.metric_related_value_spin_box.setValue(20)
            self.metric_related_value_spin_box.setEnabled(True)
        else:
            self.metric_related_value_label.text = ""
            self.metric_related_value_spin_box.setValue(0)
            self.metric_related_value_spin_box.setEnabled(False)

    def update_registration(self, is_sitk: bool, index: int) -> None:
        """
        update the UI according to the registration algorithm selected by the user.

        Parameters:
            is_sitk: boolean that indicates if the registration algorithm is from sitk
            index: the current index of the combo box
        """
        self.reset_registration()
        if is_sitk:
            self.sitk_combo_box.setCurrentIndex(index)
            self.settings_registration.setEnabled(True)
            self.elastix_combo_box.setCurrentIndex(-1)
            # if rigid, affine or bspline, those settings can be changed, thus are enabled
            if 0 <= index <= 2:
                self.metrics_combo_box.setEnabled(True)
                self.interpolator_combo_box.setEnabled(True)
                self.optimizers_combo_box.setEnabled(True)
                self.sampling_strat_combo_box.setEnabled(True)
                self.sampling_perc_spin_box.setEnabled(True)
            # if bspline, set visible psbline parameters
            if index == 2:
                self.bspline_group_box.setEnabled(True)
            # demons algorithms starts at 3
            if index >= 3:
                self.demons_group_box.setEnabled(True)
            # rigid registration are 0 and 1
            if index == 0 or index == 1:
                self.scriptPath = self.resourcePath("Scripts/Registration/Rigid.py")
            if index > 1:
                self.scriptPath = self.resourcePath("Scripts/Registration/NonRigid.py")
        # else elastix presets, scriptPath is None to verify later which function to use (custom_script_registration or elastix_registration)
        else:
            self.scriptPath = None
            self.sitk_combo_box.setCurrentIndex(-1)
            self.elastix_combo_box.setCurrentIndex(index)
        self.update_registration_optimizer()
        self.update_metrics_value()

    def register(self) -> None:
        """
        Apply a registration algorithm. Either sitk based, or using SlicerElastix algorithm.
        """

        # :COMMENT: Ensure the parameters are set.
        if not self.input_volume:
            self.display_error_message("No input volume selected.")
            return
        if not self.target_volume:
            self.display_error_message("No target volume selected.")
            return
        # :COMMENT: only allow metrics, interpolator and optimizer for rigid and bspline sitk
        if (
            self.elastix_combo_box.currentIndex == -1
            and self.sitk_combo_box.currentIndex == -1
        ):
            self.display_error_message("No registration algorithm selected.")
            return
        # Demons registration do not need metrics, interpolator and optimizer.
        if (
            self.elastix_combo_box.currentIndex == -1
            and 0 <= self.sitk_combo_box.currentIndex < 3
        ):
            if self.metrics_combo_box.currentIndex == -1:
                self.display_error_message("No metrics selected.")
                return
            if self.interpolator_combo_box.currentIndex == -1:
                self.display_error_message("No interpolator selected.")
                return
            if self.optimizers_combo_box.currentIndex == -1:
                self.display_error_message("No optimizer selected.")
                return

        # :COMMENT: utilitiy functions to get sitk images
        fixed_image = su.PullVolumeFromSlicer(self.target_volume)
        moving_image = su.PullVolumeFromSlicer(self.input_volume)
        fixed_image = sitk.Cast(fixed_image, sitk.sitkFloat32)
        moving_image = sitk.Cast(moving_image, sitk.sitkFloat32)

        # allows not to update view if registration has been cancelled
        self.registration_cancelled = False

        input = {}
        current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        volume_name = f"{self.input_volume.GetName()}_registered_{current_time}"
        input["algorithm"] = self.sitk_combo_box.currentText.replace(" ", "")
        input["volume_name"] = volume_name
        self.data_to_dictionary(input)
        if self.scriptPath:
            self.custom_script_registration(
                self.scriptPath, fixed_image, moving_image, input
            )
        else:
            self.elastix_registration()

    def data_to_dictionary(self, data_dictionary) -> None:
        """
        Fills the data_dictionary parameters with user prompts, dictionary used for registration

        Parameters:
            data_dictionary: the dictionary to fill.
        """
        # :COMMENT:---- User settings retrieve -----
        data_dictionary["histogram_bin_count"] = int(
            self.metric_related_value_spin_box.value
        )
        # :COMMENT: Sampling strategies range from 0 to 2, they are enums (None, Regular, Random), thus index is sufficient
        data_dictionary[
            "sampling_strategy"
        ] = self.sampling_strat_combo_box.currentIndex
        data_dictionary["sampling_percentage"] = self.sampling_perc_spin_box.value
        data_dictionary["metrics"] = self.metrics_combo_box.currentText.replace(" ", "")
        data_dictionary["interpolator"] = self.interpolator_combo_box.currentText
        data_dictionary["optimizer"] = self.optimizers_combo_box.currentText

        # :COMMENT: Bspline settings only
        # test : all others have positive 3 positive values (mutli-resolution approach)
        data_dictionary["transform_domain_mesh_size"] = int(
            self.transform_domain_mesh_size.text
        )
        data_dictionary["scale_factor"] = [
            int(factor) for factor in self.scale_factor.text.split(",")
        ]
        data_dictionary["shrink_factor"] = [
            int(factor) for factor in self.shrink_factor.text.split(",")
        ]
        data_dictionary["smoothing_sigmas"] = [
            int(sig) for sig in self.smoothing_sigmas.text.split(",")
        ]

        # :COMMENT: settings for gradients only
        data_dictionary["learning_rate"] = self.learning_rate_spin_box.value
        data_dictionary["nb_iteration"] = self.nb_of_iter_spin_box.value
        data_dictionary["convergence_min_val"] = float(self.conv_min_val_edit.text)
        data_dictionary["convergence_win_size"] = int(self.conv_win_size_spin_box.value)

        # :COMMENT: settings for exhaustive only
        nb_of_steps = self.nb_steps_edit.text
        data_dictionary["nb_of_steps"] = [int(step) for step in nb_of_steps.split(",")]
        self.step_length = self.step_length_edit.text
        if self.step_length == "pi":
            data_dictionary["step_length"] = pi
        else:
            data_dictionary["step_length"] = float(self.step_length)

        optimizer_scale = self.opti_scale_edit.text
        data_dictionary["optimizer_scale"] = [
            int(scale) for scale in optimizer_scale.split(",")
        ]

        # :COMMENT: settings for LBFGSB
        data_dictionary["gradient_conv_tol"] = float(self.gradient_conv_tol_edit.text)
        data_dictionary["nb_iter_lbfgsb"] = self.nb_iter_lbfgs2.value
        data_dictionary["max_nb_correction"] = self.max_nb_correction_spin_box.value
        data_dictionary["max_func_eval"] = self.max_nb_func_eval_spin_box.value

        # :COMMENT: settings for demons
        data_dictionary["demons_nb_iter"] = int(self.demons_nb_iter.text)
        data_dictionary["demons_std_dev"] = float(self.demons_std_deviation.text)

    # :TODO:Tony: déporter les tests dans ce fichier
    def custom_script_registration(
        self, scriptPath, fixed_image, moving_image, input
    ) -> None:
        """
        Calls parallelProcessing extension and execute a registration script as a background task.

        Parameters:
            scriptPath: the path to the script.
            fixed_image: the reference image.
            moving_image: the image to registrate.
            input: the dictionary that contains user parameters.
        """
        self.elastix_logic = None
        self.process_logic = ProcessesLogic(
            completedCallback=lambda: self.on_registration_completed()
        )
        self.regProcess = RegistrationProcess(
            scriptPath, fixed_image, moving_image, input
        )
        self.process_logic.addProcess(self.regProcess)
        self.button_registration.setEnabled(False)
        self.button_cancel.setEnabled(True)
        self.activate_timer_and_progress_bar()
        self.process_logic.run()

    def elastix_registration(self) -> None:
        """
        Calls SlicerElastix registration with the selected preset by the user.
        Adds a registrated volume if registration is complete.
        """
        self.regProcess = None
        self.elastix_logic = Elastix.ElastixLogic()
        self.button_registration.setEnabled(False)
        self.button_cancel.setEnabled(True)
        self.activate_timer_and_progress_bar()
        preset = self.elastix_combo_box.currentIndex
        parameterFilenames = self.elastix_logic.getRegistrationPresets()[preset][
            Elastix.RegistrationPresets_ParameterFilenames
        ]
        new_volume = mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        try:
            self.elastix_logic.registerVolumes(
                self.target_volume,
                self.input_volume,
                parameterFilenames=parameterFilenames,
                outputVolumeNode=new_volume,
            )
        except ValueError as ve:
            print(ve)
        finally:
            self.on_registration_completed()

    def on_registration_completed(self) -> None:
        """
        Handles the completion callback.
        Stops the ProgressBar and timer.
        """
        self.progressBar.setMaximum(100)
        self.progressBar.setValue(100)
        self.timer.stop()
        self.button_registration.setEnabled(True)
        self.button_cancel.setEnabled(False)
        # :COMMENT: Log the registration.
        if not self.registration_cancelled:
            if (
                self.regProcess and self.regProcess.registration_completed
            ) or self.elastix_logic:
                assert self.input_volume
                print(
                    f'"{self.input_volume.GetName()}" has been registered as "{self.volumes[len(self.volumes) - 1].GetName()}".'
                )
                self.choose_input_volume(len(self.volumes) - 1)
            if self.regProcess and self.regProcess.message_error:
                self.display_error_message(self.regProcess.message_error)

    def activate_timer_and_progress_bar(self) -> None:
        """
        Starts the progressBar activation and a timer to displays elapsed time.
        """
        self.progressBar.setVisible(True)
        self.label_status.setVisible(True)
        self.progressBar.setMinimum(0)
        self.progressBar.setMaximum(0)
        self.progressBar.setValue(0)
        self.timer = QTimer()
        self.elapsed_time = QElapsedTimer()
        self.timer.timeout.connect(self.update_status)
        self.timer.start(1000)
        self.elapsed_time.start()

    def update_status(self) -> None:
        """
        displays elapsed time
        """
        self.label_status.setText(f"status: {self.elapsed_time.elapsed()//1000}s")

    def cancel_registration_process(self) -> None:
        """
        Stops progressBar, timer and kills the registration process.
        """
        self.timer.stop()
        self.progressBar.setMaximum(100)
        self.progressBar.setValue(0)
        self.registration_cancelled = True
        if self.regProcess:
            self.regProcess.registration_completed = False
            self.terminate_process_logic()
            print("User requested cancel.")
        if self.elastix_logic:
            self.elastix_logic.abortRequested = True

    def terminate_process_logic(self) -> None:
        """
        kills the slicerParallelProcessing registration processus.
        """
        import os
        import signal

        assert self.regProcess
        os.kill(self.regProcess.processId() + 10, signal.SIGKILL)
        self.regProcess.kill()

    def verify_convergence_min_val(self) -> None:
        """
        Assert that the content of the convergence minimum value is correct.
        """
        value = self.conv_min_val_edit.text
        try:
            float(value)
            if float(value) < 0 or float(value) > 1:
                self.display_error_message("value must be between 0 and 1.")
                self.conv_min_val_edit.text = "1e-6"
        except ValueError:
            self.display_error_message("not a value.")
            self.conv_min_val_edit.text = "1e-6"

    def verify_nb_steps(self) -> None:
        """
        Assert that the content of the number of steps value is correct.
        """
        nb_of_steps = self.nb_steps_edit.text
        try:
            nb_of_steps = [int(step) for step in nb_of_steps.split(",")]
            if len(nb_of_steps) != 6:
                self.display_error_message("must have 6 values.")
                self.nb_steps_edit.text = "1, 1, 1, 0, 0, 0"
                return
            if any(step < 0 for step in nb_of_steps):
                self.display_error_message("must be positive values.")
                self.nb_steps_edit.text = "1, 1, 1, 0, 0, 0"
        except ValueError:
            self.display_error_message("not values.")
            self.nb_steps_edit.text = "1, 1, 1, 0, 0, 0"

    def verify_step_length(self) -> None:
        """
        Assert that the content of the step length value is correct.
        """
        step_length = self.step_length_edit.text
        if step_length == "pi":
            return
        try:
            float(step_length)
        except ValueError:
            self.display_error_message("not a value.")
            self.step_length_edit.text = "pi"

    def verify_opti_scale_edit(self) -> None:
        """
        Assert that the content of the optimizer scale value is correct.
        """
        optimizer_scale = self.opti_scale_edit.text
        try:
            optimizer_scale = [int(scale) for scale in optimizer_scale.split(",")]
            if len(optimizer_scale) != 6:
                self.display_error_message("must have 6 values.")
                self.opti_scale_edit.text = "1,1,1,1,1,1"
                return
            if any(scale < 0 for scale in optimizer_scale):
                self.display_error_message("must be positive values")
                self.opti_scale_edit.text = "1,1,1,1,1,1"
        except ValueError:
            self.display_error_message("not values")
            self.opti_scale_edit.text = "1,1,1,1,1,1"

    def verify_gradient_conv_tol(self) -> None:
        """
        Assert that the content of the gradient convergence tolerance value is correct.
        """
        value = self.gradient_conv_tol_edit.text
        try:
            float(value)
            if float(value) <= 0:
                self.display_error_message("must be a positive value.")
                self.gradient_conv_tol_edit.text = "1e-5"
        except ValueError:
            self.display_error_message("not a value.")
            self.gradient_conv_tol_edit.text = "1e-5"

    def verify_demons_std_deviation(self) -> None:
        """
        Assert that the content of the standard deviation value is correct.
        """
        value = self.demons_std_deviation.text
        try:
            float(value)
            if float(value) <= 0:
                self.display_error_message("must be a positive value")
                self.demons_std_deviation.text = "1.0"
        except ValueError:
            self.display_error_message("not a value.")
            self.demons_std_deviation.text = "1.0"

    def verify_transform_domain_ms(self) -> None:
        """
        Assert that the content of the transform domain mesh size value is correct.
        """
        value = self.transform_domain_mesh_size.text
        try:
            if int(value) <= 0:
                self.display_error_message("must be a positive integer.")
                self.transform_domain_mesh_size.text = "2"
        except ValueError:
            self.display_error_message("not a value or float entered.")
            self.transform_domain_mesh_size.text = "2"

    def verify_scale_factor(self) -> None:
        """
        Assert that the content of the scale factor vector is correct.
        """
        try:
            scale_factor = [int(factor) for factor in self.scale_factor.text.split(",")]
            if len(scale_factor) != 3:
                self.display_error_message("must have 3 values.")
                self.scale_factor.text = "1, 2, 4"
                return
            if any(factor < 0 for factor in scale_factor):
                self.display_error_message("must have positive values.")
                self.scale_factor.text = "1, 2, 4"
        except ValueError:
            self.display_error_message("not values.")
            self.scale_factor.text = "1, 2, 4"

    def verify_shrink_factor(self, QLineEdit) -> None:
        """
        Assert that the content of the shrink factor vector is correct.
        """
        try:
            factor = [int(factor) for factor in QLineEdit.text.split(",")]
            if len(factor) != 3:
                self.display_error_message("must have 3 values.")
                QLineEdit.text = "4, 2, 1"
                return
            if any(factor < 0 for factor in factor):
                self.display_error_message("must have positive values.")
                QLineEdit.text = "4, 2, 1"
        except ValueError:
            self.display_error_message("not values.")
            QLineEdit.text = "4, 2, 1"

    #
    # DIFFERENCE MAP
    #

    def setup_difference_map(self) -> None:
        """
        Sets up the difference map computation by retrieving and connecting the UI data.
        """

        self.ALGORITHMS = ["Mean squared error", "Gradient"]

        self.algorithms_difference_map_combo_box = self.get_ui(
            ctkComboBox, "mapFunction"
        )
        for algo in self.ALGORITHMS:
            self.algorithms_difference_map_combo_box.addItem(algo)
        self.algorithms_difference_map_combo_box.setCurrentIndex(-1)

        self.difference_map_button_apply = self.get_ui(
            QPushButton, "processMapFunction"
        )

        self.spin_box = self.get_ui(QSpinBox, "patchSizeSpinBox")

        self.mean = self.get_ui(QLabel, "mean")

        self.difference_map_button_apply.clicked.connect(self.compute_difference_map)

        self.reset_difference_map()

    def reset_difference_map(self) -> None:
        """
        Resets the difference map computation to the defaults.
        """

        self.spin_box.value = 0
        self.difference_map_current_algorithm = None

        self.update_difference_map()

    def update_difference_map(self) -> None:
        """
        Updates the difference map computation.
        """

        # :COMMENT: Nothing to update!

    def compute_difference_map(self) -> None:
        """
        Applies the choosen algorithm for difference map (Gradient or MSE)
        """

        if not self.input_volume or not self.target_volume:
            self.display_error_message("Please select both input and target volumes.")
            return

        algorithm = self.algorithms_difference_map_combo_box.currentText
        if algorithm not in self.ALGORITHMS:
            self.display_error_message("Please select an algorithm.")
            return

        difference_map_mean = -1
        try:
            if algorithm == "Mean squared error":
                (
                    self.difference_map,
                    difference_map_mean,
                ) = self.logic.difference_map(
                    self.input_volume,
                    self.target_volume,
                    f"{self.input_volume.GetName()} {self.target_volume.GetName()} P{self.spin_box.value} MSE Difference Map",
                    "absolute",
                    self.spin_box.value,
                )

            if algorithm == "Gradient":
                (
                    self.difference_map,
                    difference_map_mean,
                ) = self.logic.difference_map(
                    self.input_volume,
                    self.target_volume,
                    f"{self.input_volume.GetName()} {self.target_volume.GetName()} P{self.spin_box.value} Gradient Difference Map",
                    "gradient",
                    self.spin_box.value,
                )
        except ValueError as e:
            self.display_error_message(str(e))
            self.difference_map = None
            return

        self.update_allowed = False
        mrmlScene.AddNode(self.difference_map)
        self.update_allowed = True
        self.mean.text = difference_map_mean
        self.plot_difference_map()

    def plot_difference_map(self) -> None:
        """
        Plots the difference map on the yellow window and apply a Look Up Table
        """

        assert self.difference_map

        difference_map_display_node = self.difference_map.GetDisplayNode()
        if not difference_map_display_node:
            difference_map_display_node = vtkMRMLScalarVolumeDisplayNode()
            mrmlScene.AddNode(difference_map_display_node)
            self.difference_map.SetAndObserveDisplayNodeID(
                difference_map_display_node.GetID()
            )

        difference_map_display_node.SetAndObserveColorNodeID(
            util.getNode("ColdToHotRainbow").GetID()
        )

        self.update_specific_view(2, self.difference_map, None)

    #
    # PLUGIN LOADING
    #

    def setup_plugin_loading(self) -> None:
        """
        …
        """

        # :COMMENT: Initialize the plugin script list.
        self.plugins = {}

        # :COMMENT: Retrieve the plugin loading button.
        self.plugin_loading_button = self.get_ui(QPushButton, "PluginLoadingPushButton")

        # :COMMENT: Define the handler.
        def on_plugin_loading_button_clicked() -> None:
            """
            Opens a loading window with a label for the name of the plugin, and two horizontal layouts, one for the UI file and one for the Python file, each with a label for the name of the file and a button to load the file.
            """

            # :COMMENT: Create an empty dialog.
            dialog = QDialog(self.parent)
            dialog.setWindowTitle("Plugin Loading")

            # :COMMENT: Create a base vertical layout.
            base_layout = QVBoxLayout()
            base_layout.setContentsMargins(12, 12, 12, 12)
            base_layout.setSpacing(12)
            dialog.setLayout(base_layout)

            # :COMMENT: Create an horizontal layout with a label for the description and a line edit for the name of the plugin (My Plugin by default).
            name_label = QLabel("Plugin Name:")
            name_line_edit = QLineEdit()
            name_line_edit.setText("My Plugin")
            name_layout = QHBoxLayout()
            name_layout.setSpacing(12)
            name_layout.addWidget(name_label)
            name_layout.addWidget(name_line_edit)
            base_layout.addLayout(name_layout)

            # :COMMENT: Create an horizontal layout for the UI file loading with a label for the name of the file and a button to load this file.
            self.plugin_loading_ui_file = None
            ui_file_label = QLabel("No UI file selected.")
            ui_file_button = QPushButton()
            ui_file_button.setText("Choose an UI file…")
            ui_file_layout = QHBoxLayout()
            ui_file_layout.setSpacing(12)
            ui_file_layout.addWidget(ui_file_label)
            ui_file_layout.addWidget(ui_file_button)
            base_layout.addLayout(ui_file_layout)

            # :COMMENT: Create an horizontal layout for the Python file loading with a label for the name of the file and a button to load this file.
            self.plugin_loading_python_file = None
            python_file_label = QLabel("No Python file selected.")
            python_file_button = QPushButton()
            python_file_button.setText("Choose a Python file…")
            python_file_layout = QHBoxLayout()
            python_file_layout.setSpacing(12)
            python_file_layout.addWidget(python_file_label)
            python_file_layout.addWidget(python_file_button)
            base_layout.addLayout(python_file_layout)

            def on_ui_file_button_clicked() -> None:
                """
                Opens a file opening dialog for a UI file.
                """

                def on_ui_file_dialog_finished(result) -> None:
                    """
                    Loads the UI file.

                    Parameters:
                        result: The result of the file dialog.
                    """

                    if result == QDialog.Accepted:
                        path = ui_file_dialog.selectedFiles()[0]
                        ui_file_label.setText(os.path.basename(path))
                        self.plugin_loading_ui_file = path

                    dialog.raise_()

                # :COMMENT: Create a file dialog for the UI file.
                ui_file_dialog = QFileDialog(self.parent)
                ui_file_dialog.setFileMode(QFileDialog.ExistingFile)
                ui_file_dialog.setAcceptMode(QFileDialog.AcceptOpen)
                ui_file_dialog.setNameFilter("*.ui")
                ui_file_dialog.finished.connect(on_ui_file_dialog_finished)
                ui_file_dialog.show()

            def on_python_file_button_clicked() -> None:
                """
                Opens a file opening dialog for a Python file.
                """

                def on_python_file_dialog_finished(result) -> None:
                    """
                    Loads the Python file.

                    Parameters:
                        result: The result of the file dialog.
                    """

                    if result == QDialog.Accepted:
                        path = python_file_dialog.selectedFiles()[0]
                        python_file_label.setText(os.path.basename(path))
                        self.plugin_loading_python_file = path

                    dialog.raise_()

                # :COMMENT: Create a file dialog for the Python file.
                python_file_dialog = QFileDialog(self.parent)
                python_file_dialog.setFileMode(QFileDialog.ExistingFile)
                python_file_dialog.setAcceptMode(QFileDialog.AcceptOpen)
                python_file_dialog.setNameFilter("*.py")
                python_file_dialog.finished.connect(on_python_file_dialog_finished)
                python_file_dialog.show()

            # :COMMENT: Connect the buttons.
            ui_file_button.clicked.connect(on_ui_file_button_clicked)
            python_file_button.clicked.connect(on_python_file_button_clicked)

            def on_load_button_clicked() -> None:
                """
                Loads the new plugin.
                """

                # :COMMENT: Retrieve the plugin name.
                plugin_name = name_line_edit.text

                # :COMMENT: Check if the plugin name is valid.
                if plugin_name in self.plugins.keys():
                    self.display_error_message(
                        f'A plugin named "{plugin_name}" already exists.'
                    )
                    return

                # :COMMENT: Check if the UI file is valid.
                if not self.plugin_loading_ui_file:
                    self.display_error_message("No UI file selected.")
                    return

                # :COMMENT: Check if the Python file is valid.
                if not self.plugin_loading_python_file:
                    self.display_error_message("No Python file selected.")
                    return

                # :COMMENT: Retrieve the plugins layout.
                self.plugins_layout = self.get_ui(QVBoxLayout, "PluginsVerticalLayout")

                # :COMMENT: Add a collapsible button.
                plugin_collapsible_button = ctkCollapsibleButton()
                plugin_collapsible_button.text = plugin_name
                plugin_collapsible_button.collapsed = True
                plugin_layout = QVBoxLayout()
                plugin_layout.setContentsMargins(12, 12, 0, 12)
                plugin_layout.setSpacing(12)
                plugin_collapsible_button.setLayout(plugin_layout)
                self.plugins_layout.addWidget(plugin_collapsible_button)

                # :COMMENT: Add the UI of the file inside the collapsible widget.
                plugin_ui = util.loadUI(self.plugin_loading_ui_file)
                assert plugin_ui
                plugin_ui.setPalette(util.mainWindow().palette)
                plugin_layout.addWidget(plugin_ui)

                def on_run_button_clicked() -> None:
                    """
                    Runs the plugin.
                    """

                    plugin_folder = os.path.dirname(self.plugins[plugin_name])
                    plugin_file = os.path.basename(
                        os.path.splitext(self.plugins[plugin_name])[0]
                    )

                    import sys

                    sys.path.append(plugin_folder)

                    import importlib

                    plugin_script = importlib.import_module(plugin_file)

                    plugin_script.run(
                        ui=plugin_ui,
                        scene=mrmlScene,
                        input_volume=self.input_volume,
                        target_volume=self.target_volume,
                    )

                # :COMMENT: Add the run button to launch the plugin script.
                plugin_run_button = QPushButton()
                plugin_run_button.setText(f"Run {plugin_name}")
                plugin_layout.addWidget(plugin_run_button)
                plugin_run_button.clicked.connect(on_run_button_clicked)

                # :COMMENT: Add the plugin path to the plugin list.
                self.plugins[plugin_name] = self.plugin_loading_python_file

                # :COMMENT: Reset the temporary variables and close the dialog.
                self.plugin_loading_ui_file = None
                self.plugin_loading_python_file = None
                dialog.accept()

            # :COMMENT: Add a load button and connect it to a dedicated handler.
            load_button = QPushButton()
            load_button.setText("Load")
            base_layout.addWidget(load_button)
            load_button.clicked.connect(on_load_button_clicked)

            # :COMMENT: Show the dialog.
            dialog.show()

        # :COMMENT: Connect the handler.
        self.plugin_loading_button.clicked.connect(on_plugin_loading_button_clicked)

    def reset_plugin_loading(self) -> None:
        """
        …
        """

        # :COMMENT: Nothing to reset.

        # :COMMENt: Update the plugin loading.
        self.update_plugin_loading()

    def update_plugin_loading(self) -> None:
        """
        …
        """

        # :COMMENT: Nothing to update.

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

    def get_volume_by_name(self, name: str) -> vtkMRMLScalarVolumeDisplayNode:
        """
        Retrieves a volume by its name.

        Parameters:
            name: The name of the volume to retrieve.

        Returns:
            The VTK volume.
        """

        # :COMMENT: Search for the volume by its name.
        for i in range(len(self.volumes)):
            volume = self.volumes[i]
            if volume.GetName() == name:
                return volume

        # :COMMENT: Return None if the volume was not found.
        return None

    def add_new_volume(self, volume: vtkMRMLScalarVolumeNode, name: str) -> None:
        """
        Adds a new volume to the scene.

        Parameters:
            volume: VTK volume to be added.
            name: Type of processing.
        """

        # :COMMENT: Ensure that a volume is selected.
        assert self.input_volume

        # :COMMENT: Generate and assign a unique name to the volume.
        current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        new_name = f"{self.input_volume.GetName()}_{name}_{current_time}"
        volume.SetName(new_name)

        # :COMMENT: Update the MRML scene.
        mrmlScene.AddNode(volume)

    def get_ui(self, type, name: str):
        """
        Retrieves a UI object from the panel.

        Parameters:
            type: The type of the UI to retrieve.
            name: The name of the UI to retrieve.
        """

        ui = self.panel.findChild(type, name)
        if not ui:
            raise AssertionError(f'No {type} with name "{name}" found.')
        return ui


class CustomRegistrationTest(ScriptedLoadableModuleTest, unittest.TestCase):
    """
    Test class for the Custom Registration module used to define the tests of the module.
    """

    def __init__(self):
        ScriptedLoadableModuleTest().__init__()
        unittest.TestCase.__init__(self)

    def resourcePath(self, path: str) -> str:
        """
        Returns the absolute path to the resource with the given name.

        Parameters:
            path: The name of the resource.

        Returns:
            The absolute path to the resource.
        """

        module_path = os.path.dirname(modules.customregistration.path)
        return os.path.join(module_path, "Resources", path)

    def runTest(self):
        """
        Runs all the tests in the Custom Registration module.
        """

        self.logic = CustomRegistrationLogic()

        self.test_dummy()
        self.test_cropping()
        self.setup_test_registration()
        self.test_rigid_1()

    def test_dummy(self):
        """
        Dummy test to check if the module works as expected.
        """

        print("Dummy test passed.")

    def setup_test_registration(self):
        self.fixed_image = self.resourcePath("TestData/RegLib_C01_MRMeningioma_1.nrrd")
        self.moving_image = self.resourcePath("TestData/RegLib_C01_MRMeningioma_2.nrrd")
        self.fixed_image = sitk.ReadImage(self.fixed_image, sitk.sitkFloat32)
        self.moving_image = sitk.ReadImage(self.moving_image, sitk.sitkFloat32)

    def test_rigid_1(self):
        parameters = {}
        parameters["metrics"] = "MeanSquares"
        parameters["interpolator"] = "Linear"
        parameters["optimizer"] = "Gradient Descent"
        parameters["algorithm"] = "rigid"
        parameters["histogram_bin_count"] = 50
        parameters["sampling_strategy"] = 2
        parameters["sampling_percentage"] = 0.01
        # parameters for gradient optimizer
        parameters["learning_rate"] = 5
        parameters["nb_iteration"] = 100
        parameters["convergence_min_val"] = 1e-6
        parameters["convergence_win_size"] = 10
        from Resources.Scripts.Registration.Rigid import rigid_registration

        final_transform = rigid_registration(
            self.fixed_image, self.moving_image, parameters
        )
        expected_transform = sitk.ReadTransform(
            self.resourcePath("TestData/expected_transform_1.tfm")
        )
        self.assertIsNotNone(final_transform)
        self.assertEqual(
            final_transform.GetDimension(), expected_transform.GetDimension()
        )
        self.assertEqual(
            final_transform.GetNumberOfFixedParameters(),
            expected_transform.GetNumberOfFixedParameters(),
        )
        self.assertEqual(
            final_transform.GetNumberOfParameters(),
            expected_transform.GetNumberOfParameters(),
        )
        for x, y in zip(
            final_transform.GetParameters(), expected_transform.GetParameters()
        ):
            self.assertAlmostEqual(x, y, delta=0.01)
        for x, y in zip(
            final_transform.GetFixedParameters(),
            expected_transform.GetFixedParameters(),
        ):
            self.assertAlmostEqual(x, y, delta=0.01)

        print("rigid test 1 passed.")

    def test_cropping(self):
        # :COMMENT: Load a volume as test data.
        volume = util.loadVolume(self.resourcePath("TestData/MR-head.nrrd"))

        # :COMMENT: Define the crop parameters.
        start = [50, 50, 50]
        end = [200, 200, 100]

        # :COMMENT: Check that invalid parameters are rejected.
        with self.assertRaises(TypeError):
            self.logic.crop(volume, start, [end[i] + 1000 for i in range(3)])

        with self.assertRaises(TypeError):
            self.logic.crop(volume, [start[i] - 1000 for i in range(3)], end)

        # :COMMENT: Call our function on valid parameters.
        cropped_volume = vtkMRMLScalarVolumeNode()
        try:
            cropped_volume = self.logic.crop(volume, start, end)
        except RuntimeError:
            print("[ERROR] Cropping test failed.")
            return

        # :COMMENT: Check that the resulting cropped image has the expected dimensions.
        self.assertSequenceEqual(
            cropped_volume.GetImageData().GetDimensions(), [150, 150, 50]
        )

        # :COMMENT: Check that the resulting cropped image has the expected spacing.
        self.assertSequenceEqual(cropped_volume.GetSpacing(), volume.GetSpacing())

        # :COMMENT: Check that the resulting cropped image has the expected origin.
        self.assertSequenceEqual(cropped_volume.GetOrigin(), volume.GetOrigin())

        # :COMMENT: Check that the resulting cropped image has the expected direction.
        cropped_volume_direction = vtk.vtkMatrix4x4()
        volume_direction = vtk.vtkMatrix4x4()

        cropped_volume.GetIJKToRASDirectionMatrix(cropped_volume_direction)
        volume.GetIJKToRASDirectionMatrix(volume_direction)

        cropped_volume_direction_array = [
            [int(cropped_volume_direction.GetElement(i, j)) for j in range(4)]
            for i in range(4)
        ]
        volume_direction_array = [
            [int(volume_direction.GetElement(i, j)) for j in range(4)] for i in range(4)
        ]

        self.assertSequenceEqual(cropped_volume_direction_array, volume_direction_array)

        # :COMMENT: Check that the resulting cropped image has the expected content.
        volume_array = vtk.util.numpy_support.vtk_to_numpy(volume.GetImageData().GetPointData().GetScalars())  # type: ignore
        volume_array = np.reshape(
            volume_array, volume.GetImageData().GetDimensions()[::-1]
        )

        cropped_array = vtk.util.numpy_support.vtk_to_numpy(cropped_volume.GetImageData().GetPointData().GetScalars())  # type: ignore
        cropped_array = np.reshape(
            cropped_array, cropped_volume.GetImageData().GetDimensions()[::-1]
        )

        expected_array = volume_array[
            start[2] : end[2], start[1] : end[1], start[0] : end[0]
        ]
        self.assertTrue(np.array_equal(cropped_array, expected_array))

        mrmlScene.RemoveNode(volume)
        volume = None

        print("Cropping test passed.")


class RegistrationProcess(Process):
    """
    Class to process registration as a background task using the extension ParallelProcessing

    Parameters:
        scriptPath: path to the custom script user wants to execute (registration only)
        fixed_image : a sitk image, the image to be aligned with
        moving_image : a sitk image, the source image to be registered
        input_parameters: a dictionary used to pass parameters for the script
    """

    def __init__(self, scriptPath, fixed_image, moving_image, input_parameters):
        Process.__init__(self, scriptPath)
        self.fixed_image = fixed_image
        self.moving_image = moving_image
        self.input_parameters = input_parameters
        self.registration_completed = True
        self.message_error = None

    def prepareProcessInput(self) -> bytes:
        """
        Helper function to send input parameters to a script
        """

        input = {}
        input["fixed_image"] = self.fixed_image
        input["moving_image"] = self.moving_image
        input["parameters"] = self.input_parameters
        return pickle.dumps(input)

    def useProcessOutput(self, processOutput) -> None:
        """
        Helper function to received output parameters from the script
        Write to slicer the registrated volume if no error occured.

        Parameters:
            processOutput: a dictionary that contains the results of the script (a registration, a transform...)
        """
        if self.registration_completed:
            output = pickle.loads(processOutput)
            image_resampled = output["image_resampled"]
            volume_name = output["volume_name"]
            if not image_resampled:
                self.message_error = output["error"]
                self.registration_completed = False
                return
            print("Optimizer stop condition: " + output["stop_condition"])
            print("Number of iteration: " + str(output["nb_iteration"]))
            print("Metric value: " + str(output["metric_value"]))
            su.PushVolumeToSlicer(image_resampled, name=volume_name)


class AutocroppingValueError(ValueError):
    def __init__(self, message, axis):
        assert axis in AXIS_MAP
        super().__init__(message)
        self.axis = axis
