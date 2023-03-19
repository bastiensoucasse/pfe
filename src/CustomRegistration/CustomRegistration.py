"""
The Custom Registration module for Slicer provides the features for 3D images registration, based on the ITK library.
"""

import datetime
import os
import pickle
from math import pi

import numpy as np
import SimpleITK as sitk
import sitkUtils as su
import vtk
from ctk import ctkCollapsibleButton, ctkCollapsibleGroupBox, ctkComboBox
from Processes import Process, ProcessesLogic
from qt import (
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
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

        roi = self.sitk_to_vtk(sitk.Mask(self.vtk_to_sitk(volume), binary))
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
        Sets up the widget for the module by adding a welcome message to the layout.
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

        # :COMMENT: Set up the module regions.
        self.setup_view()
        self.setup_pascal_only_mode()
        self.setup_roi_selection()
        self.setup_cropping()
        self.setup_resampling()
        self.setup_registration()
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
        self.update_cropping()
        self.update_resampling()
        self.update_registration()
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
        # :TODO:Bastien: Add support for the difference map (during merge).
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
        self.cropping_collapsible_button.clicked.connect(self.update_cropping)

        # :COMMENT: Get the coordinates spinbox widgets.
        self.cropping_start = []
        self.cropping_end = []
        axis = ["x", "y", "z"]
        for i in range(len(axis)):
            self.cropping_start.append(self.get_ui(QSpinBox, "s" + axis[i]))
            self.cropping_end.append(self.get_ui(QSpinBox, "e" + axis[i]))

            # :COMMENT: Connect the spinbox widgets to their "on changed" function that displays the cropping preview.
            self.cropping_start[i].valueChanged.connect(self.update_cropping)
            self.cropping_end[i].valueChanged.connect(self.update_cropping)

        # :COMMENT: Get the crop button widget.
        self.cropping_button = self.get_ui(QPushButton, "crop_button")
        self.cropping_button.clicked.connect(self.crop)

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
        self.update_cropping()

    def update_cropping(self) -> None:
        """
        …
        """

        # :COMMENT: Hide the cropping box by default.
        if self.cropping_box and self.cropping_box.GetDisplayNode():
            self.cropping_box.GetDisplayNode().SetVisibility(False)

        # :COMMENT: Reset the cropping value ranges.
        if self.input_volume:
            input_volume_image_data = self.input_volume.GetImageData()
            input_volume_dimensions = input_volume_image_data.GetDimensions()
            self.cropping_preview_allowed = False
            for i in range(3):
                self.cropping_start[i].setMaximum(input_volume_dimensions[i])
                self.cropping_end[i].setMaximum(input_volume_dimensions[i])
            self.cropping_preview_allowed = True

        if (
            self.cropping_collapsible_button.isChecked()
            and self.cropping_preview_allowed
        ):
            self.preview_cropping()

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

        # :COMMENT: Set the update rule to allowed.
        self.update_allowed = True

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
        …
        """

        # :COMMENT: Link settings UI and code
        self.metrics_combo_box = self.get_ui(ctkComboBox, "ComboMetrics")
        self.interpolator_combo_box = self.get_ui(ctkComboBox, "comboBoxInterpolator")
        self.optimizers_combo_box = self.get_ui(ctkComboBox, "ComboOptimizers")
        self.histogram_bin_count_spin_box = self.get_ui(QSpinBox, "spinBoxBinCount")
        self.sampling_strat_combo_box = self.get_ui(
            ctkComboBox, "comboBoxSamplingStrat"
        )
        self.sampling_perc_spin_box = self.get_ui(
            QDoubleSpinBox, "doubleSpinBoxSamplingPerc"
        )

        # :COMMENT: registration types
        self.non_rigid_r_button = self.get_ui(QRadioButton, "radioButtonNonRigid")
        self.rigid_r_button = self.get_ui(QRadioButton, "radioButtonRigid")
        self.elastix_r_button = self.get_ui(QRadioButton, "radioButtonElastix")
        self.rigid_r_button.toggle()

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

        # :COMMENT: exhaustive parameters
        self.exhaustive_box = self.get_ui(
            ctkCollapsibleGroupBox, "CollapsibleGroupBoxExhaustive"
        )
        self.step_length_edit = self.get_ui(QLineEdit, "lineEditLength")
        self.nb_steps_edit = self.get_ui(QLineEdit, "lineEditSteps")
        self.opti_scale_edit = self.get_ui(QLineEdit, "lineEditScale")

        # :COMMENT: LBFGS2 parameters
        self.lbfgs2_box = self.get_ui(
            ctkCollapsibleGroupBox, "CollapsibleGroupBoxLBFGS2"
        )
        self.solution_accuracy_edit = self.get_ui(QLineEdit, "lineEditSolutionAccuracy")
        self.nb_iter_lbfgs2 = self.get_ui(QSpinBox, "spinBoxNbIterLBFGS2")
        self.delta_conv_tol_edit = self.get_ui(QLineEdit, "lineEditDeltaConv")

        # :COMMENT: Fill them combo boxes.
        self.metrics_combo_box.addItems(["Mean Squares", "Mattes Mutual Information"])
        self.optimizers_combo_box.addItems(["Gradient Descent", "Exhaustive", "LBFGS2"])
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

        self.optimizers_combo_box.currentIndexChanged.connect(self.update_registration)

        # :COMMENT: Initialize the registration.
        self.reset_registration()

    def reset_registration(self) -> None:
        """
        …
        """

        self.metrics_combo_box.setCurrentIndex(-1)
        self.optimizers_combo_box.setCurrentIndex(-1)
        self.interpolator_combo_box.setCurrentIndex(-1)
        self.sampling_strat_combo_box.setCurrentIndex(2)

        self.update_registration()

    def update_registration(self) -> None:
        """
        …
        """

        self.gradients_box.setEnabled(False)
        self.gradients_box.collapsed = 1
        self.exhaustive_box.setEnabled(False)
        self.exhaustive_box.collapsed = 1
        self.lbfgs2_box.setEnabled(False)
        self.lbfgs2_box.collapsed = 1

        if self.optimizers_combo_box.currentText == "Gradient Descent":
            self.gradients_box.setEnabled(True)
            self.gradients_box.collapsed = 0
        elif self.optimizers_combo_box.currentText == "Exhaustive":
            self.exhaustive_box.setEnabled(True)
            self.exhaustive_box.collapsed = 0
        elif self.optimizers_combo_box.currentText == "LBFGS2":
            self.lbfgs2_box.setEnabled(True)
            self.lbfgs2_box.collapsed = 0

    def register(self) -> None:
        """
        …
        """

        # :COMMENT: Ensure the parameters are set.
        if not self.input_volume:
            self.display_error_message("No input volume selected.")
            return
        if not self.target_volume:
            self.display_error_message("No target volume selected.")
            return
        if self.metrics_combo_box.currentIndex == -1:
            self.display_error_message("No metrics selected.")
            return
        if self.interpolator_combo_box.currentIndex == -1:
            self.display_error_message("No interpolator selected.")
            return
        if self.optimizers_combo_box.currentIndex == -1:
            self.display_error_message("No optimizer selected.")
            return

        # :COMMENT: VTK volume
        self.movingVolumeData = self.input_volume
        self.fixedVolumeData = self.target_volume

        # :COMMENT: utilitiy functions to get sitk images
        fixed_image = su.PullVolumeFromSlicer(self.fixedVolumeData)
        moving_image = su.PullVolumeFromSlicer(self.movingVolumeData)
        fixed_image = sitk.Cast(fixed_image, sitk.sitkFloat32)
        moving_image = sitk.Cast(moving_image, sitk.sitkFloat32)

        # :COMMENT: User settings retrieve
        bin_count = self.histogram_bin_count_spin_box.value
        # :COMMENT: Sampling strategies range from 0 to 2, they are enums (None, Regular, Random), thus index is sufficient
        sampling_strat = self.sampling_strat_combo_box.currentIndex
        sampling_perc = self.sampling_perc_spin_box.value

        # :COMMENT: settings for gradients only
        learning_rate = self.learning_rate_spin_box.value
        nb_iteration = self.nb_of_iter_spin_box.value
        convergence_min_val = self.conv_min_val_edit.text
        convergence_win_size = self.conv_win_size_spin_box.value

        # :COMMENT: settings for exhaustive only
        nb_of_steps = self.nb_steps_edit.text
        self.nb_of_steps = [int(step) for step in nb_of_steps.split(",")]
        self.step_length = self.step_length_edit.text
        if self.step_length == "pi":
            self.step_length = pi
        else:
            self.step_length = float(self.step_length)

        self.optimizer_scale = self.opti_scale_edit.text
        self.optimizer_scale = [int(scale) for scale in self.optimizer_scale.split(",")]

        # :COMMENT settings for LBFGS2
        solution_acc = self.solution_accuracy_edit.text
        nb_iter_lbfgs2 = self.nb_iter_lbfgs2.value
        delta_conv_tol = self.delta_conv_tol_edit.text

        # :DIRTY:Tony: For debug only (to be removed).
        # print(f"interpolator: {self.interpolator_combo_box.currentText}")
        # print(f"solution accuracy: {solution_acc}")
        # print(f"nb iter lbfgs2: {nb_iter_lbfgs2}")
        # print(f"delat conv tol: {delta_conv_tol}")

        # :BUG:Tony: Name of the new volume not applied.
        input = {}
        current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        input[
            "volume_name"
        ] = f"{self.input_volume.GetName()}_registered_{current_time}"
        input["histogram_bin_count"] = bin_count
        input["sampling_strategy"] = sampling_strat
        input["sampling_percentage"] = sampling_perc
        input["metrics"] = self.metrics_combo_box.currentText.replace(" ", "")
        input["interpolator"] = self.interpolator_combo_box.currentText.replace(" ", "")
        input["optimizer"] = self.optimizers_combo_box.currentText.replace(" ", "")
        input["learning_rate"] = learning_rate
        input["iterations"] = nb_iteration
        input["convergence_min_val"] = convergence_min_val
        input["convergence_win_size"] = convergence_win_size
        input["nb_of_steps"] = nb_of_steps
        input["step_length"] = self.step_length
        input["optimizer_scale"] = self.optimizer_scale
        input["solution_accuracy"] = solution_acc
        input["nb_iter_lbfgs2"] = nb_iter_lbfgs2
        input["delta_convergence_tolerance"] = delta_conv_tol

        # PARALLEL PROCESSING EXTENSION

        def on_registration_completed():
            """
            Handles the completion callback.
            """

            # :DIRTY:Tony: For debug only (to be removed).
            # print(logic.state())

            # :COMMENT: Log the registration.
            assert self.input_volume
            print(
                f'"{self.input_volume.GetName()}" has been registered as "{self.volumes[len(self.volumes) - 1].GetName()}".'
            )

            # :COMMENT: Reset the registration.
            self.reset_registration()

            # :COMMENT: Select the new volume to display it.
            self.choose_input_volume(len(self.volumes) - 1)

        logic = ProcessesLogic(completedCallback=lambda: on_registration_completed())
        if self.rigid_r_button.isChecked():
            scriptPath = self.resourcePath("Scripts/Registration/Rigid.py")
        else:
            scriptPath = self.resourcePath("Scripts/Registration/NonRigid.py")
        regProcess = RegistrationProcess(scriptPath, fixed_image, moving_image, input)
        logic.addProcess(regProcess)
        logic.run()

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

    def get_volume_by_name(self, name: str):
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


class CustomRegistrationTest(ScriptedLoadableModuleTest):
    """
    Test class for the Custom Registration module used to define the tests of the module.
    """

    def __init__(self):
        ScriptedLoadableModuleTest().__init__()

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

    def test_dummy(self):
        """
        Dummy test to check if the module works as expected.
        """

        print("Dummy test passed.")

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
    …

    Parameters:
        scriptPath: …
        fixedImageVolume …
        movingImageVolume: …
        input_parameters: …

    :TODO:Tony: Complete class documentation.
    """

    def __init__(
        self, scriptPath, fixedImageVolume, movingImageVolume, input_parameters
    ):
        Process.__init__(self, scriptPath)
        self.fixedImageVolume = fixedImageVolume
        self.movingImageVolume = movingImageVolume
        self.input_parameters = input_parameters

    def prepareProcessInput(self):
        """
        …

        :TODO:Tony: Complete method documentation.
        """

        input = {}
        input["fixed_image"] = self.fixedImageVolume
        input["moving_image"] = self.movingImageVolume
        input["parameters"] = self.input_parameters
        return pickle.dumps(input)

    def useProcessOutput(self, processOutput):
        """
        …

        Parameters:
            processOutput: …

        :TODO:Tony: Complete method documentation.
        """

        output = pickle.loads(processOutput)
        image_resampled = output["image_resampled"]
        pixelID = output["pixelID"]
        caster = sitk.CastImageFilter()
        caster.SetOutputPixelType(pixelID)
        image = caster.Execute(image_resampled)
        su.PushVolumeToSlicer(image)
