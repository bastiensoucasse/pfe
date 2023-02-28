"""
The Custom Registration module for Slicer provides the features for 3D images registration, based on the ITK library.
"""

import datetime

import numpy as np
import SimpleITK as sitk
import vtk
from ctk import ctkCollapsibleButton, ctkComboBox
from qt import (
    QDialog,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
)
from slicer import app, mrmlScene, util, vtkMRMLScalarVolumeNode, vtkMRMLScene
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

        # :COMMENT: Set the module metadata.
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

        # :COMMENT: Apply threshold filter.
        binary_image = sitk.BinaryThreshold(
            image, lowerThreshold=threshold, upperThreshold=1000000
        )

        # :COMMENT: Apply connected component filter.
        label_map = sitk.ConnectedComponent(binary_image)

        # :COMMENT: Find largest connected component.
        label_shape_stats = sitk.LabelShapeStatisticsImageFilter()
        label_shape_stats.Execute(label_map)

        largest_label = 1
        max_volume = 0
        for label in range(1, label_shape_stats.GetNumberOfLabels() + 1):
            volume = label_shape_stats.GetPhysicalSize(label)
            if volume > max_volume:
                max_volume = volume
                largest_label = label

        # :COMMENT: Use binary image of largest connected component as ROI.
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

        # :COMMENT: Load the script file.
        with open(script_file, "r") as f:
            code = f.read()
            exec(code, globals())

        # :COMMENT: Retrieve the function.
        function = globals()[function_name]

        # :COMMENT: Run the function.
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

        # :COMMENT: Initialize the widget.
        ScriptedLoadableModuleWidget.setup(self)

        # :COMMENT: Initialize the logic.
        self.logic = CustomRegistrationLogic()
        assert self.logic

        # :COMMENT: Initialize the volume list.
        self.volumes = mrmlScene.GetNodesByClass("vtkMRMLScalarVolumeNode")

        # :COMMENT: Load the panel UI.
        # :GLITCH:Iantsa: 3D view reappearing when layout changed.
        self.panel = util.loadUI(self.resourcePath("UI/Panel.ui"))
        assert self.panel

        # :COMMENT: Insert the panel UI into the layout.
        self.layout.addWidget(self.panel)
        collapsible_buttons = self.panel.findChildren(ctkCollapsibleButton)
        for collapsible_widget in collapsible_buttons:
            collapsible_widget.collapsed = True

        # :COMMENT: Set up the view interface.
        self.setup_view()

        # :COMMENT: Set up the preprocessing.
        self.setup_roi_selection()
        self.setup_cropping()
        self.setup_resampling()

        # :COMMENT: Set up the selected/target volume architecture.
        self.setup_selected_volume()
        self.setup_target_volume()

        # :COMMENT: Add observer to update combobox when new volume is added to MRML Scene.
        self.scene_observers = [
            mrmlScene.AddObserver(vtkMRMLScene.NodeAddedEvent, self.update_volume_list),
            mrmlScene.AddObserver(
                vtkMRMLScene.NodeRemovedEvent, self.update_volume_list
            ),
            # :TODO:Bastien: Add observer for volume modified event.
        ]

        # :COMMENT: Launch the first volume list update.
        self.update_volume_list()

    def cleanup(self) -> None:
        """
        Cleans up the module from any observer.

        Called when reloading the module.
        """

        for observer in self.scene_observers:
            mrmlScene.RemoveObserver(observer)

    #
    # VOLUMES MANAGING
    #

    def update_volume_list(self, caller=None, event=None) -> None:
        """
        Updates the list of volumes in the volume combobox when a change is detected in the MRML Scene.

        Parameters:
            caller: The widget calling this method.
            event: The event that triggered this method.
        """

        # :COMMENT: Retrieve the volumes in the scene.
        self.volumes = mrmlScene.GetNodesByClass("vtkMRMLScalarVolumeNode")

        # :COMMENT: Update the volume combo boxes and information.
        self.update_volume_combo_boxes_and_information_labels("all")

        # :COMMENT: Reset the view.
        # :GLITCH:Bastien: Flash when loading volume.
        self.reset_view()

    def update_volume_combo_boxes_and_information_labels(
        self, variation: str = "all"
    ) -> None:
        """
        Updates the volume combo boxes and information labels (dimensions, pixel size, spacing…).

        Parameters:
            variation: Either "selected", "target", or "all".
        """

        # :TODO:Iantsa: Add pixel size and spacing to the information, along with the dimensions.

        # :COMMENT: Handle the "all" variation.
        if variation == "all":
            self.update_volume_combo_boxes_and_information_labels("selected")
            self.update_volume_combo_boxes_and_information_labels("target")
            return

        # :COMMENT: Ensure the variation is valid.
        assert variation in ["selected", "target"]

        # :COMMENT: Define the combo boxes.
        # :MERGE:Bastien: Add support for the second volume combo boxes (selected and target).
        if variation == "selected":
            volume_combo_box = self.selected_volume_combo_box
            volume_dimensions_label = self.panel.findChild(
                QLabel, "SelectedVolumeDimensionsValueLabel"
            )
        else:
            volume_combo_box = self.target_volume_combo_box
            volume_dimensions_label = self.panel.findChild(
                QLabel, "ResamplingTargetVolumeDimensionsValueLabel"
            )

        # :COMMENT: Define the combo box filling.
        def fill_volume_combo_box(volume_combo_box) -> None:
            """
            Fills the volume combo box with the available volumes and utility options.

            Parameters:
                volume_combo_box: The volume combo box to fill as an object.
            """

            # :COMMENT: Reset the volume combo box.
            volume_combo_box.clear()

            # :COMMENT: Add the available volumes to the combo box.
            for i in range(self.volumes.GetNumberOfItems()):
                volume = self.volumes.GetItemAsObject(i)
                volume_combo_box.addItem(volume.GetName())

            # :COMMENT: Add the utility options to the combo box.
            volume_combo_box.addItem("Rename current volume…")
            volume_combo_box.addItem("Delete current volume…")

        # :COMMENT: Reset the volume combo box.
        fill_volume_combo_box(volume_combo_box)

        # :COMMENT: Set the combo box position.
        if variation == "selected" and self.selected_volume:
            volume_combo_box.setCurrentIndex(self.selected_volume_index)
            volume_image_data = self.selected_volume.GetImageData()
            volume_dimensions = volume_image_data.GetDimensions()
            volume_dimensions_label.setText(
                "{} x {} x {}".format(
                    volume_dimensions[0],
                    volume_dimensions[1],
                    volume_dimensions[2],
                )
            )
        elif variation == "target" and self.target_volume:
            volume_combo_box.setCurrentIndex(self.target_volume_index)
            volume_image_data = self.target_volume.GetImageData()
            volume_dimensions = volume_image_data.GetDimensions()
            volume_dimensions_label.setText(
                "{} x {} x {}".format(
                    volume_dimensions[0],
                    volume_dimensions[1],
                    volume_dimensions[2],
                )
            )
        else:
            volume_combo_box.setCurrentIndex(-1)
            volume_dimensions_label.setText("…")

    #
    # VIEW INTERFACE
    #

    def setup_view(self) -> None:
        """
        Sets up the viewer interface by retrieving the 2D views widgets and clearing the 2D views.
        """

        VIEWS = ["Red", "Green", "Yellow"]

        # :COMMENT: List of vtkMRMLSliceCompositeNode objects that provide an interface for manipulating the properties of a slice composite node.
        self.slice_composite_nodes = []

        # :COMMENT: List of vtkMRMLSliceLogic objects that provide logic for manipulating a slice view of a volume.
        self.slice_logic = []

        # :COMMENT: Retrieve the objects for each view.
        for i in range(len(VIEWS)):
            self.slice_composite_nodes.append(
                mrmlScene.GetNodeByID("vtkMRMLSliceCompositeNode" + VIEWS[i])
            )
            self.slice_logic.append(
                app.layoutManager().sliceWidget(VIEWS[i]).sliceLogic()
            )

            # :COMMENT: Clear the 2D view.
            self.slice_composite_nodes[i].SetBackgroundVolumeID("")

        # :COMMENT: Hide the 3D view.
        threeDWidget = app.layoutManager().threeDWidget(0)
        threeDWidget.setVisible(False)

    def update_view(
        self, volume: vtkMRMLScalarVolumeNode, view_id: int, orientation: str
    ) -> None:
        """
        Updates a given 2D view with the selected volume.

        Parameters:
            volume: The selected volume.
            view_id: The 2D view ID.
            orientation: The orientation of the 2D view ("Axial", "Coronal", or "Sagittal").
        """

        # :COMMENT: Set to blank if no volume.
        if not volume:
            self.slice_composite_nodes[view_id].SetBackgroundVolumeID("")
            return

        # :COMMENT: Ensure the orientation is valid.
        assert orientation in ["Axial", "Coronal", "Sagittal"]

        # :COMMENT: Display the selected volume.
        self.slice_composite_nodes[view_id].SetBackgroundVolumeID(volume.GetID())

        # :COMMENT: Update the slice view.
        slice_node = self.slice_logic[view_id].GetSliceNode()
        if orientation == "Axial":
            slice_node.SetOrientationToAxial()
        if orientation == "Coronal":
            slice_node.SetOrientationToCoronal()
        if orientation == "Sagittal":
            slice_node.SetOrientationToSagittal()

        self.slice_logic[view_id].FitSliceToAll()

    def reset_view(self) -> None:
        """
        Sets all the view to their volume, or to blank if there is no volume assigned.
        """

        if self.selected_volume:
            self.update_view(self.selected_volume, 0, "Axial")
        else:
            self.update_view(None, 0, "Axial")

        if self.target_volume:
            self.update_view(self.target_volume, 1, "Axial")
        else:
            self.update_view(None, 1, "Axial")

        # :MERGE:Bastien: Add support for the difference map.
        self.update_view(None, 2, "Axial")

    #
    # ROI SELECTION
    #

    def setup_roi_selection(self) -> None:
        """
        Sets up the ROI selection widget.
        """

        def on_roi_selection_threshold_slider_value_changed():
            """
            Updates the ROI selection threshold value to match the slider.

            Called when the ROI selection threshold slider value is changed.
            """

            # :COMMENT: Retrieve the threshold value.
            threshold = self.roi_selection_threshold_slider.value

            # :COMMENT: Update the label accordingly.
            self.roi_selection_threshold_value_label.setText(str(threshold))

        # :COMMENT: Initialize the ROI.
        self.selected_volume_roi = None

        # :COMMENT: Get the ROI selection threshold slider.
        self.roi_selection_threshold_slider = self.panel.findChild(
            QSlider, "ROISelectionThresholdSlider"
        )
        assert self.roi_selection_threshold_slider

        # :COMMENT: Get the ROI selection threshold value label.
        self.roi_selection_threshold_value_label = self.panel.findChild(
            QLabel, "ROISelectionThresholdValueLabel"
        )
        assert self.roi_selection_threshold_value_label

        # :COMMENT: Connect the ROI selection threshold slider to its "on changed" function.
        self.roi_selection_threshold_slider.valueChanged.connect(
            on_roi_selection_threshold_slider_value_changed
        )

        # :COMMENT: Get the ROI selection button.
        self.roi_selection_button = self.panel.findChild(
            QPushButton, "ROISelectionButton"
        )
        assert self.roi_selection_button

        # :COMMENT: Connect the ROI selection button to the algorithm.
        self.roi_selection_button.clicked.connect(self.select_roi)

    def reset_roi_selection(self) -> None:
        """
        Resets the ROI selection.
        """

        # :COMMENT: Reset the ROI selection.
        if self.selected_volume_roi:
            self.selected_volume_roi = None
            print("ROI has been reset.")

        # :COMMENT: Update the label accordingly.
        self.roi_selection_threshold_slider.setValue(0)
        self.roi_selection_threshold_value_label.setText("0")

    def select_roi(self) -> None:
        """
        Selects the ROI.
        """

        # :COMMENT: Ensure that a volume is selected.
        if not self.selected_volume:
            self.display_error_message("Please select a volume to select a ROI from.")
            return

        # :COMMENT: Retrieve the threshold value.
        threshold = self.roi_selection_threshold_slider.value

        # :COMMENT: Call the ROI selection algorithm.
        self.selected_volume_roi = self.logic.select_roi(
            self.vtk_to_sitk(self.selected_volume), threshold
        )

        # :COMMENT: Log the ROI selection.
        print(
            f'ROI has been selected with a threshold value of {threshold} in "{self.selected_volume.GetName()}".'
        )

    #
    # CROPPING
    #

    def setup_cropping(self) -> None:
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

    def reset_cropping(self) -> None:
        """
        Reset the cropping parameters.
        """

        # :COMMENT: Set all values to 0.
        for i in range(3):
            self.cropping_start[i].value = 0
            self.cropping_end[i].value = 0

    def crop(self) -> None:
        """
        Crops a volume using the selected algorithm.
        """

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

        # :DIRTY/TRICKY:Iantsa: Brute test that only works for one case.
        # Create a new ROI markup node
        roiNode = mrmlScene.AddNewNodeByClass(
            "vtkMRMLMarkupsROINode", "Cropping Boundary Test"
        )

        # Get the bounds of the volume
        bounds = [0, 0, 0, 0, 0, 0]
        vtk_image.GetBounds(bounds)

        # Calculate the center and radius of the volume
        center = [(bounds[i] + bounds[i + 1]) / 2 for i in range(0, 5, 2)]
        radius = [size[i] / 2 for i in range(3)]

        print(
            "[DEBUG] Cropped volume dimensions:",
            vtk_image.GetImageData().GetDimensions(),
        )

        # Set the center of the ROI to the center of the volume
        roiNode.SetXYZ(center)

        rad = [radius[2] * 1.3, radius[0], radius[1]]

        roiNode.SetRadiusXYZ(rad)

        print("[DEBUG] Radius:", rad)

        roiDimensions = roiNode.GetSize()
        print(f"[DEBUG] ROI dimensions: {roiDimensions}")
        # :END_DIRTY/TRICKY:Iantsa:

        # :COMMENT: Log the cropping.
        new_size = vtk_image.GetImageData().GetDimensions()
        print(
            f'"{self.selected_volume.GetName()}" has been cropped to size ({new_size[0]}x{new_size[1]}x{new_size[2]}) as "{vtk_image.GetName()}".'
        )

    #
    # RESAMPLING
    #

    def setup_resampling(self) -> None:
        """
        Sets up the resampling widget by linking the UI to the scene and algorithm.
        """

        # :COMMENT: Get the resampling button.
        self.resampling_button = self.panel.findChild(QPushButton, "ResamplingButton")
        assert self.resampling_button

        # :COMMENT: Connect the resampling button to the algorithm.
        self.resampling_button.clicked.connect(self.resample)

    def reset_resampling(self) -> None:
        """
        Updates the list of available resampling targets in the resampling target volume combo box.
        """

        # :COMMENT: Nothing to do here.

    def resample(self) -> None:
        """
        Retrieves the selected and target volumes and runs the resampling algorithm.
        """

        # :COMMENT: Ensure that a volume is selected as well as a target volume.
        if not self.selected_volume:
            self.display_error_message("Please select a volume to resample.")
            return
        if not self.target_volume:
            self.display_error_message("Please choose a target volume.")
            return

        # :COMMENT: Call the resampling algorithm.
        resampled_volume = self.sitk_to_vtk(
            self.logic.run(
                self.resourcePath("Scripts/Resampling.py"),
                "resample",
                self.vtk_to_sitk(self.selected_volume),
                self.vtk_to_sitk(self.target_volume),
            )
        )

        # :COMMENT: Transfer the initial volume metadata.
        self.transfer_volume_metadata(self.selected_volume, resampled_volume)

        # :COMMENT: Save the resampled volume.
        self.add_new_volume(resampled_volume, "resampled")

        # :COMMENT: Log the resampling.
        print(
            f'"{self.selected_volume.GetName()}" has been resampled to match "{self.target_volume.GetName()}" as "{resampled_volume.GetName()}".'
        )

    #
    # SELECTED VOLUME
    #

    # :TODO:Bastien: Factorize selected/target volume architecture.

    def setup_selected_volume(self) -> None:
        """
        Sets up the selected volume architecture by initializing the data and retrieving the UI widgets.
        """

        # :COMMENT: Define the changed event.
        def on_selected_volume_combo_box_changed(index: int) -> None:
            """
            Handles change of selected volume with options.

            Called when an item in a selected volume combobox is selected.

            Parameters:
                index: The selected volume index.
            """

            OPTIONS = ["Delete current volume…", "Rename current volume…"]

            # :COMMENT: Retrieve the selection text.
            name = self.selected_volume_combo_box.currentText

            # :COMMENT: Handle the different options.
            if name in OPTIONS:
                # :COMMENT: Ensure that there is at least one volume imported.
                if self.volumes.GetNumberOfItems() < 1:
                    self.update_volume_list()
                    self.display_error_message("No volumes imported.")
                    return

                # :COMMENT: Ensure that a volume is selected.
                if not self.selected_volume:
                    self.update_volume_list()
                    self.display_error_message("Please select a volume first.")
                    return

                if name == "Rename current volume…":
                    self.rename_selected_volume()
                    return

                if name == "Delete current volume…":
                    self.delete_selected_volume()
                    return

            # :COMMENT: Select the volume at specified index otherwise.
            self.choose_selected_volume(index)

        # :COMMENT: Initialize the selected volume.
        self.selected_volume = None
        self.selected_volume_index = None

        # :COMMENT: Get the selected volume combo box.
        self.selected_volume_combo_box = self.panel.findChild(
            ctkComboBox, "SelectedVolumeComboBox"
        )
        assert self.selected_volume_combo_box
        self.selected_volume_combo_box.activated.connect(
            on_selected_volume_combo_box_changed
        )

        # :MERGE:Bastien: Add support for the second selected volume combo boxes.

    def reset_selected_volume(self) -> None:
        """
        Resets the selected volume to None.
        """

        # :COMMENT: Reset the selected volume.
        self.selected_volume = None
        self.selected_volume_index = None

        # :COMMENT: Clear the view (top visualization).
        self.slice_composite_nodes[0].SetBackgroundVolumeID("")

    def choose_selected_volume(self, index: int) -> None:
        """
        Selects a volume.
        """

        # :COMMENT: Set the volume as selected.
        self.selected_volume_index = index
        self.selected_volume = self.volumes.GetItemAsObject(index)
        assert self.selected_volume
        self.update_volume_list()

        # :COMMENT: Update the cropping parameters accordingly.
        selected_volume_image_data = self.selected_volume.GetImageData()
        selected_volume_dimensions = selected_volume_image_data.GetDimensions()
        for i in range(3):
            self.cropping_start[i].setMaximum(selected_volume_dimensions[i] - 1)
            self.cropping_end[i].setMaximum(selected_volume_dimensions[i] - 1)

        # :COMMENT: Reset the module data.
        self.reset_roi_selection()
        self.reset_cropping()
        self.reset_resampling()

        # :COMMENT: Update the view (upper visualization).
        self.update_view(self.selected_volume, 0, "Axial")

    def rename_selected_volume(self) -> None:
        """
        Loads the renaming feature with a minimal window.
        """

        # :COMMENT: Define the handler.
        def rename_volume_handler(result) -> None:
            """
            Applies the renaming of the selected volume.

            Parameters:
                result: The result of the input dialog.
            """

            if result == QDialog.Accepted:
                # :COMMENT: Ensure that a volume is selected.
                assert self.selected_volume

                # :COMMENT: Retrieve the new name and apply it.
                new_name = self.renaming_input_dialog.textValue()
                self.selected_volume.SetName(new_name)

                # :COMMENT: Log the renaming.
                print(
                    f'"{self.renaming_old_name}" has been renamed to "{self.selected_volume.GetName()}".'
                )

                # :DIRTY:Bastien: Find a way to add modified node event observer in the setup.
                self.update_volume_list()

        # :COMMENT: Ensure that a volume is selected.
        assert self.selected_volume

        # :COMMENT: Save the old name for logging.
        self.renaming_old_name = self.selected_volume.GetName()

        # :COMMENT: Open an input dialog for the new name.
        self.renaming_input_dialog = QInputDialog(None)
        self.renaming_input_dialog.setWindowTitle("Rename Volume")
        self.renaming_input_dialog.setLabelText("Enter the new name:")
        self.renaming_input_dialog.setModal(True)
        self.renaming_input_dialog.setTextValue(self.selected_volume.GetName())
        self.renaming_input_dialog.finished.connect(rename_volume_handler)
        self.renaming_input_dialog.show()

    def delete_selected_volume(self) -> None:
        """
        Deletes the current selected volume.
        """

        assert self.selected_volume
        volume = self.selected_volume
        if self.selected_volume_index == self.target_volume_index:
            self.reset_target_volume()
        self.reset_selected_volume()
        mrmlScene.RemoveNode(volume)
        print(f'"{volume.GetName()}" has been deleted.')

    #
    # TARGET VOLUME
    #

    def setup_target_volume(self) -> None:
        """
        Sets up the target volume architecture by initializing the data and retrieving the UI widgets.
        """

        # :COMMENT: Define the changed event.
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
                if self.volumes.GetNumberOfItems() < 1:
                    self.update_volume_list()
                    self.display_error_message("No volumes imported.")
                    return

                # :COMMENT: Ensure that a volume is target.
                if not self.target_volume:
                    self.update_volume_list()
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

        # :COMMENT: Initialize the target volume.
        self.target_volume = None
        self.target_volume_index = None

        # :COMMENT: Get the target volume combo box.
        self.target_volume_combo_box = self.panel.findChild(
            ctkComboBox, "ResamplingTargetVolumeComboBox"
        )
        assert self.target_volume_combo_box
        self.target_volume_combo_box.activated.connect(
            on_target_volume_combo_box_changed
        )

        # :MERGE:Bastien: Add support for the second target volume combo boxes.

    def reset_target_volume(self) -> None:
        """
        Resets the target volume to None.
        """

        # :COMMENT: Reset the target volume.
        self.target_volume = None
        self.target_volume_index = None

        # :COMMENT: Clear the view (bottom left visualization).
        self.slice_composite_nodes[1].SetBackgroundVolumeID("")

    def choose_target_volume(self, index: int) -> None:
        """
        Selects a volume.
        """

        # :COMMENT: Set the volume as target.
        self.target_volume_index = index
        self.target_volume = self.volumes.GetItemAsObject(index)
        assert self.target_volume
        self.update_volume_list()

        # :COMMENT: Update the view (bottom left visualization).
        self.update_view(self.target_volume, 1, "Axial")

    def rename_target_volume(self) -> None:
        """
        Loads the renaming feature with a minimal window.
        """

        # :COMMENT: Define the handler.
        def rename_volume_handler(result) -> None:
            """
            Applies the renaming of the target volume.

            Parameters:
                result: The result of the input dialog.
            """

            if result == QDialog.Accepted:
                # :COMMENT: Ensure that a volume is target.
                assert self.target_volume

                # :COMMENT: Retrieve the new name and apply it.
                new_name = self.renaming_input_dialog.textValue()
                self.target_volume.SetName(new_name)

                # :COMMENT: Log the renaming.
                print(
                    f'"{self.renaming_old_name}" has been renamed to "{self.target_volume.GetName()}".'
                )

                # :DIRTY:Bastien: Find a way to add modified node event observer in the setup.
                self.update_volume_list()

        # :COMMENT: Ensure that a volume is target.
        assert self.target_volume

        # :COMMENT: Save the old name for logging.
        self.renaming_old_name = self.target_volume.GetName()

        # :COMMENT: Open an input dialog for the new name.
        self.renaming_input_dialog = QInputDialog(None)
        self.renaming_input_dialog.setWindowTitle("Rename Volume")
        self.renaming_input_dialog.setLabelText("Enter the new name:")
        self.renaming_input_dialog.setModal(True)
        self.renaming_input_dialog.setTextValue(self.target_volume.GetName())
        self.renaming_input_dialog.finished.connect(rename_volume_handler)
        self.renaming_input_dialog.show()

    def delete_target_volume(self) -> None:
        """
        Deletes the current target volume.
        """

        assert self.target_volume
        volume = self.target_volume
        if self.target_volume_index == self.selected_volume_index:
            self.reset_selected_volume()
        self.reset_target_volume()
        mrmlScene.RemoveNode(volume)
        print(f'"{volume.GetName()}" has been deleted.')

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
        for i in range(self.volumes.GetNumberOfItems()):
            volume = self.volumes.GetItemAsObject(i)
            if volume.GetName() == name:
                return volume

        # :COMMENT: Return None if the volume was not found.
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

        # :COMMENT: Retrieve the metadata from the source volume.
        spacing = source_volume.GetSpacing()
        origin = source_volume.GetOrigin()
        ijk_to_ras_direction_matrix = vtk.vtkMatrix4x4()
        source_volume.GetIJKToRASDirectionMatrix(ijk_to_ras_direction_matrix)

        # :COMMENT: Apply the metadata to the target volume.
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
