"""
The Custom Registration module for Slicer provides the features for 3D images registration, based on the ITK library.
"""

import datetime
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
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
)
from slicer import app, mrmlScene, util, vtkMRMLScalarVolumeNode, vtkMRMLScene
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

        # :COMMENT: Collapse all the collapsible buttons.
        collapsible_buttons = self.panel.findChildren(ctkCollapsibleButton)
        for collapsible_widget in collapsible_buttons:
            collapsible_widget.collapsed = True

        # :COMMENT: Set up the Pascal Mode Only.
        self.setup_pascal_only_mode()

        # :COMMENT: Set up the view interface.
        self.setup_view()

        # :COMMENT: Set up the preprocessing.
        self.setup_roi_selection()
        self.setup_cropping()
        self.setup_resampling()

        #:COMMENT: Set up the registration.
        self.setup_registration()

        # :COMMENT: Set up the selected/target volume architecture.
        self.setup_input_volume()
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

    def reset(self) -> None:
        """
        Resets all parameters to their default values.
        """

        self.reset_input_volume()
        self.reset_target_volume()

        self.reset_pascal_only_mode()
        self.reset_view()
        self.reset_roi_selection()
        self.reset_cropping()
        self.reset_resampling()
        self.reset_registration()

    def cleanup(self) -> None:
        """
        Cleans up the module from any observer.

        Called when reloading the module.
        """

        self.reset()

        for observer in self.scene_observers:
            mrmlScene.RemoveObserver(observer)

    #
    # PASCAL ONLY MODE
    #

    def setup_pascal_only_mode(self) -> None:
        """
        Sets up the Pascal Mode Only checkbox.
        """

        self.threeD_widget = app.layoutManager().threeDWidget(0)
        assert self.threeD_widget

        self.pascal_mode_checkbox = self.panel.findChild(
            QCheckBox, "PascalOnlyModeCheckBox"
        )
        assert self.pascal_mode_checkbox

        self.pascal_mode_checkbox.clicked.connect(self.manage_pascal_only_mode)

        # :COMMENT: Set the Pascal Only Mode to disabled by default.
        self.reset_pascal_only_mode()

    def reset_pascal_only_mode(self) -> None:
        """
        Resets the Pascal Only Mode to its default state: disabled by default.
        """

        self.pascal_mode_checkbox.setChecked(False)
        self.threeD_widget.setVisible(False)

    def manage_pascal_only_mode(self) -> None:
        """
        Displays or hides adequatly the 3D view depending on the Pascal Mode checkbox.
        """

        if self.pascal_mode_checkbox.isChecked():
            self.threeD_widget.setVisible(True)
        else:
            self.threeD_widget.setVisible(False)

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
        Updates the volume combo boxes and information labels (dimensions, spacing…).

        Parameters:
            variation: Either "input", "target", or "all".
        """

        # :TODO:Bastien: Factorize, if possible.

        # :COMMENT: Handle the "all" variation.
        if variation == "all":
            self.update_volume_combo_boxes_and_information_labels("input")
            self.update_volume_combo_boxes_and_information_labels("target")
            return

        # :COMMENT: Ensure the variation is valid.
        assert variation in ["input", "target"]

        # :COMMENT: Define the combo boxes.
        if variation == "input":
            volume_combo_boxes = [
                self.panel.findChild(ctkComboBox, "PreprocessingInputVolumeComboBox"),
                self.panel.findChild(ctkComboBox, "RegistrationInputVolumeComboBox"),
            ]
            volume_dimensions_labels = [
                self.panel.findChild(
                    QLabel, "PreprocessingInputVolumeDimensionsValueLabel"
                ),
                self.panel.findChild(
                    QLabel, "RegistrationInputVolumeDimensionsValueLabel"
                ),
            ]
            volume_spacing_labels = [
                self.panel.findChild(
                    QLabel, "PreprocessingInputVolumeSpacingValueLabel"
                ),
                self.panel.findChild(
                    QLabel, "RegistrationInputVolumeSpacingValueLabel"
                ),
            ]
        else:
            volume_combo_boxes = [
                self.panel.findChild(ctkComboBox, "PreprocessingTargetVolumeComboBox"),
                self.panel.findChild(ctkComboBox, "RegistrationTargetVolumeComboBox"),
            ]
            volume_dimensions_labels = [
                self.panel.findChild(
                    QLabel, "PreprocessingTargetVolumeDimensionsValueLabel"
                ),
                self.panel.findChild(
                    QLabel, "RegistrationTargetVolumeDimensionsValueLabel"
                ),
            ]
            volume_spacing_labels = [
                self.panel.findChild(
                    QLabel, "PreprocessingTargetVolumeSpacingValueLabel"
                ),
                self.panel.findChild(
                    QLabel, "RegistrationTargetVolumeSpacingValueLabel"
                ),
            ]

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

        for i in range(len(volume_combo_boxes)):
            # :COMMENT: Retrieve the UI items.
            volume_combo_box = volume_combo_boxes[i]
            volume_dimensions_label = volume_dimensions_labels[i]
            volume_spacing_label = volume_spacing_labels[i]

            # :COMMENT: Reset the volume combo box.
            fill_volume_combo_box(volume_combo_box)

            # :COMMENT: Set the combo box position.
            if variation == "input" and self.input_volume:
                volume_combo_box.setCurrentIndex(self.input_volume_index)
                volume_image_data = self.input_volume.GetImageData()

                volume_dimensions = volume_image_data.GetDimensions()
                volume_dimensions_label.setText(
                    "{} x {} x {}".format(
                        volume_dimensions[0],
                        volume_dimensions[1],
                        volume_dimensions[2],
                    )
                )

                volume_spacing = self.input_volume.GetSpacing()
                volume_spacing_label.setText(
                    "{:.1f} x {:.1f} x {:.1f}".format(
                        volume_spacing[0],
                        volume_spacing[1],
                        volume_spacing[2],
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

                volume_spacing = [self.target_volume.GetSpacing()[i] for i in range(3)]
                volume_spacing_label.setText(
                    "{:.1f} x {:.1f} x {:.1f}".format(
                        volume_spacing[0],
                        volume_spacing[1],
                        volume_spacing[2],
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
            assert self.slice_composite_nodes[i]

            self.slice_logic.append(
                app.layoutManager().sliceWidget(VIEWS[i]).sliceLogic()
            )
            assert self.slice_logic[i]

            # :COMMENT: Clear the 2D view.
            self.slice_composite_nodes[i].SetBackgroundVolumeID("")

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

        if self.input_volume:
            self.update_view(self.input_volume, 0, "Axial")
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

        # :COMMENT: Initialize the ROI.
        self.reset_roi_selection()

    def reset_roi_selection(self) -> None:
        """
        Resets the ROI selection.
        """

        # :COMMENT: Reset the ROI selection.
        self.remove_roi()

        # :COMMENT: Update the label accordingly.
        self.roi_selection_threshold_slider.setValue(0)
        self.roi_selection_threshold_value_label.setText("0")

    def remove_roi(self) -> None:
        """
        Removes the in-memory ROI.
        """

        self.input_volume_roi = None

    def select_roi(self) -> None:
        """
        Selects the ROI.
        """

        # :COMMENT: Ensure that a volume is selected.
        if not self.input_volume:
            self.display_error_message("Please select a volume to select a ROI from.")
            return

        # :COMMENT: Retrieve the threshold value.
        threshold = self.roi_selection_threshold_slider.value

        # :COMMENT: Call the ROI selection algorithm.
        self.input_volume_roi = self.logic.select_roi(
            self.vtk_to_sitk(self.input_volume), threshold
        )

        # :COMMENT: Log the ROI selection.
        print(
            f'ROI has been selected with a threshold value of {threshold} in "{self.input_volume.GetName()}".'
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

        # :COMMENT: Get the coordinates spinbox widgets.
        self.cropping_start = []
        self.cropping_end = []
        axis = ["x", "y", "z"]
        for i in range(len(axis)):
            self.cropping_start.append(self.panel.findChild(QSpinBox, "s" + axis[i]))
            self.cropping_end.append(self.panel.findChild(QSpinBox, "e" + axis[i]))

            # :COMMENT: Connect the spinbox widgets to their "on changed" function that displays the cropping preview.
            self.cropping_start[i].valueChanged.connect(self.preview_cropping)
            self.cropping_end[i].valueChanged.connect(self.preview_cropping)

        # :COMMENT: Initialize the cropping.
        self.reset_cropping()

    def reset_cropping(self) -> None:
        """
        Reset the cropping parameters.
        """

        # :COMMENT: Set all values to 0.
        for i in range(3):
            self.cropping_start[i].value = 0
            self.cropping_end[i].value = 0

        # :COMMENT: Reset the cropping preview.
        self.cropped_volume = None
        self.cropping_box = None

    def preview_cropping(self) -> None:
        """
        Generates a bounding box to preview the cropping.
        """

        # :DIRTY/TRICKY:Iantsa: Volume cropped each time a parameter is changed by user, even if the volume is not cropped in the end.

        # :COMMENT: Ensure that a volume is selected.
        if not self.input_volume:
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

        # :COMMENT: Save selected volume's data.
        data_backup = [
            self.input_volume.GetSpacing(),
            self.input_volume.GetOrigin(),
            vtk.vtkMatrix4x4(),
        ]
        self.input_volume.GetIJKToRASDirectionMatrix(data_backup[2])

        # :COMMENT: Convert the volume to a SimpleITK image.
        sitk_image = self.vtk_to_sitk(self.input_volume)

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

        # :COMMENT: Save the temporary cropped volume.
        self.cropped_volume = vtk_image

        # :COMMENT: Delete the previous cropping box from the scene if exists.
        if self.cropping_box:
            mrmlScene.RemoveNode(self.cropping_box)

        # :COMMENT: Create a new cropping box.
        # :DIRTY/GLITCH:Iantsa: Even if user never crops, the cropping box is still displayed.
        # :TODO:Iantsa: Create a checkbox to enable/disable displaying of the cropping box.
        self.cropping_box = mrmlScene.AddNewNodeByClass(
            "vtkMRMLMarkupsROINode", "Cropping Preview"
        )

        # :COMMENT: Display cropping box only in red view.
        self.cropping_box.GetDisplayNode().SetViewNodeIDs(["vtkMRMLSliceNodeRed"])

        # :COMMENT: Get the bounds of the volume.
        bounds = [0, 0, 0, 0, 0, 0]
        vtk_image.GetBounds(bounds)

        # :COMMENT: Calculate the center and radius of the volume.
        # :BUG:Iantsa: If the starting value is changed, the center is not updated properly.
        center = [(bounds[i] + bounds[i + 1]) / 2 for i in range(0, 5, 2)]
        radius = [size[i] / 2 for i in range(3)]

        # :COMMENT: Transform the center and radius according to the volume's orientation and spacing.
        transform_matrix = np.array(
            [[data_backup[2].GetElement(i, j) for j in range(3)] for i in range(3)]
        )

        transformed_center = np.array(center) + np.matmul(transform_matrix, start_val)
        transformed_radius = np.matmul(
            transform_matrix, np.array(data_backup[0]) * np.array(radius)
        )

        # :COMMENT: Set the center and radius of the cropping box to the transformed center and radius.
        self.cropping_box.SetXYZ(transformed_center)
        self.cropping_box.SetRadiusXYZ(transformed_radius)
        # :END_DIRTY/TRICKY:

    def crop(self) -> None:
        """
        Crops a volume using the selected algorithm.
        """

        # :COMMENT: Ensure that a volume is selected.
        if not self.input_volume:
            self.display_error_message("Please select a volume to crop.")
            return

        # :BUG:Iantsa: Not handled yet (can be non existent if crop button clicked without changing the default parameters)
        if not self.cropped_volume:  # and not self.cropping_box:
            return

        # :COMMENT: Add the VTK Volume Node to the scene.
        self.add_new_volume(self.cropped_volume, "cropped")

        # :COMMENT: Log the cropping.
        new_size = self.cropped_volume.GetImageData().GetDimensions()
        print(
            f'"{self.input_volume.GetName()}" has been cropped to size ({new_size[0]}x{new_size[1]}x{new_size[2]}) as "{self.cropped_volume.GetName()}".'
        )

        # :COMMENT: Select the cropped volume.
        self.choose_input_volume(self.volumes.GetNumberOfItems() - 1)

        # :COMMENT: Delete the cropping box (should exist if cropped_volume also exists)
        mrmlScene.RemoveNode(self.cropping_box)

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

        # :COMMENT: Initialize the resampling.
        self.reset_resampling()

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
        if not self.input_volume:
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
                self.vtk_to_sitk(self.input_volume),
                self.vtk_to_sitk(self.target_volume),
            )
        )

        # :COMMENT: Keep the original volume metadata.
        self.transfer_volume_metadata(self.input_volume, resampled_volume)

        # :COMMENT: Transfer the spacing.
        self.target_volume.SetSpacing(self.input_volume.GetSpacing())

        # :COMMENT: Save the resampled volume.
        self.add_new_volume(resampled_volume, "resampled")

        # :COMMENT: Log the resampling.
        print(
            f'"{self.input_volume.GetName()}" has been resampled to match "{self.target_volume.GetName()}" as "{resampled_volume.GetName()}".'
        )

        # :COMMENT: Select the resampled volume.
        self.choose_input_volume(self.volumes.GetNumberOfItems() - 1)

    #
    # REGISTRATION
    #

    def setup_registration(self) -> None:
        """
        Sets up the preprocessing widget by retrieving the volume selection widget and initializing it.
        """

        # :COMMENT: Link settings UI and code
        self.metrics_combo_box = self.panel.findChild(ctkComboBox, "ComboMetrics")
        self.interpolator_combo_box = self.panel.findChild(
            ctkComboBox, "comboBoxInterpolator"
        )
        self.optimizers_combo_box = self.panel.findChild(ctkComboBox, "ComboOptimizers")
        self.volume_name_edit = self.panel.findChild(QLineEdit, "lineEditNewVolumeName")
        self.histogram_bin_count_spin_box = self.panel.findChild(
            QSpinBox, "spinBoxBinCount"
        )
        self.sampling_strat_combo_box = self.panel.findChild(
            ctkComboBox, "comboBoxSamplingStrat"
        )
        self.sampling_perc_spin_box = self.panel.findChild(
            QDoubleSpinBox, "doubleSpinBoxSamplingPerc"
        )

        # :COMMENT: registration types
        self.non_rigid_r_button = self.panel.findChild(
            QRadioButton, "radioButtonNonRigid"
        )
        self.rigid_r_button = self.panel.findChild(QRadioButton, "radioButtonRigid")
        self.elastix_r_button = self.panel.findChild(QRadioButton, "radioButtonElastix")
        self.rigid_r_button.toggle()

        # :COMMENT: Gradients parameters
        self.gradients_box = self.panel.findChild(
            ctkCollapsibleGroupBox, "CollapsibleGroupBoxGradient"
        )
        self.learning_rate_spin_box = self.panel.findChild(
            QDoubleSpinBox, "doubleSpinBoxLearningR"
        )
        self.nb_of_iter_spin_box = self.panel.findChild(QSpinBox, "spinBoxNbIter")
        self.conv_min_val_edit = self.panel.findChild(QLineEdit, "lineEditConvMinVal")
        self.conv_win_size_spin_box = self.panel.findChild(
            QSpinBox, "spinBoxConvWinSize"
        )

        # :COMMENT: exhaustive parameters
        self.exhaustive_box = self.panel.findChild(
            ctkCollapsibleGroupBox, "CollapsibleGroupBoxExhaustive"
        )
        self.step_length_edit = self.panel.findChild(QLineEdit, "lineEditLength")
        self.nb_steps_edit = self.panel.findChild(QLineEdit, "lineEditSteps")
        self.opti_scale_edit = self.panel.findChild(QLineEdit, "lineEditScale")

        # :COMMENT: LBFGS2 parameters
        self.lbfgs2_box = self.panel.findChild(
            ctkCollapsibleGroupBox, "CollapsibleGroupBoxLBFGS2"
        )
        self.solution_accuracy_edit = self.panel.findChild(
            QLineEdit, "lineEditSolutionAccuracy"
        )
        self.nb_iter_lbfgs2 = self.panel.findChild(QSpinBox, "spinBoxNbIterLBFGS2")
        self.delta_conv_tol_edit = self.panel.findChild(QLineEdit, "lineEditDeltaConv")

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
        self.button_registration = self.panel.findChild(
            QPushButton, "PushButtonRegistration"
        )
        self.button_registration.clicked.connect(self.register)

        self.optimizers_combo_box.currentIndexChanged.connect(
            self.update_optimizer_parameters_group_box
        )

        # :COMMENT: Initialize the registration.
        self.reset_registration()

    def reset_registration(self) -> None:
        """
        Resets all the registration parameters to their default values.
        """

        self.metrics_combo_box.setCurrentIndex(-1)
        self.optimizers_combo_box.setCurrentIndex(-1)
        self.interpolator_combo_box.setCurrentIndex(-1)
        self.sampling_strat_combo_box.setCurrentIndex(2)

        self.update_optimizer_parameters_group_box()

    def update_optimizer_parameters_group_box(self) -> None:
        """
        Updates the optimizer parameters group box based on the chosen optimizer algorithm.
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

    def register(self):
        """
        Launches the registration process.
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
        input["volume_name"] = self.volume_name_edit.text
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
            # :DIRTY:Tony: Change the name to the one in the parameters (see BUG), and remove the assert.
            assert self.input_volume
            print(
                f'"{self.input_volume.GetName()}" has been registered as "{self.volumes.GetItemAsObject(self.volumes.GetNumberOfItems() - 1).GetName()}".'
            )

            # :COMMENT: Select the new volume to display it.
            self.choose_input_volume(self.volumes.GetNumberOfItems() - 1)

        logic = ProcessesLogic(completedCallback=lambda: on_registration_completed())
        if self.rigid_r_button.isChecked():
            scriptPath = self.resourcePath("Scripts/Registration/Rigid.py")
        else:
            scriptPath = self.resourcePath("Scripts/Registration/NonRigid.py")
        regProcess = RegistrationProcess(scriptPath, fixed_image, moving_image, input)
        logic.addProcess(regProcess)
        logic.run()

    #
    # INPUT VOLUME
    #

    def setup_input_volume(self) -> None:
        """
        Sets up the input volume architecture by initializing the data and retrieving the UI widgets.
        """

        def on_input_volume_combo_box_changed(
            index: int, combo_box: ctkComboBox
        ) -> None:
            """
            Handles change of input volume with options.

            Called when an item in an input volume combobox is selected.

            Parameters:
                index: The input volume index.
            """

            OPTIONS = ["Delete current volume…", "Rename current volume…"]

            # :COMMENT: Retrieve the selection text.
            name = combo_box.currentText

            # :COMMENT: Handle the different options.
            if name in OPTIONS:
                # :COMMENT: Ensure that there is at least one volume imported.
                if self.volumes.GetNumberOfItems() < 1:
                    self.update_volume_list()
                    self.display_error_message("No volumes imported.")
                    return

                # :COMMENT: Ensure that a volume is selected as input.
                if not self.input_volume:
                    self.update_volume_list()
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

        def on_preprocessing_input_volume_combo_box_changed(index: int) -> None:
            on_input_volume_combo_box_changed(
                index, self.preprocessing_input_volume_combo_box
            )

        def on_registration_input_volume_combo_box_changed(index: int) -> None:
            on_input_volume_combo_box_changed(
                index, self.registration_input_volume_combo_box
            )

        # :COMMENT: Initialize the input volume.
        self.input_volume = None
        self.input_volume_index = None

        # :COMMENT: Get and connection the preprocessing input volume combo box.
        self.preprocessing_input_volume_combo_box = self.panel.findChild(
            ctkComboBox, "PreprocessingInputVolumeComboBox"
        )
        assert self.preprocessing_input_volume_combo_box
        self.preprocessing_input_volume_combo_box.activated.connect(
            on_preprocessing_input_volume_combo_box_changed
        )

        # :COMMENT: Get and connection the registration input volume combo box.
        self.registration_input_volume_combo_box = self.panel.findChild(
            ctkComboBox, "RegistrationInputVolumeComboBox"
        )
        assert self.registration_input_volume_combo_box
        self.registration_input_volume_combo_box.activated.connect(
            on_registration_input_volume_combo_box_changed
        )

    def reset_input_volume(self) -> None:
        """
        Resets the input volume to None.
        """

        # :COMMENT: Reset the input volume.
        self.input_volume = None
        self.input_volume_index = None

        # :COMMENT: Remove the input volume in-memory ROI.
        self.remove_roi()

        # :COMMENT: Clear the view (top visualization).
        self.slice_composite_nodes[0].SetBackgroundVolumeID("")

    def choose_input_volume(self, index: int) -> None:
        """
        Selects an input volume.
        """

        # :COMMENT: Set the volume as input.
        self.input_volume_index = index
        self.input_volume = self.volumes.GetItemAsObject(index)
        assert self.input_volume
        self.update_volume_list()

        # :COMMENT: Update the cropping parameters accordingly.
        input_volume_image_data = self.input_volume.GetImageData()
        input_volume_dimensions = input_volume_image_data.GetDimensions()
        for i in range(3):
            self.cropping_start[i].setMaximum(input_volume_dimensions[i] - 1)
            self.cropping_end[i].setMaximum(input_volume_dimensions[i] - 1)

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
            self.update_volume_list()

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
        mrmlScene.RemoveNode(volume)
        print(f'"{volume.GetName()}" has been deleted.')

    #
    # TARGET VOLUME
    #

    def setup_target_volume(self) -> None:
        """
        Sets up the target volume architecture by initializing the data and retrieving the UI widgets.
        """

        def on_target_volume_combo_box_changed(
            index: int, combo_box: ctkComboBox
        ) -> None:
            """
            Handles change of target volume with options.

            Called when an item in an target volume combobox is selected.

            Parameters:
                index: The target volume index.
            """

            OPTIONS = ["Delete current volume…", "Rename current volume…"]

            # :COMMENT: Retrieve the selection text.
            name = combo_box.currentText

            # :COMMENT: Handle the different options.
            if name in OPTIONS:
                # :COMMENT: Ensure that there is at least one volume imported.
                if self.volumes.GetNumberOfItems() < 1:
                    self.update_volume_list()
                    self.display_error_message("No volumes imported.")
                    return

                # :COMMENT: Ensure that a volume is selected as target.
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

        def on_preprocessing_target_volume_combo_box_changed(index: int) -> None:
            on_target_volume_combo_box_changed(
                index, self.preprocessing_target_volume_combo_box
            )

        def on_registration_target_volume_combo_box_changed(index: int) -> None:
            on_target_volume_combo_box_changed(
                index, self.registration_target_volume_combo_box
            )

        # :COMMENT: Initialize the target volume.
        self.target_volume = None
        self.target_volume_index = None

        # :COMMENT: Get and connection the preprocessing target volume combo box.
        self.preprocessing_target_volume_combo_box = self.panel.findChild(
            ctkComboBox, "PreprocessingTargetVolumeComboBox"
        )
        assert self.preprocessing_target_volume_combo_box
        self.preprocessing_target_volume_combo_box.activated.connect(
            on_preprocessing_target_volume_combo_box_changed
        )

        # :COMMENT: Get and connection the registration target volume combo box.
        self.registration_target_volume_combo_box = self.panel.findChild(
            ctkComboBox, "RegistrationTargetVolumeComboBox"
        )
        assert self.registration_target_volume_combo_box
        self.registration_target_volume_combo_box.activated.connect(
            on_registration_target_volume_combo_box_changed
        )

    def reset_target_volume(self) -> None:
        """
        Resets the target volume to None.
        """

        # :COMMENT: Reset the target volume.
        self.target_volume = None
        self.target_volume_index = None

        # :COMMENT: Clear the view (top visualization).
        self.slice_composite_nodes[1].SetBackgroundVolumeID("")

    def choose_target_volume(self, index: int) -> None:
        """
        Selects an target volume.
        """

        # :COMMENT: Set the volume as target.
        self.target_volume_index = index
        self.target_volume = self.volumes.GetItemAsObject(index)
        assert self.target_volume
        self.update_volume_list()

        # :COMMENT: Update the cropping parameters accordingly.
        target_volume_image_data = self.target_volume.GetImageData()
        target_volume_dimensions = target_volume_image_data.GetDimensions()
        for i in range(3):
            self.cropping_start[i].setMaximum(target_volume_dimensions[i] - 1)
            self.cropping_end[i].setMaximum(target_volume_dimensions[i] - 1)

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
            self.update_volume_list()

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
        assert self.input_volume

        # :COMMENT: Generate and assign a unique name to the volume.
        current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        new_name = f"{self.input_volume.GetName()}_{name}_{current_time}"
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


class CustomRegistrationTest(ScriptedLoadableModuleTest):
    """
    Test class for the Custom Registration module used to define the tests of the module.
    """

    def __init__(self):
        ScriptedLoadableModuleTest().__init__()

    def runTest(self):
        """
        Runs all the tests in the Custom Registration module.
        """

        self.test_dummy()

    def test_dummy(self):
        """
        Dummy test to check if the module works as expected.
        """

        print("Dummy test passed.")


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
