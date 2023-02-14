"""
The Custom Registration module for Slicer provides the features for 3D images registration, based on the ITK library.
"""

import datetime

import numpy as np
import SimpleITK as sitk
import vtk
from qt import (  # QDialog,; QInputDialog,; QLineEdit,
    QComboBox,
    QLabel,
    QMessageBox,
    QPushButton,
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

    def cropping_algorithm(self, volume, start_val, size):
        """
        Crops a volume using the selected algorithm.

        Parameters:
            volume: The SimpleITK volume to be cropped.
            start_val: The start index of the cropping region.
            size: The size of the cropping region.

        Returns:
            The cropped SimpleITK image.
        """

        crop_filter = sitk.ExtractImageFilter()
        crop_filter.SetSize(size)
        crop_filter.SetIndex(start_val)
        cropped_image = crop_filter.Execute(volume)
        return cropped_image

    # :IDK: Could change class.
    def cleanup(self):
        """
        Cleans up the module.
        """

        # Remove the observer.
        mrmlScene.RemoveObserver(self.observerTag)


# :TODO: Homogenise the names of variables (image, node, volume)
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

        # Load the UI.
        self.panel_ui = util.loadUI(self.resourcePath("UI/Panel.ui"))
        self.layout.addWidget(self.panel_ui)

        # Setup the preprocessing.
        self.preprocessing_setup()
        self.cropping_setup()
        self.resampling_setup()

    #
    # PREPROCESSING
    #

    def preprocessing_setup(self):
        """
        Sets up the preprocessing widget by retrieving the volume selection widget and initializing it.
        """

        # :COMMENT: Get the volume combobox widget.
        self.volumeComboBox = self.panel_ui.findChild(QComboBox, "volume")
        self.volumeComboBox.activated.connect(self.on_volume_activated)

        # :COMMENT: Add the available volumes and options to the combobox.
        self.volumes = mrmlScene.GetNodesByClass("vtkMRMLScalarVolumeNode")

        # :BUG: Supposed to be a placeholder, but still appears in list (shouldn't)
        self.volumeComboBox.insertItem(0, "Select a Volume")
        self.currentVolumeIndex = -1

        for i in range(self.volumes.GetNumberOfItems()):
            volume = self.volumes.GetItemAsObject(i)
            self.volumeComboBox.addItem(volume.GetName())

        self.volumeComboBox.addItem("Rename current Volume")
        self.volumeComboBox.addItem("Delete current Volume")

        # :COMMENT: Add observer to update combobox when new volume is added to MRML Scene.
        self.observerTag = mrmlScene.AddObserver(
            vtkMRMLScene.NodeAddedEvent, self.update_volume_list
        )

    #
    # :TODO: Sort into CROPPING and TOOLS sections.
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

        options = ["Select a Volume", "Delete current Volume", "Rename current Volume"]

        name = self.volumeComboBox.currentText
        if name in options:
            # :TODO: Add renaming and deleting features.
            if name == "Delete current Volume":
                print("[DEBUG]", name, "option not implemented yet!\n")

                # print(
                #     f"\"{self.currentVolume.GetName()}\" has been deleted."
                # )

            if name == "Rename current Volume":
                print("[DEBUG]", name, "option not implemented yet!\n")

                # old_name = self.currentVolume.GetName()
                # # ...
                # print(
                #     f"\"{old_name}\" has been renamed as {self.currentVolume.GetName()}."
                # )

            self.currentVolumeIndex = -1

        else:
            self.currentVolumeIndex = index
            # :DIRTY: -1 because "Select a Volume" is the first item in the combobox: needs to be fixed so indices are equal.
            self.currentVolume = self.volumes.GetItemAsObject(index - 1)
            volume_data = self.currentVolume.GetImageData()
            self.currentVolumeDim = volume_data.GetDimensions()
            self.update_dimensions_display()

            for i in range(3):
                self.start[i].setMaximum(self.currentVolumeDim[i])
                self.end[i].setMaximum(self.currentVolumeDim[i])

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
        cropped_image = self.logic.cropping_algorithm(sitk_image, start_val, size)

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

        dim_label = self.panel_ui.findChild(QLabel, "dim_label")
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

        # Retrieve the resampling target image combo box and connect it to its on changed event.
        self.resampling_target_image_combo_box = self.panel_ui.findChild(
            QComboBox, "ResamplingTargetImageComboBox"
        )
        self.resampling_target_image_combo_box.activated.connect(
            self.on_resampling_target_image_changed
        )

        # Add the available images to the resampling target image combo box.
        for i in range(self.volumes.GetNumberOfItems()):
            image_name = self.volumes.GetItemAsObject(i).GetName()
            self.resampling_target_image_combo_box.addItem(image_name)

        # Link the resample button to the algorithm.
        self.resample_button = self.panel_ui.findChild(QPushButton, "ResampleButton")
        self.resample_button.clicked.connect(self.resample)

    def on_resampling_target_image_changed(self):
        """
        Updates the resampling target image in memory.

        Called when the resampling target image combo box is changed.
        """

        # Retrieve the resampling target image name in the combo box.
        resampling_target_image_name = (
            self.resampling_target_image_combo_box.currentText
        )

        # Update the resampling target image.
        self.resampling_target_image = self.get_image_by_name(
            resampling_target_image_name
        )

        # Log the resampling target image change.
        print(f'Resampling target image: "{resampling_target_image_name}."')

    def resample(self):
        """
        Retrieves the selected and target images and runs the resampling algorithm.
        """

        # Retrieve the preprocessing selected and target images.
        selected_image = self.currentVolume
        target_image = self.resampling_target_image

        if selected_image is None or target_image is None:
            self.error_message("Please select an image to process and a target image.")
            return

        # Log the resampling running.
        print(
            f'Resampling "{selected_image.GetName()}" to match "{target_image.GetName()}"…'
        )

        # Call the resampling algorithm.
        resampled_image = self.sitk_to_vtk(
            self.logic.run(
                self.resourcePath("Scripts/Resampling.py"),
                "resample",
                self.vtk_to_sitk(selected_image),
                self.vtk_to_sitk(target_image),
            )
        )

        # Save the resampled image.
        self.add_new_volume(resampled_image, "resampled")

        # Log the resampling done.
        print(
            f'"{selected_image.GetName()}" has been resampled to match "{target_image.GetName()}" as "{resampled_image.GetName()}".'
        )

    #
    # UTILITIES
    #

    def get_image_by_name(self, name: str):
        """
        Retrieves the image by its name.

        Parameters:
            name: The name of the image to retrieve.

        Returns:
            The VTK image.
        """

        # Search for the image by its name.
        for i in range(self.volumes.GetNumberOfItems()):
            image = self.volumes.GetItemAsObject(i)
            if image.GetName() == name:
                return image

        # Return None if the image was not found.
        return None
