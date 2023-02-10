"""
The Custom Registration module for Slicer provides the features for 3D images registration, based on the ITK library.
"""
import datetime

import numpy as np
import SimpleITK as sitk
import vtk
from qt import QComboBox, QInputDialog, QLabel, QMessageBox, QPushButton, QSpinBox, QDialog, QLineEdit
from slicer import mrmlScene, util, vtkMRMLScalarVolumeNode
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleWidget,
)


class CustomRegistration(ScriptedLoadableModule):
    """
    Main class for the Custom Registration module used to define the module's metadata.
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.set_metadata()

    def set_metadata(self):
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
        self.parent.helpText = "The Custom Registration module for Slicer provides the features for 3D images registration, based on the ITK library."
        self.parent.acknowledgementText = (
            "This project is supported by Fabien Baldacci."
        )


class CustomRegistrationLogic(ScriptedLoadableModuleLogic):
    """
    Logic class for the Custom Registration module used to define the module's algorithms.
    """

    def __init__(self):
        ScriptedLoadableModuleLogic.__init__(self)


class CustomRegistrationWidget(ScriptedLoadableModuleWidget):
    """
    Widget class for the Custom Registration module used to define the module's panel interface.
    """

    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)

    def setup(self):
        """
        Sets up the widget for the module by adding a welcome message to the layout.
        """

        # :COMMENT: Initialize setup from Slicer.
        ScriptedLoadableModuleWidget.setup(self)

        # :COMMENT: Load UI file.
        self.panel_ui = util.loadUI(self.resourcePath("UI/Panel.ui"))
        self.layout.addWidget(self.panel_ui)

        self.crop_setup()

    def crop_setup(self):
        # :COMMENT: Get the crop button widget.
        self.crop_button = self.panel_ui.findChild(QPushButton, "crop_button")
        self.crop_button.clicked.connect(self.crop)

        # :COMMENT: Get the volume combobox widget.
        self.volumeComboBox = self.panel_ui.findChild(QComboBox, "volume")
        self.volumeComboBox.activated.connect(self.onVolumeActivated)

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
        # :BUG: List not updated when new volume loaded.
        # self.observerTag = mrmlScene.AddObserver(slicer.vtkMRMLScene.NodeAddedEvent, self.updateCombobox)

        # :COMMENT: Get the coordinates entered by user in spinbox widgets.
        self.start = []
        self.end = []
        for i in ['x', 'y', 'z']:
            self.start.append(self.panel_ui.findChild(QSpinBox, "s" + i))
            self.end.append(self.panel_ui.findChild(QSpinBox, "e" + i))

    def onVolumeActivated(self, index):
        options = ["Select a Volume", "Delete current Volume", "Rename current Volume"]

        name = self.volumeComboBox.currentText
        if name in options:
            # :TODO: Add renaming and deleting features.
            if name == "Delete current Volume":
                print("[DEBUG]", name, "not implemented yet!\n")

            if name == "Rename current Volume":
                print("[DEBUG]", name, "not implemented yet!\n")
                # self.renameVolume()

            self.currentVolumeIndex = -1

        else:
            self.currentVolumeIndex = index
            self.currentVolume = self.volumes.GetItemAsObject(index - 1)
            volume_data = self.currentVolume.GetImageData()
            self.currentVolumeDim = volume_data.GetDimensions()
            print("[DEBUG] Volume Dimensions:", self.currentVolumeDim)
            self.update_volume_dimensions()

            for i in range(3):
                self.start[i].setMaximum(self.currentVolumeDim[i])
                self.end[i].setMaximum(self.currentVolumeDim[i])

    def crop(self):
        # :COMMENT: Do not do anything if crop button clicked but no volume selected.
        # :TODO: Solve, knowing that it works for the "Select a Volume" case.
        if self.currentVolumeIndex < 0:
            return

        print("[DEBUG] Current volume name:", self.currentVolume.GetName())
        print("[DEBUG] Current volume index:", self.currentVolumeIndex)

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

        # :COMMENT: Get the selected volume and convert it to a SimpleITK image.
        selected_volume = self.currentVolume
        # print("[DEBUG]", type(selected_volume))
        vtk_image = selected_volume.GetImageData()
        np_array = vtk.util.numpy_support.vtk_to_numpy(
            vtk_image.GetPointData().GetScalars()
        )
        np_array = np.reshape(np_array, (vtk_image.GetDimensions()[::-1]))
        # print("[DEBUG]", type(np_array))
        sitk_image = sitk.GetImageFromArray(np_array)
        # print("[DEBUG]", type(sitk_image))
        print("[DEBUG] Size:", sitk_image.GetSize())

        # :COMMENT: Get the size of the crop region.
        size = [end_val[i] - start_val[i] + 1 for i in range(3)]

        # :COMMENT: Crop the image.
        extract_filter = sitk.ExtractImageFilter()
        extract_filter.SetSize(size)
        extract_filter.SetIndex(start_val)
        cropped_image = extract_filter.Execute(sitk_image)
        # print("[DEBUG]", type(cropped_image))
        print("[DEBUG] Size:", cropped_image.GetSize())

        # :COMMENT: Create a new volume node
        new_volume_node = vtkMRMLScalarVolumeNode()
        current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        new_name = f"{selected_volume.GetName()}_cropped_{current_time}"
        new_volume_node.SetName(new_name)

        # :COMMENT: Convert the cropped SimpleITK image back to a NumPy array
        np_array = sitk.GetArrayFromImage(cropped_image)

        # :COMMENT: Convert the numpy array to a vtkImageData object
        vtk_image = vtk.vtkImageData()
        vtk_image.SetDimensions(np_array.shape[::-1])
        vtk_image.AllocateScalars(vtk.VTK_FLOAT, 1)
        vtk_array = vtk.util.numpy_support.numpy_to_vtk(np_array.flatten())
        vtk_image.GetPointData().SetScalars(vtk_array)

        # :COMMENT: Update the MRML scene
        mrmlScene.AddNode(new_volume_node)
        new_volume_node.SetAndObserveImageData(vtk_image)

        # :TRICKY: Refresh the volume combobox but should be handled by the observer.
        # self.volumeComboBox.insertItem(self.currentVolumeIndex + 1, new_volume_node.GetName())
        # self.volumeComboBox.setCurrentIndex(self.currentVolumeIndex + 1)

        # :COMMENT: Create a vtkMRMLScalarVolumeNode object
        # volume_node = vtkMRMLScalarVolumeNode()
        # volume_node.SetAndObserveImageData(vtk_image)
        # print("[DEBUG]", type(volume_node))

        # sitk.WriteImage(cropped_image, '/Users/iantsaprovost/Desktop/test.nrrd')

    # :BUG: Doesn't update the list of available volumes.
    # def updateCombobox(self, caller, event):
    #     volumes = mrmlScene.GetNodesByClass("vtkMRMLNode")
    #     volumesNames = []
    #     for i in range(volumes.GetNumberOfItems()):
    #         volume = volumes.GetItemAsObject(i)
    #         if volume.GetName() != "":
    #             volumesNames.append(volume.GetName())
    #     self.comboBox.clear()
    #     self.comboBox.addItems(volumesNames)

    def update_volume_dimensions(self):
        dim_label = self.panel_ui.findChild(QLabel, "dim_label")
        dim_label.setText(
            "{} x {} x {}".format(
                self.currentVolumeDim[0],
                self.currentVolumeDim[1],
                self.currentVolumeDim[2],
            )
        )

    def error_message(self, message):
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