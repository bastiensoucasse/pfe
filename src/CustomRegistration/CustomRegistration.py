"""
The Custom Registration module for Slicer provides the features for 3D images registration, based on the ITK library.
"""
import SimpleITK as sitk
import vtk
import numpy as np
from qt import QPushButton, QSpinBox, QComboBox
from slicer import util, mrmlScene, vtkMRMLScalarVolumeNode
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

        for i in range(self.volumes.GetNumberOfItems()):
            volume = self.volumes.GetItemAsObject(i)
            self.volumeComboBox.addItem(volume.GetName())

        self.volumeComboBox.addItem("Rename current Volume")
        self.volumeComboBox.addItem("Delete current Volume")

        # :COMMENT: Add observer to update combobox when new volume is added to MRML Scene.
        # :BUG: List not updated when new volume loaded.
        # self.observerTag = mrmlScene.AddObserver(slicer.vtkMRMLScene.NodeAddedEvent, self.updateCombobox)

        # :COMMENT: Get the coordinates spinbox widgets.
        # :DIRTY: To be factorized
        self.sx = self.panel_ui.findChild(QSpinBox, "sx")
        self.sy = self.panel_ui.findChild(QSpinBox, "sy")
        self.sz = self.panel_ui.findChild(QSpinBox, "sz")
        self.ex = self.panel_ui.findChild(QSpinBox, "ex")
        self.ey = self.panel_ui.findChild(QSpinBox, "ey")
        self.ez = self.panel_ui.findChild(QSpinBox, "ez")

    def onVolumeActivated(self, index):
        options = ["Select a Volume", "Delete current Volume", "Rename current Volume"]

        name = self.volumeComboBox.currentText
        if name in options:
            # :TODO: Add renaming and deleting features.
            print("[DEBUG]", name, "not implemented yet!\n")

        else:
            self.currentVolumeIndex = index
            self.currentVolume = self.volumes.GetItemAsObject(index-1)
            volumeData = self.currentVolume.GetImageData()
            volumeDim = volumeData.GetDimensions()
            print("[DEBUG] Volume Dimensions:", volumeDim)
            # :DIRTY: To be factorized
            self.sx.setMaximum(volumeDim[0])
            self.sy.setMaximum(volumeDim[1])
            self.sz.setMaximum(volumeDim[2])
            self.ex.setMaximum(volumeDim[0])
            self.ey.setMaximum(volumeDim[1])
            self.ez.setMaximum(volumeDim[2])

    def crop(self):
        print("[DEBUG] Current volume name:", self.currentVolume.GetName())
        print("[DEBUG] Current volume index:", self.currentVolumeIndex)

        # :COMMENT: Retrieve coordinates input.
        # :DIRTY: To be factorized.
        start_x = self.sx.value
        start_y = self.sy.value
        start_z = self.sz.value
        end_x = self.ex.value
        end_y = self.ey.value
        end_z = self.ez.value
        print("[DEBUG] Start X:", start_x)
        print("[DEBUG] Start Y:", start_y)
        print("[DEBUG] Start Z:", start_z)
        print("[DEBUG] End X:", end_x)
        print("[DEBUG] End Y:", end_y)
        print("[DEBUG] End 2:", end_z, "\n")

        # :TODO: Add checking of coordinates (start must be < end)

        # :COMMENT: Get the selected volume and convert it to a SimpleITK image.
        selected_volume = self.currentVolume
        # print("[DEBUG]", type(selected_volume))
        vtk_image = selected_volume.GetImageData()
        np_array = vtk.util.numpy_support.vtk_to_numpy(vtk_image.GetPointData().GetScalars())
        np_array = np.reshape(np_array, (vtk_image.GetDimensions()[::-1]))
        # print("[DEBUG]", type(np_array))
        sitk_image = sitk.GetImageFromArray(np_array)
        # print("[DEBUG]", type(sitk_image))
        print("[DEBUG] Size:", sitk_image.GetSize())

        # :COMMENT: Get the size of the crop region
        start = [self.sx.value, self.sy.value, self.sz.value]
        end = [self.ex.value, self.ey.value, self.ez.value]
        size = [end[0]-start[0]+1, end[1]-start[1]+1, end[2]-start[2]+1]

        # :COMMENT: Crop the image
        extract_filter = sitk.ExtractImageFilter()
        extract_filter.SetSize(size)
        extract_filter.SetIndex(start)
        cropped_image = extract_filter.Execute(sitk_image)
        # print("[DEBUG]", type(cropped_image))
        print("[DEBUG] Size:", cropped_image.GetSize())

        # :COMMENT: Convert the cropped SimpleITK image back to a NumPy array
        # np_array = sitk.GetArrayFromImage(cropped_image)

        # :COMMENT: Convert the numpy array to a vtkImageData object
        # vtk_image = vtk.vtkImageData()
        # vtk_image.SetDimensions(np_array.shape[::-1])
        # vtk_image.AllocateScalars(vtk.VTK_FLOAT, 1)
        # vtk_array = vtk.util.numpy_support.numpy_to_vtk(np_array.flatten())
        # vtk_image.GetPointData().SetScalars(vtk_array)

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
