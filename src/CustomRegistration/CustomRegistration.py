"""
The Custom Registration module for Slicer provides the features for 3D images registration, based on the ITK library.
"""

import slicer
from qt import QPushButton, QSpinBox, QComboBox
from slicer import util, mrmlScene
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

        # :COMMENT: Get the crop button widget.
        self.crop_button = self.panel_ui.findChild(QPushButton, "crop_button")
        self.crop_button.clicked.connect(self.crop)

        # :COMMENT: Get the volume combobox widget.
        self.volumeComboBox = self.panel_ui.findChild(QComboBox, "volume")
        self.volumeComboBox.activated.connect(self.onVolumeActivated)

        # :COMMENT: Add the available volumes and options to the combobox.
        volumes = mrmlScene.GetNodesByClass("vtkMRMLScalarVolumeNode")

        # :BUG: Supposed to be a placeholder, but still appears in list (shouldn't)
        self.volumeComboBox.insertItem(0, "Select a Volume")

        for i in range(volumes.GetNumberOfItems()):
            volume = volumes.GetItemAsObject(i)
            self.volumeComboBox.addItem(volume.GetName())

        self.volumeComboBox.addItem("Rename current Volume")
        self.volumeComboBox.addItem("Delete current Volume")

        # :COMMENT: Add observer to update combobox when new volume is added to MRML Scene.
        # :BUG: List doesn't update when new loaded volume.
        # self.observerTag = mrmlScene.AddObserver(slicer.vtkMRMLScene.NodeAddedEvent, self.updateCombobox)


    def crop(self):
        # :COMMENT: Print name, ID and dimensions of selected volume.
        volumes = mrmlScene.GetNodesByClass("vtkMRMLScalarVolumeNode")
        volume = volumes.GetItemAsObject(self.currentVolumeIndex-1)
        volumeData = volume.GetImageData()
        print("Current volume name:", volume.GetName())
        print("Current volume index:", self.currentVolumeIndex)
        print("Dimensions:", volumeData.GetDimensions(), "\n")

        # :COMMENT: Retrieve coordinates input and print them.
        start_x = self.panel_ui.findChild(QSpinBox, "sx").value
        start_y = self.panel_ui.findChild(QSpinBox, "sy").value
        start_z = self.panel_ui.findChild(QSpinBox, "sz").value
        end_x = self.panel_ui.findChild(QSpinBox, "ex").value
        end_y = self.panel_ui.findChild(QSpinBox, "ey").value
        end_z = self.panel_ui.findChild(QSpinBox, "ez").value
        print("Start X:", start_x)
        print("Start Y:", start_y)
        print("Start Z:", start_z)
        print("End X:", end_x)
        print("End Y:", end_y)
        print("End 2:", end_z)

    def onVolumeActivated(self, index):
        options = ["Select a Volume", "Delete current Volume", "Rename current Volume"]

        name = self.volumeComboBox.currentText
        if name in options:
            # :TODO: Add renaming and deleting features.
            print(name, "not implemented yet!\n")

        else:
            self.currentVolumeIndex = index
            # :TODO: Set min and max for coordinates

    # def updateCombobox(self, caller, event):
    #     volumes = mrmlScene.GetNodesByClass("vtkMRMLNode")
    #     volumesNames = []
    #     for i in range(volumes.GetNumberOfItems()):
    #         volume = volumes.GetItemAsObject(i)
    #         if volume.GetName() != "":
    #             volumesNames.append(volume.GetName())
    #     self.comboBox.clear()
    #     self.comboBox.addItems(volumesNames)
