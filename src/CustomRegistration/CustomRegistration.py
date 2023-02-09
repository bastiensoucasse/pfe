"""
The Custom Registration module for Slicer provides the features for 3D images registration, based on the ITK library.
"""
import vtk, qt, ctk, slicer
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

        # Initialize setup from Slicer.
        ScriptedLoadableModuleWidget.setup(self)

        # Load UI file.
        self.panel_ui = util.loadUI(self.resourcePath("UI/Panel.ui"))
        self.layout.addWidget(self.panel_ui)

          # :COMMENT: Get all volumes, (merci Iantsa pour le code).
        self.volumes = mrmlScene.GetNodesByClass("vtkMRMLScalarVolumeNode")

        # :COMMENT: insert volumes for fixed and moving images
        self.fixed_image_combo_box = self.panel_ui.findChild(ctk.ctkComboBox, "ComboFixedImage")
        self.fixed_image_combo_box.addItems([volume.GetName() for volume in self.volumes])

        self.moving_image_combo_box = self.panel_ui.findChild(ctk.ctkComboBox, "ComboMovingImage")
        self.moving_image_combo_box.addItems([volume.GetName() for volume in self.volumes])

        # :COMMENT: handle button
        self.button_registration = self.panel_ui.findChild(ctk.ctkPushButton, "PushButtonRegistration")
        self.button_registration.clicked.connect(self.rigid_registration)


    def rigid_registration(self):
        print("[DEBUG] test")



