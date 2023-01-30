"""
The Custom Registration module for Slicer provides the features for 3D images registration, based on the ITK library.
"""

import qt
import slicer
from slicer.ScriptedLoadableModule import ScriptedLoadableModule, ScriptedLoadableModuleLogic, ScriptedLoadableModuleWidget


class CustomRegistration(ScriptedLoadableModule):
    """
    Main class for the Custom Registration module used to define the module‘s metadata.
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
        self.parent.contributors = ["Wissam Boussella (Université de Bordeaux)", "Iantsa Provost (Université de Bordeaux)", "Bastien Soucasse (Université de Bordeaux)", "Tony Wolff (Université de Bordeaux)"]
        self.parent.helpText = "The Custom Registration module for Slicer provides the features for 3D images registration, based on the ITK library."
        self.parent.acknowledgementText = "Supported by Fabien Machin (Université de Bordeaux)."


class CustomRegistrationLogic(ScriptedLoadableModuleLogic):
    """
    Logic class for the Custom Registration module used to define the module‘s algorithms.
    """

    def __init__(self):
        ScriptedLoadableModuleLogic.__init__(self)


class CustomRegistrationWidget(ScriptedLoadableModuleWidget):
    """
    Widget class for the Custom Registration module used to define the module‘s panel interface.
    """

    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)

    def setup(self):
        """
        Sets up the widget for the module by adding a welcome message to the layout.
        """

        ScriptedLoadableModuleWidget.setup(self)

        welcome_label = qt.QLabel("Welcome to my custom extension.")
        self.layout.addWidget(welcome_label)
