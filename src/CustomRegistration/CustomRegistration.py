"""
The Custom Registration module for Slicer provides the features for 3D images registration, based on the ITK library.
"""

import qt

import slicer, ctk, vtk
from slicer.ScriptedLoadableModule import ScriptedLoadableModule, ScriptedLoadableModuleLogic, ScriptedLoadableModuleWidget
# import SlicerCustomAppUtilities

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
        self.parent.contributors = ["Wissam Boussella (Université de Bordeaux)", "Iantsa Provost (Université de Bordeaux)",
                                    "Bastien Soucasse (Université de Bordeaux)", "Tony Wolff (Université de Bordeaux)"]
        self.parent.helpText = "The Custom Registration module for Slicer provides the features for 3D images registration, based on the ITK library."
        self.parent.acknowledgementText = "Supported by Fabien Machin (Université de Bordeaux)."


class CustomRegistrationLogic(ScriptedLoadableModuleLogic):
    """
    Logic class for the Custom Registration module used to define the module‘s algorithms.
    """

    def __init__(self):
        ScriptedLoadableModuleLogic.__init__(self)

    def mseDisplay(self,input1,input2):
        #TODO : renvoyer un tableau d'intensité d'erreur 
        #Que faire quand ils n'ont pas la meme résolution ??
        pass

    def mean(self,input1,input2):
        #TODO : renvoyer la moyenne d'erreur entre les deux input
        #Que faire quand ils n'ont pas la meme résolution ??
        pass



class CustomRegistrationWidget(ScriptedLoadableModuleWidget):
    """
    Widget class for the Custom Registration module used to define the module‘s panel interface.
    """

    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)

    def printS():
        print("on est la ")
    def setup(self):
        """
        Sets up the widget for the module by adding a welcome message to the layout.
        """

        ScriptedLoadableModuleWidget.setup(self)

        welcome_label = qt.QLabel("Welcome .")
        self.layout.addWidget(welcome_label)
       
        path =  '/home/wboussella/Documents/M2/pfe/pfe/src/CustomRegistration/UI/mse.ui'  #TODO

        self.loadUI(path)

    def loadUI(self,path):
        self.uiWidget = slicer.util.loadUI(self.resourcePath(path))

        """
        Pour l'instant je n'utilise pas ma propre ui car je n'en ressent pas le besoin, mon interface est simple
        """
        # self.layout.addWidget(self.uiWidget)

        # self.ui = slicer.util.childWidgetVariables(self.uiWidget)

        slicer.app.layoutManager().sliceWidget("Red").hide()
        slicer.app.layoutManager().sliceWidget("Yellow").hide()
        
        # tuto suivi ici : https://docs.google.com/presentation/d/1JXIfs0rAM7DwZAho57Jqz14MRn2BIMrjB17Uj_7Yztc/edit#slide=id.g420896289_0251

        # Premier input selector 
        parametersCollapsideButton = ctk.ctkCollapsibleButton()
        parametersCollapsideButton.text = "param"
        parametersFormLayout = qt.QFormLayout(parametersCollapsideButton)
        self.layout.addWidget(parametersCollapsideButton)
        self.inputSelector1 = slicer.qMRMLNodeComboBox()
        self.inputSelector1.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.inputSelector1.selectNodeUponCreation = True
        self.inputSelector1.addEnabled = False
        self.inputSelector1.removeEnabled= False
        self.inputSelector1.noneEnabled = False
        self.inputSelector1.showHidden = False
        self.inputSelector1.showChildNodeTypes = False
        self.inputSelector1.setMRMLScene ( slicer.mrmlScene )
        self.inputSelector1.setToolTip( "node 1")
        parametersFormLayout.addRow("first volume : ",self.inputSelector1)


        self.inputSelector2 = slicer.qMRMLNodeComboBox()
        self.inputSelector2.nodeTypes = ["vtkMRMLScalarVolumeNode"]

        self.inputSelector2.selectNodeUponCreation = True
        self.inputSelector2.addEnabled = False

        self.inputSelector2.removeEnabled= False
        self.inputSelector2.noneEnabled = False
        self.inputSelector2.showHidden = False
        self.inputSelector2.showChildNodeTypes = False
        self.inputSelector2.setMRMLScene ( slicer.mrmlScene )
        self.inputSelector2.setToolTip( "node 2")
        parametersFormLayout.addRow("second volume volume : ",self.inputSelector2)
    
    def onApplyButton(self):
        function = CustomRegistrationLogic()
        #TODO faire un switch dans le futur 
        mean = function.mseDisplay(self.inputSelector1.currentNode(),self.inputSelector2.inputSelector1.currentNode())

    




        

        
        
