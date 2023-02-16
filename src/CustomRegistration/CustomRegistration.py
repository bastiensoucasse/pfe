"""
The Custom Registration module for Slicer provides the features for 3D images registration, based on the ITK library.
"""

import qt

import slicer
import vtk,ctk
import slicer
import SimpleITK as sitk
import numpy as np
# from slicer.parameterNodeWrapper import *
# from MRMLCorePython import vtkMRMLModelNode
# from sitk import sitkUtils

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

    
    def mseDisplay(self, imageData1, imageData2):
        

        dims1 = imageData1.GetImageData().GetDimensions()
        dims2 = imageData2.GetImageData().GetDimensions()

        if dims1 != dims2:
            raise ValueError("Images must have the same dimensions")

        outputImage = vtk.vtkImageData()
        outputImage.SetDimensions(dims1)
        outputImage.AllocateScalars(vtk.VTK_FLOAT, 1)

        
        mini = 100000
        maxi = 0 
        for z in range(dims1[2]):
            print("z= ",z)
            for y in range(dims1[1]):
                for x in range(dims1[0]):
                    pixel1 = imageData1.GetImageData().GetScalarComponentAsFloat(x, y, z, 0)
                    pixel2 = imageData2.GetImageData().GetScalarComponentAsFloat(x, y, z, 0)
                    diff = abs(pixel1 - pixel2)
                    if diff < mini : 
                        mini = diff

                    elif diff>maxi :
                        maxi = diff
                    outputImage.SetScalarComponentFromFloat(x, y, z, 0, diff)
                    # outputImage.SetScalarComponentFromFloat(x, y, z, 1, diff)
                    # outputImage.SetScalarComponentFromFloat(x, y, z, 2, diff)

        print ("mini maxi : ",mini , maxi)
        outputNode = slicer.vtkMRMLScalarVolumeNode()
        outputNode.SetAndObserveImageData(outputImage)
        outputNode.SetName("SquareDifference")
        slicer.mrmlScene.AddNode(outputNode)


    def mean(self, input1, input2):
        # TODO : renvoyer la moyenne d'erreur entre les deux input
        # Que faire quand ils n'ont pas la meme résolution ??

        # https://discourse.itk.org/t/compute-a-mse-metric-for-a-fixed-and-moving-image-manual-registration/5161/3
        imageData1 = input1.GetImageData()
        imageData2 = input2.GetImageData()
        dimensions = imageData1.GetDimensions()
        numberOfScalarComponents = imageData1.GetNumberOfScalarComponents()

        mean = 0

        for z in range(dimensions[2]):
            print("slice z = ", z)
            for y in range(dimensions[1]):
                for x in range(dimensions[0]):
                    pixelIndex = [x, y, z, 0]
                    pixelValue1 = imageData1.GetScalarComponentAsDouble(x, y, z, 0)
                    pixelValue2 = imageData2.GetScalarComponentAsDouble(x, y, z, 0)

                    mean = mean + abs(pixelValue1 - pixelValue2)

        return mean/(dimensions[2]*dimensions[1]*dimensions[0])
                    # Do something with the pixel value

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

        path = '/home/wboussella/Documents/M2/pfe/pfe/src/CustomRegistration/UI/mse.ui'  # TODO

        self.loadUI(path)

    def loadUI(self, path):
        self.uiWidget = slicer.util.loadUI(self.resourcePath(path))

        """
        Pour l'instant je n'utilise pas ma propre ui car je n'en ressent pas le besoin, mon interface est simple
        """
        # self.layout.addWidget(self.uiWidget)

        # self.ui = slicer.util.childWidgetVariables(self.uiWidget)

        # slicer.app.layoutManager().sliceWidget("Red").hide()
        # slicer.app.layoutManager().sliceWidget("Yellow").hide()

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
        self.inputSelector1.removeEnabled = False
        self.inputSelector1.noneEnabled = False
        self.inputSelector1.showHidden = False
        self.inputSelector1.showChildNodeTypes = False
        self.inputSelector1.setMRMLScene(slicer.mrmlScene)
        self.inputSelector1.setToolTip("node 1")
        parametersFormLayout.addRow("first volume : ", self.inputSelector1)

        self.inputSelector2 = slicer.qMRMLNodeComboBox()
        self.inputSelector2.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.inputSelector2.selectNodeUponCreation = True
        self.inputSelector2.addEnabled = False
        self.inputSelector2.removeEnabled = False
        self.inputSelector2.noneEnabled = False
        self.inputSelector2.showHidden = False
        self.inputSelector2.showChildNodeTypes = False
        self.inputSelector2.setMRMLScene(slicer.mrmlScene)
        self.inputSelector2.setToolTip("node 2")
        parametersFormLayout.addRow("second volume : ", self.inputSelector2)

        print(self.inputSelector1.currentNode().GetImageData().GetScalarRange())
        # inputImage = sitkUtils.PullVolumeFromSlicer(self.inputSelector1.currentNode())
        
        tensors = np.array( self.inputSelector1.currentNode().GetImageData().GetPointData().GetTensors())

        function = CustomRegistrationLogic()
        # mean = function.mean(slicer.util.getNode('T1'),slicer.util.getNode('T2'))

        function.mseDisplay(slicer.util.getNode('T2'),slicer.util.getNode('T1'))
        
        # print("the mean is ",mean)
        
        # extent = self.inputSelector1.currentNode().GetImageData().GetExtent()
        # idx = 0
        # for k in range(extent[4], extent[5]+1):
        #     for j in range(extent[2], extent[3]+1):
        #         for i in range(extent[0], extent[1]+1):
        #             print(idx)
                    
        #             idx += 1

        
    def onApplyButton(self):
    # function = CustomRegistrationLogic()
    # #TODO faire un switch dans le futur
    # mean = function.mseDisplay(self.inputSelector1.currentNode(),self.inputSelector2.inputSelector1.currentNode())

    # print("moyenne= : ",mean)
        pass
