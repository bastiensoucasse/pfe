"""
The Custom Registration module for Slicer provides the features for 3D images registration, based on the ITK library.
"""
import vtk, qt, ctk, slicer, numpy as np
import SimpleITK as sitk
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

        # :COMMENT: Link settings UI and code
        self.metrics_combo_box = self.panel_ui.findChild(ctk.ctkComboBox, "ComboMetrics")
        self.interpolator_combo_box = self.panel_ui.findChild(qt.QComboBox, "comboBoxInterpolator")
        self.optimizers_combo_box = self.panel_ui.findChild(ctk.ctkComboBox, "ComboOptimizers")

        self.volume_name = self.panel_ui.findChild(qt.QLineEdit, "lineEditNewVolumeName")
        self.histogram_bin_count_spin_box = self.panel_ui.findChild(qt.QSpinBox, "spinBoxBinCount")
        self.sampling_strat_combo_box = self.panel_ui.findChild(qt.QComboBox, "comboBoxSamplingStrat")
        self.sampling_perc = self.panel_ui.findChild(qt.QDoubleSpinBox, "doubleSpinBoxSamplingPerc")

        # :COMMENT: Gradients parameters
        self.learning_rate_spin_box = self.panel_ui.findChild(qt.QDoubleSpinBox, "doubleSpinBoxLearningR")
        self.nb_of_iter_spin_box = self.panel_ui.findChild(qt.QSpinBox, "spinBoxNbIter")
        self.conv_min_val = self.panel_ui.findChild(qt.QLineEdit, "lineEditConvMinVal")
        self.conv_win_size = self.panel_ui.findChild(qt.QSpinBox, "spinBoxConvWinSize")

        # :COMMENT: handle button
        self.button_registration = self.panel_ui.findChild(ctk.ctkPushButton, "PushButtonRegistration")
        self.button_registration.clicked.connect(self.rigid_registration)
        self.initUiComboBox()




    def vtk2sitk(self, vtkimg):
        np_array = vtk.util.numpy_support.vtk_to_numpy(vtkimg.GetPointData().GetScalars())
        np_array = np.reshape(np_array, (vtkimg.GetDimensions()[::-1]))
        sitk_image = sitk.GetImageFromArray(np_array)
        sitk_image = sitk.Cast(sitk_image, sitk.sitkFloat32)
        return sitk_image

    # from github : https://github.com/dave3d/dicom2stl/blob/main/utils/sitk2vtk.py
    def sitk2vtk(self, sitkimg):
        size = list(sitkimg.GetSize())
        origin = list(sitkimg.GetOrigin())
        spacing = list(sitkimg.GetSpacing())
        ncomp = sitkimg.GetNumberOfComponentsPerPixel()
        direction = sitkimg.GetDirection()

        # there doesn't seem to be a way to specify the image orientation in VTK

        # convert the SimpleITK image to a numpy array
        i2 = sitk.GetArrayFromImage(sitkimg)

        vtk_image = vtk.vtkImageData()

        # VTK expects 3-dimensional parameters
        if len(size) == 2:
            size.append(1)

        if len(origin) == 2:
            origin.append(0.0)

        if len(spacing) == 2:
            spacing.append(spacing[0])

        if len(direction) == 4:
            direction = [
                direction[0],
                direction[1],
                0.0,
                direction[2],
                direction[3],
                0.0,
                0.0,
                0.0,
                1.0,
            ]

        vtk_image.SetDimensions(size)
        vtk_image.SetSpacing(spacing)
        vtk_image.SetOrigin(origin)
        vtk_image.SetExtent(0, size[0] - 1, 0, size[1] - 1, 0, size[2] - 1)

        if vtk.vtkVersion.GetVTKMajorVersion() < 9:
            print("Warning: VTK version <9.  No direction matrix.")
        else:
            vtk_image.SetDirectionMatrix(direction)

        depth_array = vtk.util.numpy_support.numpy_to_vtk(i2.ravel())
        depth_array.SetNumberOfComponents(ncomp)
        vtk_image.GetPointData().SetScalars(depth_array)

        vtk_image.Modified()
        return vtk_image

    # Registration algorithme using simpleITK
    # :BUG: the registration is not centered
    def rigid_registration(self):
        # :COMMENT: if no image is selected
        if self.fixed_image_combo_box.currentIndex == 0 or self.moving_image_combo_box.currentIndex == 0:
            print("[DEBUG]: no volume selected !")
            return

        fixed_image_index = self.fixed_image_combo_box.currentIndex
        moving_image_index = self.moving_image_combo_box.currentIndex
        
        # :COMMENT: itk volume
        self.fixedVolumeData = self.volumes.GetItemAsObject(fixed_image_index-1).GetImageData()
        self.movingVolumeData = self.volumes.GetItemAsObject(moving_image_index-1).GetImageData()

        # :COMMENT: conversion to sitk volumes
        fixed_image = self.vtk2sitk(self.fixedVolumeData)
        moving_image = self.vtk2sitk(self.movingVolumeData)

        # :COMMENT: User settings retrieve
        bin_count = self.histogram_bin_count_spin_box.value
            # :COMMENT: Sampling strategies range from 0 to 2, they are enums (None, Regular, Random), thus index is sufficient
        sampling_strat = self.sampling_strat_combo_box.currentIndex
        sampling_perc = self.sampling_perc.value

        # :COMMENT: settings for gradients only
        learning_rate = self.learning_rate_spin_box.value
        nb_iteration = self.nb_of_iter_spin_box.value
        convergence_min_val = self.conv_min_val.text
        convergence_win_size = self.conv_win_size.value

        # :COMMENT: FROM simpleitk docs : https://simpleitk.readthedocs.io/en/master/link_ImageRegistrationMethod1_docs.html
        # a simple 3D rigid registration method
        R = sitk.ImageRegistrationMethod()
        self.selectMetrics(R, bin_count)
        R.SetMetricSamplingStrategy(sampling_strat)
        print(f"sampling strat random enum: {R.RANDOM}")
        print(f"selected sampling strat index:{sampling_strat}")
        R.SetMetricSamplingPercentage(sampling_perc)
        R.SetOptimizerAsGradientDescent(learningRate=learning_rate, numberOfIterations=nb_iteration, convergenceMinimumValue=float(convergence_min_val), convergenceWindowSize=convergence_win_size)
        initial_transform = sitk.CenteredTransformInitializer(fixed_image, 
                                                      moving_image, 
                                                      sitk.Euler3DTransform(), 
                                                      sitk.CenteredTransformInitializerFilter.GEOMETRY)
        R.SetInitialTransform(initial_transform, inPlace=False)
        R.SetInterpolator(sitk.sitkLinear)
        R.SetOptimizerScalesFromPhysicalShift()

        # final_transform = R.Execute(fixed_image, moving_image)

        # print("-------")
        # print(f"Optimizer stop condition: {R.GetOptimizerStopConditionDescription()}")
        # print(f" Iteration: {R.GetOptimizerIteration()}")
        # print(f" Metric value: {R.GetMetricValue()}")
        
        # moving_resampled = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())
        # self.create_volume(moving_resampled, fixed_image)
    
    # create a new volume
    def create_volume(self, sitkimg, fixed_img):
        volumeNode=slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        volumeNode.SetName(self.volume_name.text)
        itk_moved_volume = self.sitk2vtk(sitkimg)
        volumeNode.SetAndObserveImageData(itk_moved_volume)
        slicer.util.setSliceViewerLayers(volumeNode, fit=True)
        print(f"[DEBUG]: new volume {volumeNode.GetName()} created !")

    def initUiComboBox(self):
        self.metrics_combo_box.addItems(["Mean Squares",
                                            "Demons",
                                            "Correlation",
                                            "ANTS Neighborhood Correlation",
                                            "Joint Histogram Mutual Information",
                                            "Mattes Mutual Information"])
        self.optimizers_combo_box.addItems(["Gradient Descent",
                                            "Exhaustive",
                                            "Nelder-Mead downhill simplex",
                                            "Powell",
                                            "1+1 evolutionary optimizer"])
        self.interpolator_combo_box.addItems(["Linear",
                                            "Nearest Neighbors",
                                            "BSpline1",
                                            "BSpline2",
                                            "Gaussian"])
        self.sampling_strat_combo_box.addItems(["Regular",
                                                "Random"])

    # :COMMENT: call the selected metrics by the user
    # :TRICKY: getattr calls the function from object R, provided by metrics combo box
    def selectMetrics(self, R, bin_count):
        metrics = self.metrics_combo_box.currentText.replace(" ", "")
        print(f"[DEBUG]: metrics: {metrics}")
        metrics_function = getattr(R, f"SetMetricAs{metrics}")
        metrics_function(bin_count)





