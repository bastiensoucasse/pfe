"""
The Custom Registration module for Slicer provides the features for 3D images registration, based on the ITK library.
"""
import vtk, qt, ctk, slicer, numpy as np
import SimpleITK as sitk
from slicer import util, mrmlScene
from Processes import Process, ProcessesLogic
import sitkUtils as su
import sys, os
from math import pi
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleWidget,
    ScriptedLoadableModuleTest
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
        # :COMMENT: hide 3D view
        threeDWidget = slicer.app.layoutManager().threeDWidget(0)
        threeDWidget.setVisible(False)
        self.volume_selection_setup()
        self.update_gui()

    def volume_selection_setup(self) -> None:
        """
        Sets up the preprocessing widget by retrieving the volume selection widget and initializing it.
        """
        # :COMMENT: Link settings UI and code
        self.metrics_combo_box = self.panel_ui.findChild(ctk.ctkComboBox, "ComboMetrics")
        self.interpolator_combo_box = self.panel_ui.findChild(qt.QComboBox, "comboBoxInterpolator")
        self.optimizers_combo_box = self.panel_ui.findChild(ctk.ctkComboBox, "ComboOptimizers")
        self.volume_name_edit = self.panel_ui.findChild(qt.QLineEdit, "lineEditNewVolumeName")
        self.histogram_bin_count_spin_box = self.panel_ui.findChild(qt.QSpinBox, "spinBoxBinCount")
        self.sampling_strat_combo_box = self.panel_ui.findChild(qt.QComboBox, "comboBoxSamplingStrat")
        self.sampling_perc_spin_box = self.panel_ui.findChild(qt.QDoubleSpinBox, "doubleSpinBoxSamplingPerc")

        # :COMMENT: Gradients parameters
        self.gradients_box = self.panel_ui.findChild(ctk.ctkCollapsibleGroupBox, "CollapsibleGroupBoxGradient")
        self.learning_rate_spin_box = self.panel_ui.findChild(qt.QDoubleSpinBox, "doubleSpinBoxLearningR")
        self.nb_of_iter_spin_box = self.panel_ui.findChild(qt.QSpinBox, "spinBoxNbIter")
        self.conv_min_val_edit = self.panel_ui.findChild(qt.QLineEdit, "lineEditConvMinVal")
        self.conv_win_size_spin_box = self.panel_ui.findChild(qt.QSpinBox, "spinBoxConvWinSize")

        # :COMMENT: exhaustive parameters
        self.exhaustive_box = self.panel_ui.findChild(ctk.ctkCollapsibleGroupBox, "CollapsibleGroupBoxExhaustive")
        self.step_length_edit = self.panel_ui.findChild(qt.QLineEdit, "lineEditLength")
        self.nb_steps_edit = self.panel_ui.findChild(qt.QLineEdit, "lineEditSteps")
        self.opti_scale_edit = self.panel_ui.findChild(qt.QLineEdit, "lineEditScale")

        self.fill_combo_box()
        self.fixed_image_combo_box.setCurrentIndex(-1)
        self.moving_image_combo_box.setCurrentIndex(-1)
        # :COMMENT: handle button
        self.button_registration = self.panel_ui.findChild(ctk.ctkPushButton, "PushButtonRegistration")
        self.button_registration.clicked.connect(self.rigid_registration)

        
        self.fixed_image_combo_box.activated.connect(lambda index=self.fixed_image_combo_box.currentIndex, combobox=self.fixed_image_combo_box: self.on_volume_combo_box_changed(index, combobox))
        self.moving_image_combo_box.activated.connect(lambda index=self.moving_image_combo_box.currentIndex, combobox=self.moving_image_combo_box: self.on_volume_combo_box_changed(index, combobox))
        self.optimizers_combo_box.currentIndexChanged.connect(self.update_gui)
        self.selected_volume = None

    def vtk_to_sitk(self, volume: slicer.vtkMRMLScalarVolumeNode) -> sitk.Image:
        """
        Converts a VTK volume into a SimpleITK image.
        Parameters:
            volume: The VTK volume to convert.
        Returns:
            The SimpleITK image.
        """

        volume_image_data = volume.GetImageData()
        np_array = vtk.util.numpy_support.vtk_to_numpy(
            volume_image_data.GetPointData().GetScalars()
        )
        np_array = np.reshape(np_array, volume_image_data.GetDimensions()[::-1])
        image = sitk.GetImageFromArray(np_array)
        return image

    def sitk_to_vtk(self, image: sitk.Image) -> slicer.vtkMRMLScalarVolumeNode:
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
        vtk_array = vtk.util.numpy_support.numpy_to_vtk(np_array.flatten())
        volume_image_data.GetPointData().SetScalars(vtk_array)
        volume = slicer.vtkMRMLScalarVolumeNode()
        volume.SetAndObserveImageData(volume_image_data)
        return volume
    
    def has_registration_parameters_set(self) -> bool:
        if self.fixed_image_combo_box.currentIndex == -1 or self.moving_image_combo_box.currentIndex == -1:
            print("[DEBUG]: no volume selected !")
            return False
        if self.metrics_combo_box.currentIndex == 0:
            print("No metrics selected !")
            return False
        if self.optimizers_combo_box.currentIndex == 0:
            print("No optimizer selected !")
            return False
        if self.interpolator_combo_box.currentIndex == 0:
            print("No interpolator selected !")
            return False
        return True


    # Registration algorithm using simpleITK
    def rigid_registration(self):
        # :COMMENT: if no image is selected
        if not self.has_registration_parameters_set():
            return

        fixed_image_index = self.fixed_image_combo_box.currentIndex
        moving_image_index = self.moving_image_combo_box.currentIndex
        
        # :COMMENT: itk volume
        self.fixedVolumeData = self.volumes.GetItemAsObject(fixed_image_index)
        self.movingVolumeData = self.volumes.GetItemAsObject(moving_image_index)

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

        print(f"step length: {self.step_length}")
        print(f"nb steps : {self.nb_of_steps}")
        print(f"optimizer scale: {self.optimizer_scale}")

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
        """
        USING PARALLEL PROCESSING EXTENSION
        """
        def onProcessesCompleted(testClass):
            # when test finishes, we succeeded!
            print(logic.state())
            print('Test passed!')
            self.volumes = mrmlScene.GetNodesByClass("vtkMRMLScalarVolumeNode")
            new_volume = self.volumes.GetItemAsObject(self.volumes.GetNumberOfItems()-1)
            slicer.util.setSliceViewerLayers(background=new_volume, foreground=self.fixedVolumeData)
            slicer.util.resetSliceViews()

        logic = ProcessesLogic(completedCallback=lambda : onProcessesCompleted(self))
        thisPath = qt.QFileInfo(__file__).path()
        scriptPath = os.path.join(thisPath, "..", "scripts", "reg2.py")
        regProcess = RegistrationProcess(scriptPath, fixed_image, moving_image, input)
        logic.addProcess(regProcess)
        logic.run()


    # :COMMENT: helper function to determine when the threaded registration ends
    # transfer metadata and add the volume
    def onProcessingStatusUpdate(self, cliNode, event) -> None:
        if cliNode.GetStatus() & cliNode.Completed:
            if cliNode.GetStatus() & cliNode.ErrorsMask:
                errorText = cliNode.GetErrorText()
                print("CLI execution failed: " + errorText)
            else:
                node_name = cliNode.GetParameterAsString("outputVolume")
                output_volume = slicer.util.getNode(node_name)
                #:TODO: transform string to node
                #volume = self.sitk_to_vtk(resampled)
                #self.transfer_volume_metadate(self.movingVolumeData, output_volume)
                self.add_volume(output_volume)
                #print(f"[DEBUG]: {moving_image.GetName()}  as been registrated with parameters :\n< {self.parametersToPrint}> as {volume.GetName()}.")
        
    def transfer_volume_metadate(self, original_volume, moved_volume) -> None:
        spacing =  original_volume.GetSpacing()
        origin =  original_volume.GetOrigin()
        ijk_to_ras_direction_matrix = vtk.vtkMatrix4x4()
        original_volume.GetIJKToRASDirectionMatrix(ijk_to_ras_direction_matrix)

        # Apply the metadata to the target volume.
        moved_volume.SetSpacing(spacing)
        moved_volume.SetOrigin(origin)
        moved_volume.SetIJKToRASDirectionMatrix(ijk_to_ras_direction_matrix)

    def add_volume(self, volume) -> None:
        volume.SetName(self.volume_name_edit.text)
        #mrmlScene.AddNode(volume)
        slicer.util.setSliceViewerLayers(volume, fit=True)

    # :COMMENT: helper function to setup UI
    def fill_combo_box(self) -> None:
        self.metrics_combo_box.addItems(["Mean Squares",
                                            "Mattes Mutual Information"])
        self.optimizers_combo_box.addItems(["Gradient Descent",
                                            "Exhaustive"])
        self.interpolator_combo_box.addItems(["Linear",
                                            "Nearest Neighbors",
                                            "BSpline1",
                                            "BSpline2",
                                            "Gaussian"])
        self.sampling_strat_combo_box.addItems(["None",
                                                "Regular",
                                                "Random"])

        self.volumes = mrmlScene.GetNodesByClass("vtkMRMLScalarVolumeNode")
        
        # :COMMENT: insert volumes for fixed and moving images
        self.fixed_image_combo_box = self.panel_ui.findChild(ctk.ctkComboBox, "ComboFixedImage")
        self.fixed_image_combo_box.addItems([volume.GetName() for volume in self.volumes])

        self.moving_image_combo_box = self.panel_ui.findChild(ctk.ctkComboBox, "ComboMovingImage")
        self.moving_image_combo_box.addItems([volume.GetName() for volume in self.volumes])

        self.fixed_image_combo_box.addItem("Rename current volume…")
        self.fixed_image_combo_box.addItem("Delete current volume…")
        self.moving_image_combo_box.addItem("Rename current volume…")
        self.moving_image_combo_box.addItem("Delete current volume…")

    def on_volume_combo_box_changed(self, index: int, combo_box: ctk.ctkComboBox) -> None:
        """
        Handles the selection of volume for fixed and moving combo box, updates it

        Parameters:
            index: the selected volume or option
        """
        
        OPTIONS = ["Delete current volume…", "Rename current volume…"]
        name = combo_box.currentText
        self.active_combo_box = combo_box
        print(combo_box.currentText)
        print(f"index: {index}")
        # :COMMENT: Handle the different options.
        if name in OPTIONS:
            if self.volumes.GetNumberOfItems() < 1:
                self.display_error_message("No volumes imported.")
                self.update_active_combo_box()
                return

            if not self.selected_volume:
                self.display_error_message("Please select a volume first.")
                self.update_active_combo_box()
                return

            if name == "Rename current volume…":
                self.rename_volume()

            if name == "Delete current volume…":
                self.delete_volume()

        # :COMMENT: Select a new volume.
        else:
            self.selected_volume_index = index
            self.selected_volume = self.volumes.GetItemAsObject(index)

    def rename_volume(self) -> None:
        """
        Loads the renaming feature with a minimal window.
        """

        assert self.selected_volume
        self.renaming_old_name = self.selected_volume.GetName()
        self.renaming_input_dialog = qt.QInputDialog(None)
        self.renaming_input_dialog.setWindowTitle("Rename Volume")
        self.renaming_input_dialog.setLabelText("Enter the new name:")
        self.renaming_input_dialog.setModal(True)
        self.renaming_input_dialog.setTextValue(self.selected_volume.GetName())
        self.renaming_input_dialog.finished.connect(self.handle_rename_volume)
        self.renaming_input_dialog.show()

    def handle_rename_volume(self, result) -> None:
        """
        Applies the renaming of the selected volume.
        Parameters:
            result: The result of the input dialog.
        """

        if result == qt.QDialog.Accepted:
            assert self.selected_volume
            new_name = self.renaming_input_dialog.textValue()
            self.selected_volume.SetName(new_name)
            print(
                f'"{self.renaming_old_name}" has been renamed to "{self.selected_volume.GetName()}".'
            )
        self.update_active_combo_box()

    def update_active_combo_box(self) -> None:
        """
        Updates the selected volume combo box.
        """

        self.active_combo_box.clear()
        self.active_combo_box.addItems([volume.GetName() for volume in self.volumes])
        self.active_combo_box.addItem("Rename current volume…")
        self.active_combo_box.addItem("Delete current volume…")
        if self.selected_volume and self.selected_volume_index != -1 :
            self.active_combo_box.setCurrentIndex(self.selected_volume_index)
        else:
            print(self.selected_volume)
            self.active_combo_box.setCurrentIndex(-1)

    def delete_volume(self) -> None:
        """
        Deletes the currently selected volume.
        """

        assert self.selected_volume
        mrmlScene.RemoveNode(self.selected_volume)

        name = self.selected_volume.GetName()
        self.selected_volume = None
        self.selected_volume_index = None
        self.update_volume_list()

        print(f'"{name}" has been deleted.')

    def update_volume_list(self, caller=None, event=None) -> None:
        """
        Updates the list of volumes in the volume combobox when a change is detected in the MRML Scene.
        Parameters:
            caller: The widget calling this method.
            event: The event that triggered this method.
        """

        self.volumes = mrmlScene.GetNodesByClass("vtkMRMLScalarVolumeNode")
        self.update_active_combo_box()

    def reset_all_combo_box(self) -> None:
        """
        as name states
        """
        self.fixed_image_combo_box.clear()
        self.moving_image_combo_box.clear()
        self.metrics_combo_box.clear()
        self.optimizers_combo_box.clear()
        self.interpolator_combo_box.clear()
        self.fixed_image_combo_box.setCurrentIndex(-1)
        self.moving_image_combo_box.setCurrentIndex(-1)
        self.sampling_strat_combo_box.setCurrentIndex(2)

    # :TRICKY: getattr calls the function from object R, function is provided by metrics combo box
    def select_metrics(self, R, bin_count) -> None:
        """
        call the selected metrics by the user
        """
        metrics = self.metrics_combo_box.currentText.replace(" ", "")
        print(f"[DEBUG]: metrics: {metrics}")
        metrics_function = getattr(R, f"SetMetricAs{metrics}")
        if(metrics=="MattesMutualInformation"):
            metrics_function(bin_count)
        else:
            metrics_function()

    def select_interpolator(self, R) -> None:
        """
        set the interpolator selected by the user
        """
        interpolator = self.interpolator_combo_box.currentText.replace(" ", "")
        interpolator = getattr(sitk, f"sitk{interpolator}")
        print(f"[DEBUG]: interpolator: {interpolator}")
        R.SetInterpolator(interpolator)

    def select_optimizer_and_setup(self, R, learning_rate, nb_iteration, convergence_min_val, convergence_win_size) -> None:
        """
        set the optimizer (gradient descent, exhaustive...) and their respective parameters to be executed.
        """
        optimizerName = self.optimizers_combo_box.currentText.replace(" ", "")
        print(f"[DEBUG]: optimizer {optimizerName}")
        optimizer = getattr(R, f"SetOptimizerAs{optimizerName}")
        if optimizerName == "GradientDescent":
            self.parametersToPrint = f" Learning rate: {learning_rate}\n number of iterations: {nb_iteration}\n convergence minimum value: {convergence_min_val}\n convergence window size: {convergence_win_size}"
            optimizer(learningRate=learning_rate, numberOfIterations=nb_iteration, convergenceMinimumValue=float(convergence_min_val), convergenceWindowSize=convergence_win_size)
        elif optimizerName == "Exhaustive":
            self.parametersToPrint = f" number of steps: {self.nb_of_steps}\n step length: {self.step_length}\n optimizer scale: {self.optimizer_scale}"
            optimizer(numberOfSteps=self.nb_of_steps, stepLength = self.step_length)
            R.SetOptimizerScales(self.optimizer_scale)

    def update_gui(self) -> None:
        """
        updates the UI based on the choice of the user concerning the optimizer function
        """
        if self.optimizers_combo_box.currentText == "Gradient Descent":
            self.gradients_box.setEnabled(True)
            self.gradients_box.collapsed = 0
            self.exhaustive_box.setEnabled(False)
            self.exhaustive_box.collapsed = 1
        elif self.optimizers_combo_box.currentText == "Exhaustive":
            self.exhaustive_box.setEnabled(True)
            self.exhaustive_box.collapsed = 0
            self.gradients_box.setEnabled(False)
            self.gradients_box.collapsed = 1
        else:
            self.gradients_box.setEnabled(False)
            self.gradients_box.collapsed = 1      
            self.exhaustive_box.setEnabled(False)
            self.exhaustive_box.collapsed = 1  
        self.sampling_strat_combo_box.setCurrentIndex(2)

    # :COMMENT: from doc : https://simpleitk.readthedocs.io/en/master/link_ImageRegistrationMethod3_docs.html
    # Useful to analyse results during the registration process
    def command_iteration(self, method):
        if method.GetOptimizerIteration() == 0:
            print("Estimated Scales: ", method.GetOptimizerScales())
        print(
            f"{method.GetOptimizerIteration():3} "
            + f"= {method.GetMetricValue():7.5f} "
            + f": {method.GetOptimizerPosition()}"
        )

    def display_error_message(self, message: str) -> None:
        """
        Displays an error message.
        Parameters:
            message: Message to be displayed.
        """

        msg = qt.QMessageBox()
        msg.setIcon(qt.QMessageBox.Critical)
        msg.setText(message)
        msg.setWindowTitle("Error")
        msg.exec_()

import pickle
class RegistrationProcess(Process):

  def __init__(self, scriptPath, fixedImageVolume, movingImageVolume, input_parameters):
    Process.__init__(self, scriptPath)
    self.fixedImageVolume = fixedImageVolume
    self.movingImageVolume = movingImageVolume
    self.input_parameters = input_parameters

  def prepareProcessInput(self):
    input = {}
    input['fixed_image'] = self.fixedImageVolume
    input['moving_image'] = self.movingImageVolume
    input['parameters'] = self.input_parameters
    return pickle.dumps(input)

  def useProcessOutput(self, processOutput):
    import abc
    output = pickle.loads(processOutput)
    # thisPath = qt.QFileInfo(__file__).path()
    image_resampled= output['image_resampled']
    pixelID = output["pixelID"]
    caster = sitk.CastImageFilter()
    caster.SetOutputPixelType(pixelID)
    image = caster.Execute(image_resampled)
    su.PushVolumeToSlicer(image)
    # thisPath = os.path.join(thisPath, "registered_image.mha")
    # sitk.WriteImage(image, thisPath)
    # loadedVolumeNode = slicer.util.loadVolume(thisPath)
    # loadedVolumeNode.SetSpacing(image.GetSpacing())
    # loadedVolumeNode.SetOrigin(image.GetOrigin())

    

class CustomRegistrationTest(ScriptedLoadableModuleTest):

    def setUp(self):
        mrmlScene.Clear(0)
        
    def runTest(self):
        self.setUp()
        self.test_dummy()

    def test_dummy(self):
        print("TESTING : TODO")