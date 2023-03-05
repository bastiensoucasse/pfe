"""
Duplicate plugin script.

This script is provided as a sample plugin to demonstrate how to write a plugin for Custom Registration.
Be sure to write or generate a .ui file for your plugin, containing the widgets used to enter the parameters for your plugin script.
"""

import sys

# :COMMENT: Slicer already has a number of useful modules, including: slicer, vtk, ctk, qt, etc.
# :COMMENT: They can be imported even if they are not found on your system, as slicer knows where to find them.
# :COMMENT: Look into Slicer's documentation for more information.
import qt
import slicer


def run(
    ui: qt.QWidget,
    scene: slicer.vtkMRMLScene,
    input_volume: slicer.vtkMRMLScalarVolumeNode = None,
    target_volume: slicer.vtkMRMLScalarVolumeNode = None,
) -> None:
    """
    Duplicates the input volume in the scene provided, with a custom name provided in the UI.

    The run function is the entry point for the plugin.
    It is called when the plugin run button is clicked in the UI.
    You must write this function in your plugin script file (with the same parameters too) in order to run the plugin.

    Parameters:
        ui: The plugin UI to retrieve data from (parameters, values…). This is a Slicer UI element based on Qt. You can access any element in this UI by using `ui.findChild(type, name)`.
        scene: Slicer's MRML Scene (for adding new volumes or removing ones). You can check its usage in Slicer's documentation.
        input_volume: The input volume defined in Custom Registration extension module (appears in Custom Registration's top view). This is a Slicer vtkMRMLVolumeNode.
        target_volume: The target volume defined in Custom Registration extension module (appears in Custom Registration's bottom-left view). This is a Slicer vtkMRMLVolumeNode.
    """

    # :COMMENT: Ensure that an input volume is defined.
    if not input_volume:
        print("Duplicate Plugin: No input volume defined.", file=sys.stderr)
        return

    # :COMMENT: Retrieve the new name from the UI.
    duplicate_volume_name_line_edit = ui.findChild(
        qt.QLineEdit, "NewVolumeNameLineEdit"
    )
    assert duplicate_volume_name_line_edit
    duplicate_volume_name = duplicate_volume_name_line_edit.text

    # :COMMENT: Ensure that a new name is entered.
    if not duplicate_volume_name or duplicate_volume_name == "":
        print("Duplicate Plugin: No duplicate volume name entered.", file=sys.stderr)
        return

    # :COMMENT: Duplicate the input volume.
    duplicate_volume = scene.AddNewNodeByClass(
        "vtkMRMLScalarVolumeNode", duplicate_volume_name
    )
    duplicate_volume.CopyContent(input_volume)

    # :COMMENT: Log the completion of the duplicate operation.
    print("Duplicate Plugin: Done.")
