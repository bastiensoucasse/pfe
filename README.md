# Custom Registration

![Custom Registration Logo](logo.png)

Custom Registration is a 3D image registration extension for 3D Slicer. It allows applying preprocessing to the images (ROI selection, cropping, and resampling), running rigid and non-rigid registration algorithms, computing and visualizing difference maps, and running your algorithms with your graphical interface plugged in.

**Members:** [Wissam Boussella](mailto:wissamboussella@gmail.com), [Iantsa Provost](mailto:iantsa.provost@gmail.com), [Bastien Soucasse](mailto:bastien.soucasse@icloud.com), and [Tony Wolff](mailto:toto.wolff@live.fr).

**Supervisors:** [Fabien Baldacci](mailto:fabien.baldacci@u-bordeaux.fr) and [Pascal Desbarats](mailto:pascal.desbarats@labri.fr).

## Getting Started

### 3D Slicer

As Custom Registation is an extension, you must have [3D Slicer](https://slicer.org) installed on your system. You can either install or build the software by following [the documentation](https://slicer.readthedocs.io/en/latest/user_guide/getting_started.html).

### Installing the Extension

Follow these instructions to install the extension from a release archive. Note you can instead build the extension from the source by following the instructions in [Building from Source](#building-from-source).

**The installation process from a release archive is not yet available. Please refer to the manual build instructions in [Building from Source](#building-from-source).**

### Building from Source

Follow these instructions to build the extension from the source. Note you can instead install the extension from a release archive by following the instructions in [Installing the Extension](#installing-the-extension).

#### 3D Slicer Developer Mode

Activate the developer mode in 3D Slicer if not already done. You can achieve this by going into the _Application Settings_ (in the _Edit_ menu). Under Developer, check _Enable developer mode_ and restart the software.

#### 3D Slicer Source Build

To import the extension into 3D Slicer, you need to open the _Extension Wizard_ (in the _Developer Tools_ section of the module selector). Hit _Select Extension_ and open the `src` directory containing the sources of the extension. In the next dialog, make sure to load the _Custom Registration_ module.

### Launching the Extension Module

In Slicer, you can now find the Custom Registration module in the module selector (inside the _PFE_ section).

Launching this module might update the view and change the layout of the software. All previous data is saved and can be used from the Custom Registration module.

## Using Custom Registration

**This section is still under development.**
