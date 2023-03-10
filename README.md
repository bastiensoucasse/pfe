# Custom Registration

![Custom Registration Logo](images/custom_registration_logo.png)

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

Activate the developer mode in 3D Slicer if not already done. You can achieve this by going into the _Application Settings_ (in the _Edit_ menu). Under _Developer_, check _Enable developer mode_ and restart the software.

#### 3D Slicer Source Build

To import the extension into 3D Slicer, you need to open the _Extension Wizard_ (in the _Developer Tools_ section of the module selector). Hit _Select Extension_ and open the `src` directory containing the sources of the extension. In the next dialog, make sure to load the _Custom Registration_ module.

### Launching the Extension Module

In Slicer, you can now find the Custom Registration module in the module selector (inside the _PFE_ section).

Launching this module might update the view and change the layout of the software. All previous data is saved and can be used from the Custom Registration module.

## Using Custom Registration

### Preprocessing

**This section is still under development.**

### Registration

**This section is still under development.**

### Plugins

Plugins are the easiest way to add new features to Custom Registration by importing graphical interfaces and algorithms into the extension module.

You can load existing plugins from the _Plugins_ resource folder (`src/CustomRegistration/Resources/Plugins`). But you can also create and load your own plugins to add features of your choice to Custom Registration.

A plugin consists of two files:

- A graphical interface stored in an UI (XML) file: `.ui`. This file must contain all the widgets you will want to manipulate to apply parameters to your algorithm. You can write your own or more easily generate one with dedicated tools such as Slicer's pre-built designer, which you can find into the _Application Settings_ (in the _Edit_ menu), under _Developer_ menu, or Qt Designer (also included in Qt Creator)–but you might be missing Slicer's custom widgets with the latter.
- A script stored in a Python file: `.py`. The script must contain a `run` function. Keep it mind that this function is only an entry point for your plugin. You can write all your script inside, use different functions and files, or even call algorithms written in different languages or from libraries. The `run` function receives four parameters:
    - `ui`: Your plugin graphical interface to retrieve data from (parameters, other values…). You can retrieve any element from your graphical interface as an object by using `ui.findChild(type, name)`, to access and control its value.
    - `scene`: Slicer's MRML Scene (for adding, removing, or editing volumes and other nodes). You can check its usage in [Slicer's documentation](https://slicer.readthedocs.io/en/latest/developer_guide/mrml_overview.html).
    - `input_volume`: The input volume defined in Custom Registration (appearing in the top view), which is the one that is either preprocessed or registered. If no volume is selected as input, its value will be `None`.
    - `target_volume`: The target volume defined in Custom Registration (appearing in the bottom-left view), which is the one that is selected as target to process the input volume accordingly. If no volume is selected as target, its value will be `None`.

#### Loading a Plugin

Head over to the _Plugins_ section of the Custom Registration module panel, where you can find a _Load Plugin_ button. When clicked, a _Plugin Loading_ dialog window opens. You can choose a name for your plugin, and load the graphical interface and script files. You can then hit _Load_ to add your plugin to the extension module.

![Plugin Loading Dialog Window](images/plugin_loading_dialog_window.png)

Once loaded, your plugin appears after the plugin loading area, under a collapible button named as your plugin. You can there find your graphical interface and a _Run_ button to execute your script.

![Plugins Section](images/plugins_section.png)
