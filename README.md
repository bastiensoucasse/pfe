# 3D Image Registration Software

**Members:** [Wissam Boussella](mailto:wissam.boussella@etu.u-bordeaux.fr), [Iantsa Provost](mailto:iantsa.provost@gmail.com), [Bastien Soucasse](mailto:bastien.soucasse@icloud.com), and [Tony Wolff](mailto:tony.wolff@etu.u-bordeaux.fr).

**Supervisor:** [Fabien Baldacci](mailto:fabien.baldacci@u-bordeaux.fr).

## Presentation

The 3D image registration software allows to apply classical image registration algorithms with the following steps.

- Loading two 3D images.
- Applying pre-processing (cropping, resampling, ROI selection…).
- Applying a registration algorithm (rigid or non-rigid).
- Visualizing the result with a difference map.

The development is based on [Slicer](https://slicer.org) for the interface part and [ITK](https://itk.org) for the algorithmic part. The languages used are Python for the Slicer extension and C++ to interact with the algorithms.

## Requirements

On your system, a Python 3 installation is required and it is recommended to use a dedicated virtual environment. You then need to install the Slicer application and the Python requirements (including ITK) for the project.

### Automatic Installation

If your system is an Apple macOS or a Linux distrubution, you can simply run the script `requirements/install_dependencies.sh` which should automatically install the Slicer application and all the Python requirements. Make sure to run this script inside your virtual environment if you do use one.

### Manual Installation

1. Follow the instructions from the Slicer documentation to install the application manually on your system.
2. Install the Python requirements with the `requirements/requirements.txt` file. You can use the following command from the root folder. Make sure to run this command inside your virtual environment if you do use one. (You might need to use `pip3` instead of `pip` depending on your system.)

```shell
pip install -r requirements/requirements.txt
```
