# 3D Image Registration Software

**Members:** [Wissam Boussella](mailto:wissam.boussella@etu.u-bordeaux.fr), [Iantsa Provost](mailto:iantsa.provost@gmail.com), [Bastien Soucasse](mailto:bastien.soucasse@icloud.com), and [Tony Wolff](mailto:tony.wolff@etu.u-bordeaux.fr).

**Supervisor:** [Fabien Baldacci](mailto:fabien.baldacci@u-bordeaux.fr).

## Presentation

The 3D image registration software allows to apply classical image registration algorithms with the following steps.

- Loading two 3D images.
- Applying some pre-processing (cropping, resampling, ROI selection…).
- Applying a registration algorithm (rigid or non-rigid).
- Visualizing the result with a difference map.

## Libraries

The development is based on [ITK](https://itk.org) for the algorithmic part, and on [Slicer](https://slicer.org) for the interface part. The languages used are Python for the Slicer extension, and C++ to interact with the algorithms.
