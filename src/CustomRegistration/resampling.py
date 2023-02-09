"""
Resampling algorithm and testing script.
"""

import SimpleITK as sitk


def resample(reference_image: sitk.Image, input_image: sitk.Image):
    """
    Resamples the input image to the reference image's size, spacing, and origin.

    Parameters:
        reference_image (sitk.Image): The reference image used as a template for the resampling.
        input_image (sitk.Image): The input image to be resampled.

    Returns:
        The resampled image.
    """

    # Get the reference image's size, spacing, and origin.
    reference_size = reference_image.GetSize()
    reference_spacing = reference_image.GetSpacing()
    reference_origin = reference_image.GetOrigin()

    # Set the output size, spacing, and origin to be the same as the reference image.
    output_size = reference_size
    output_spacing = reference_spacing
    output_origin = reference_origin

    # Use linear interpolation.
    interpolator = sitk.sitkLinear

    # Resample the input image to match the reference image's size, spacing, and origin.
    resampled_image = sitk.Resample(
        input_image,
        output_size,
        sitk.Transform(),  # No transform is used.
        interpolator,
        output_origin,
        output_spacing,
    )

    # Return the resampled image.
    return resampled_image


if __name__ == "__main__":
    import sys

    # Check if the required number of arguments have been provided.
    if len(sys.argv) != 4:
        print(
            "Usage:",
            sys.argv[0],
            "<reference_image> <input_image> <output_image>.",
            sep=" ",
        )
        sys.exit(1)

    # Read the arguments.
    reference_path = sys.argv[1]
    input_path = sys.argv[2]
    output_path = sys.argv[3]

    # Retrieve the reference and input images.
    reference_image = sitk.ReadImage(reference_path)
    input_image = sitk.ReadImage(input_path)

    # Generated a resampled image from the reference and input images.
    resampled_image = resample(reference_image, input_image)

    # Write the resampled image to the specified output path.
    sitk.WriteImage(resampled_image, output_path)
