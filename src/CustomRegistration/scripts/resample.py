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


def test_resample():
    """
    Tests the resampling function by generating a reference image and multiple test images.

    The test images are resampled to match the reference image's size, spacing, and origin.
    The function then asserts that the resampled images' size, spacing, and origin are the same as the reference image.
    """

    # Define the reference image's size.
    REFERENCE_SIZE: tuple = (100, 100, 50)

    # Define the test sizes.
    TEST_SIZES: list[tuple] = [
        (50, 50, 50),
        (500, 500, 500),
        (100, 50, 10),
    ]

    # Generate a reference image.
    reference_image = sitk.Image(
        REFERENCE_SIZE,
        sitk.sitkFloat32,
    )

    # Display the reference image data.
    print("Reference image:")
    print(f"\t- Size: {reference_image.GetSize()}.")
    print(f"\t- Spacing: {reference_image.GetSpacing()}.")
    print(f"\t- Origin: {reference_image.GetOrigin()}.")

    # Loop through the test sizes.
    for i in range(len(TEST_SIZES)):
        # Generate an input image.
        input_image = sitk.Image(
            TEST_SIZES[i],
            sitk.sitkFloat32,
        )

        # Resample the input image to match the reference image's size, spacing, and origin.
        resampled_image = resample(reference_image, input_image)

        # Test if the resampled image size, spacing, and origin are the same as the reference image.
        assert resampled_image.GetSize() == reference_image.GetSize()
        assert resampled_image.GetSpacing() == reference_image.GetSpacing()
        assert resampled_image.GetOrigin() == reference_image.GetOrigin()

        # Display the resampled image data.
        print(f"\nTest {i}:")
        print(f"\t- Size: {input_image.GetSize()} to {resampled_image.GetSize()}.")
        print(
            f"\t- Spacing: {input_image.GetSpacing()} to {resampled_image.GetSpacing()}."
        )
        print(
            f"\t- Origin: {input_image.GetOrigin()} to {resampled_image.GetOrigin()}."
        )

    # Print a message indicating that the tests have passed.
    print("\nAll tests passed.")


def usage():
    """
    Displays the usage message.
    """

    print("Usage:")
    print("\t" + sys.argv[0] + " <reference_image> <input_image> <output_image>")
    print(
        "\t\tResamples the input image to match the size, spacing, and origin of the reference image."
    )
    print("\t" + sys.argv[0] + " --test")
    print("\t\tTests the resampling algorithm.")


if __name__ == "__main__":
    import sys

    # Check if there are only two arguments.
    if len(sys.argv) == 2:
        # Show usage if the argument "--help" is specified.
        if "--help" in sys.argv:
            usage()
            sys.exit(0)

        # Test the resampling algorithm if the argument "--test" is specified.
        if "--test" in sys.argv:
            test_resample()
            sys.exit(0)

    # Check if the required number of arguments have been provided.
    if len(sys.argv) != 4:
        usage()
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
