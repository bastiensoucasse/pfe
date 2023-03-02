"""
Resampling algorithm and testing script.
"""

import SimpleITK as sitk


def resample(input_image: sitk.Image, target_image: sitk.Image) -> sitk.Image:
    """
    Resamples the input image to the reference image's size, spacing, and origin.

    Parameters:
        input_image: The input image to be resampled.
        reference_image: The reference image used as a template for the resampling.

    Returns:
        The resampled image.
    """

    # Use the default transform.
    transform = sitk.Transform()

    # Use the default interpolation.
    interpolator = sitk.sitkLinear

    # Resample the input image to match the reference image's size, spacing, and origin.
    resampled_image = sitk.Resample(
        input_image,
        target_image,
        transform,
        interpolator,
    )

    # Return the resampled image.
    return resampled_image


def test_resample() -> None:
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
    target_image = sitk.Image(
        REFERENCE_SIZE,
        sitk.sitkFloat32,
    )

    # Display the reference image data.
    print("Reference image:")
    print(f"\t- Spacing: {target_image.GetSpacing()}.")
    print(f"\t- Origin: {target_image.GetOrigin()}.")

    # Loop through the test sizes.
    for i in range(len(TEST_SIZES)):
        # Generate an input image.
        input_image = sitk.Image(
            TEST_SIZES[i],
            sitk.sitkFloat32,
        )

        # Resample the input image to match the reference image's size, spacing, and origin.
        resampled_image = resample(input_image, target_image)

        # Test if the resampled image size, spacing, and origin are the same as the reference image.
        assert resampled_image.GetSpacing() == target_image.GetSpacing()
        assert resampled_image.GetOrigin() == target_image.GetOrigin()

        # Display the resampled image data.
        print(f"\nTest {i}:")
        print(
            f"\t- Spacing: {input_image.GetSpacing()} to {resampled_image.GetSpacing()}."
        )
        print(
            f"\t- Origin: {input_image.GetOrigin()} to {resampled_image.GetOrigin()}."
        )

    # Print a message indicating that the tests have passed.
    print("\nAll tests passed.")


def usage() -> None:
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
