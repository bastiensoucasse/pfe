import numpy as np
import SimpleITK as sitk

IMAGE_SIZE = (200, 200, 200)
NOISE = True

# Define the image size and center of the sphere
center = (
    IMAGE_SIZE[0] // 2,
    IMAGE_SIZE[1] // 2,
    IMAGE_SIZE[2] // 2,
)

# Define the diameter of the sphere as 1/3 of the image size in every dimension
diameter = (2 * IMAGE_SIZE[0] // 3, 2 * IMAGE_SIZE[1] // 3, 2 * IMAGE_SIZE[2] // 3)

# Create a binary NumPy array with zeros
binary_array = np.zeros(IMAGE_SIZE)

# Define the radius of the sphere
radius = min(diameter) // 2

# Calculate the minimum and maximum indices for the sphere in each dimension
min_indices = tuple(center[i] - diameter[i] // 2 for i in range(3))
max_indices = tuple(center[i] + diameter[i] // 2 for i in range(3))

# Create a mesh grid of coordinates
x, y, z = np.ogrid[
    min_indices[0] : max_indices[0],
    min_indices[1] : max_indices[1],
    min_indices[2] : max_indices[2],
]

# Calculate the distance from each point to the center of the sphere
distance_from_center = np.sqrt(
    (x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2
)

# Set the binary array values to 1 where the distance from the center is less than the radius
binary_array[
    min_indices[0] : max_indices[0],
    min_indices[1] : max_indices[1],
    min_indices[2] : max_indices[2],
][distance_from_center <= radius] = (
    2 if NOISE else 1
)

# Add random noise around the sphere with pixel values of 1
if NOISE:
    noise = np.random.rand(*IMAGE_SIZE) < 0.5
    noise[
        min_indices[0] : max_indices[0],
        min_indices[1] : max_indices[1],
        min_indices[2] : max_indices[2],
    ] = 0
    binary_array[noise] = 1

# Convert the binary array to a SimpleITK image
sphere_image = sitk.GetImageFromArray(binary_array)
sphere_image.SetOrigin((-center[0], -center[1], -center[2]))

# Save the image as an NRRD file
sitk.WriteImage(sphere_image, f'Sphere{"Noised" if NOISE else ""}.nrrd')
