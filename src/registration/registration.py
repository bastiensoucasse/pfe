#!/usr/bin/env python-real
import SimpleITK as sitk
import sys

def main(fixed_image, moving_image, output):
    fixed_image = sitk.ReadImage(fixed_image)
    moving_image = sitk.ReadImage(moving_image)
    fixed_image = sitk.Cast(fixed_image, sitk.sitkFloat32)
    moving_image = sitk.Cast(moving_image, sitk.sitkFloat32)
    
    initial_transform = sitk.CenteredTransformInitializer(fixed_image, 
                                                      moving_image, 
                                                      sitk.Euler3DTransform(), 
                                                      sitk.CenteredTransformInitializerFilter.GEOMETRY)

    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.01)

    R.SetInterpolator(sitk.sitkLinear)

    R.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    R.SetOptimizerScalesFromPhysicalShift()
    R.SetInitialTransform(initial_transform, inPlace=False)

    final_transform = R.Execute(fixed_image, moving_image)
    print("-------")
    print(f"Optimizer stop condition: {R.GetOptimizerStopConditionDescription()}")
    print(f" Iteration: {R.GetOptimizerIteration()}")
    print(f" Metric value: {R.GetMetricValue()}")
    pixelID = moving_image.GetPixelID()
    resampled = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())
    caster = sitk.CastImageFilter()
    caster.SetOutputPixelType(pixelID)
    image = caster.Execute(resampled)
    print(output)
    writer = sitk.ImageFileWriter()
    writer.SetFileName(output)
    writer.Execute(image)

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: registration <input> <input2> <output>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
