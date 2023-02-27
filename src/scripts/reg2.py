import SimpleITK as sitk
import pickle
import sys


pickledInput = sys.stdin.buffer.read()
input = pickle.loads(pickledInput)
fixed_image = input["fixed_image"]
moving_image = input["moving_image"]

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


resampled = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())
# caster = sitk.CastImageFilter()
# caster.SetOutputPixelType(pixelID)
# image = caster.Execute(resampled)

output = {}
output['image_resampled'] = resampled
output["pixelID"] = fixed_image.GetPixelID()

sys.stdout.buffer.write(pickle.dumps(output))