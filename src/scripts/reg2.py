import SimpleITK as sitk
import pickle
import sys


pickledInput = sys.stdin.buffer.read()
input = pickle.loads(pickledInput)
fixed_image = input["fixed_image"]
moving_image = input["moving_image"]
# user inputs
parameters = input["parameters"]
volume_name = parameters["volume_name"]
metrics_name = parameters["metrics"]
interpolator_name = parameters["interpolator"]
optimizer_name = parameters["optimizer"]
bin_count = parameters["histogram_bin_count"]
sampling_strat = parameters["sampling_strategy"]
sampling_perc = parameters["sampling_percentage"]

#parameters for gradient optimizer
learning_rate = parameters["learning_rate"]
nb_iteration = parameters["iterations"]
convergence_min_val = parameters["convergence_min_val"]
convergence_win_size = parameters["convergence_win_size"]
#parameters for exhaustive optimizer
nb_of_steps = parameters["nb_of_steps"]
step_length = parameters["step_length"]
optimizer_scale = parameters["optimizer_scale"]

def main():
    initial_transform = sitk.CenteredTransformInitializer(fixed_image, 
                                                        moving_image, 
                                                        sitk.Euler3DTransform(), 
                                                        sitk.CenteredTransformInitializerFilter.GEOMETRY)

    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=bin_count)
    R.SetMetricSamplingStrategy(sampling_strat)
    R.SetMetricSamplingPercentage(sampling_perc)
    select_metrics(R, bin_count)
    select_interpolator(R)
    select_optimizer_and_setup(R)
    R.SetOptimizerScalesFromPhysicalShift()
    R.SetInitialTransform(initial_transform, inPlace=False)

    final_transform = R.Execute(fixed_image, moving_image)
    print("-------")
    print(f"Optimizer stop condition: {R.GetOptimizerStopConditionDescription()}")
    print(f" Iteration: {R.GetOptimizerIteration()}")
    print(f" Metric value: {R.GetMetricValue()}")


    resampled = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())

    output = {}
    output['image_resampled'] = resampled
    output["pixelID"] = fixed_image.GetPixelID()

    sys.stdout.buffer.write(pickle.dumps(output))

def select_metrics(R, bin_count) -> None:
    """
    call the selected metrics by the user
    """
    print(f"[DEBUG]: metrics: {metrics_name}")
    metrics_function = getattr(R, f"SetMetricAs{metrics_name}")
    if(metrics_name=="MattesMutualInformation"):
        metrics_function(bin_count)
    else:
        metrics_function()

def select_interpolator(R) -> None:
    """
    set the interpolator selected by the user
    """
    interpolator = getattr(sitk, f"sitk{interpolator_name}")
    print(f"[DEBUG]: interpolator: {interpolator}")
    R.SetInterpolator(interpolator)

def select_optimizer_and_setup(R) -> None:
    """
    set the optimizer (gradient descent, exhaustive...) and their respective parameters to be executed.
    """
    print(f"[DEBUG]: optimizer {optimizer_name}")
    optimizer = getattr(R, f"SetOptimizerAs{optimizer_name}")
    if optimizer_name == "GradientDescent":
        parametersToPrint = f" Learning rate: {learning_rate}\n number of iterations: {nb_iteration}\n convergence minimum value: {convergence_min_val}\n convergence window size: {convergence_win_size}"
        optimizer(learningRate=learning_rate, numberOfIterations=nb_iteration, convergenceMinimumValue=float(convergence_min_val), convergenceWindowSize=convergence_win_size)
    elif optimizer_name == "Exhaustive":
        parametersToPrint = f" number of steps: {nb_of_steps}\n step length: {step_length}\n optimizer scale: {optimizer_scale}"
        optimizer(numberOfSteps=nb_of_steps, stepLength = step_length)
        R.SetOptimizerScales(optimizer_scale)

main()