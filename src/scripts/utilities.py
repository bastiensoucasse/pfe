import SimpleITK as sitk

def select_metrics(R, bin_count, metrics_name) -> None:
    """
    call the selected metrics by the user
    """
    print(f"[DEBUG]: metrics: {metrics_name}")
    metrics_function = getattr(R, f"SetMetricAs{metrics_name}")
    if(metrics_name=="MattesMutualInformation"):
        metrics_function(bin_count)
    else:
        metrics_function()

def select_interpolator(R, interpolator_name) -> None:
    """
    set the interpolator selected by the user
    """
    interpolator = getattr(sitk, f"sitk{interpolator_name}")
    print(f"[DEBUG]: interpolator: {interpolator}")
    R.SetInterpolator(interpolator)

def select_optimizer_and_setup(R, optimizer_name, learning_rate, nb_iteration, convergence_min_val, convergence_win_size, nb_of_steps, step_length, optimizer_scale, solution_acc, nb_iter_lbfgs2, delta_conv_tol) -> None:
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
    elif optimizer == "LBFGS2":
        optimizer(solutionAccuracy=solution_acc, numberOfIterations=nb_iter_lbfgs2, deltaConvergenceTolerance=delta_conv_tol)