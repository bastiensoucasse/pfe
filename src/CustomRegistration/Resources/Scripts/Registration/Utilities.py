"""
Registration algorithm and testing script utilities.
"""

import SimpleITK as sitk
interpolator_enum = {}
interpolator_enum["Linear"] = 2
interpolator_enum["Nearest Neighbor"] = 1
interpolator_enum["BSpline1"] = 12
interpolator_enum["BSpline2"] = 13
interpolator_enum["BSpline3"] = 11
interpolator_enum["Gaussian"] = 4

def select_metrics(R, bin_count, metrics_name) -> None:
    """
    Calls the user-defined metrics function.

    Parameters:
        R: Registration Method, sitk object
        bin_count: affects how the Mattes mutual information metric is computed.
                determines how many bins are used to estimate the probability density 
                of each image and their joint distribution.
        metrics_name: the user selected metrics (MeanSquares, MattesMutualInformation...)
    """

    metrics_function = getattr(R, f"SetMetricAs{metrics_name}")
    if metrics_name == "MattesMutualInformation":
        metrics_function(bin_count)
    else:
        metrics_function()


def select_interpolator(R, interpolator_name) -> None:
    """
    Sets the interpolator chosen by the user.

    Parameters:
        R: sitk object (ImageRegistrationMethod), interpolator is directly applied on this object
        interpolator_name: a string corresponding to the interpolator
    """
    R.SetInterpolator(interpolator_enum[interpolator_name])


def select_gradient_descent_optimizer(R, learning_rate, nb_iteration, convergence_min_val, convergence_win_size) -> None:
    """
    Selects the gradient descent optimizer

    Parameters:
        R: sitk object (ImageRegistrationMethod)
        learning_rate: float, The step size along the direction of the gradient for each iteration.
        nb_iteration: integer, The maximum number of iterations to perform.
        convergence_min_val: float, The minimum value of the metric change that indicates convergence. 
        convergence_win_size: integer, The number of previous iterations used to calculate the metric change.
    """
    R.SetOptimizerAsGradientDescent(learningRate=learning_rate, 
                                    numberOfIterations=nb_iteration, 
                                    convergenceMinimumValue=convergence_min_val, 
                                    convergenceWindowSize=convergence_win_size)
    
def select_exhaustive_optimizer(R, nb_of_steps, step_length, optimizer_scale) -> None:
    """
    Selects the exhaustive optimizer

    Parameters:
        R: sitk object (ImageRegistrationMethod)
        nb_of_steps: A vector of integers that specifies the number of steps along each dimension of the 
            transform's parameter space.
        step_length: vector of floats that specifies the increment along each dimension of the transform's parameter space.
        optimizer_scale: vector of floats that specifies how to scale each dimension of the transform's parameter space.
    """
    R.SetOptimizerAsExhaustive(numberOfSteps=nb_of_steps, stepLength=step_length)
    R.SetOptimizerScales(optimizer_scale)


def select_lbfgsb_optimizer(R, gradient_conv_tol, nb_iter_lbfgsb, max_nb_correction, max_func_eval) -> None:
    """
    Selects LBFGSB optimizer

    Parameters:
        R: sitk object (ImageRegistrationMethod)
        gradient_conv_tol: A float that specifies the convergence criterion for the projected gradient. 
        nb_iter_lbfgsb: specifies the maximum number of iterations to perform.
        max_nb_correction: An integer that specifies the maximum number of variable metric corrections used to define the limited memory matrix
        max_func_eval: An integer that specifies the maximum number of function evaluations allowed. 
    """
    R.SetOptimizerAsLBFGSB(
        gradientConvergenceTolerance=gradient_conv_tol,
        numberOfIterations=nb_iter_lbfgsb,
        maximumNumberOfCorrections=max_nb_correction,
        maximumNumberOfFunctionEvaluations=max_func_eval)

def select_optimizer_and_setup(R, parameters_dict) -> None:
    """
    Sets the optimizer (gradient descent, exhaustive…) and the respective parameters to be executed.

    Parameters:
        R: sitk object (ImageRegistrationMethod)
        parameters_dict : a dictionary that contains all sets of parameters for the optimizers
    """
    optimizer_name = parameters_dict["optimizer"]
    if optimizer_name == "Gradient Descent":
        select_gradient_descent_optimizer(R, 
        parameters_dict["learning_rate"], 
        parameters_dict["nb_iteration"], 
        parameters_dict["convergence_min_val"], 
        parameters_dict["convergence_win_size"])
    elif optimizer_name == "Exhaustive":
        select_exhaustive_optimizer(R, parameters_dict["nb_of_steps"], parameters_dict["step_length"], parameters_dict["optimizer_scale"])
    elif optimizer_name == "LBFGSB":
        select_lbfgsb_optimizer(R, parameters_dict["gradient_conv_tol"], parameters_dict["nb_iter_lbfgsb"], parameters_dict["max_nb_correction"], parameters_dict["max_func_eval"])
    else:
        raise ValueError("incorrect optimizer")

def get_demons_algorithm(name) -> sitk.DemonsRegistrationFilter:
    """
    Gets and the associated demons algorithm with the name parameter.

    Parameters:
        name : the name of the demons algorithm (Demons, DiffeomorphicDemons...)
    """
    demons = getattr(sitk, f"{name}RegistrationFilter")
    return demons()