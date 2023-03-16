import SimpleITK as sitk
import sys
import os


def command_iteration(method):
    print(
        f"{method.GetOptimizerIteration():3} "
        + f"= {method.GetMetricValue():10.5f} "
        + f": {method.GetOptimizerPosition()}"
    )

def rigid_reg(moving, fixed):
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMeanSquares()
    R.SetMetricSamplingStrategy(R.RANDOM)
    #avoid randomization with by putting a seed
    R.SetMetricSamplingPercentage(0.01, seed=10)

    # Set the optimizer to LBFGS2
    R.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5,
    numberOfIterations=100,
    maximumNumberOfCorrections=5,
    maximumNumberOfFunctionEvaluations=1000)
    initial_transform = sitk.CenteredTransformInitializer(
        fixed,
        moving,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    R.SetInitialTransform(initial_transform)
    R.SetInterpolator(sitk.sitkLinear)
    R.SetOptimizerScalesFromIndexShift()

    R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))

    outTx = R.Execute(fixed, moving)

    print("-------")
    print(outTx)
    print(f"Optimizer stop condition: {R.GetOptimizerStopConditionDescription()}")
    print(f" Iteration: {R.GetOptimizerIteration()}")
    print(f" Metric value: {R.GetMetricValue()}")
    return outTx


def non_rigid_reg(moving, fixed):
    transformDomainMeshSize=[1]*fixed.GetDimension()
    tx = sitk.BSplineTransformInitializer(fixed, transformDomainMeshSize)

    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(50)
    R.SetMetricSamplingPercentage(0.01, seed=10)
    R.SetMetricSamplingStrategy(R.RANDOM)

    R.SetOptimizerAsGradientDescent(1, 100, 1e-5, 10)
    
    shrink_factor = [4, 2, 1]
    smoothing_sigmas = [2, 1, 0]
    R.SetInitialTransformAsBSpline(tx, inPlace=True, scaleFactors=[1, 2, 4])
    R.SetInterpolator(sitk.sitkLinear)
    R.SetShrinkFactorsPerLevel(shrink_factor)
    R.SetSmoothingSigmasPerLevel(smoothing_sigmas)
    outTx = R.Execute(fixed, moving)

    print("-------")
    print(outTx)
    print("Optimizer stop condition: {0}".format(R.GetOptimizerStopConditionDescription()))
    print(" Iteration: {0}".format(R.GetOptimizerIteration()))
    print(" Metric value: {0}".format(R.GetMetricValue()))
    return outTx

def main(args):
    if len(args) < 2:
        print(
            "Usage:",
            "rigid reg",
            "<fixedImageFilter> <movingImageFile>"
        )
        sys.exit(1)

    fixed = sitk.ReadImage(args[1], sitk.sitkFloat32)

    moving = sitk.ReadImage(args[2], sitk.sitkFloat32)

    outTx = non_rigid_reg(moving, fixed)

    sitk.WriteTransform(outTx, "expected_transform_5.tfm")

main(sys.argv)