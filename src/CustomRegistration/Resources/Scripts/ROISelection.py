"""
ROI Selection algorithm and testing script.
"""

import SimpleITK as sitk


def select_roi(image: sitk.Image, threshold: float = 0.5):
    """
    Selects ROI from an image.

    Parameters:
        image: The input image.
        threshold (optional): The threshold value. The default is 0.5.
    """
    raise NotImplementedError


def test_select_roi():
    """
    Tests the ROI selection function.
    """

    raise NotImplementedError
