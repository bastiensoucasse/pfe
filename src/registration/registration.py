#!/usr/bin/env python-real

import sys
def main(R, fixed_image, moving_image, output):
    import SimpleITK as sitk

    final_transform = R.Execute(fixed_image, moving_image)
    output[0] = fixed_image
    output[1] = moving_image
    output[2] = final_transform
    output[3] = R

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: registration <Registration method> <fixed_image> <moving_image> <output>")
        sys.exit(1)
    main(sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.arg[4])
