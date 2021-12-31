# USAGE
# python test_pyramid.py -i image1.jpg -s 1.5

# import packages
from obj_detect.object_detector.helpers import pyramid
import argparse
import cv2

# set up argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
ap.add_argument("-s", "--scale", type=float, default=1.5, help = "scale factor size")
args = vars(ap.parse_args())

# load the image
image = cv2.imread(args["image"])

# loop over the layers of the image pyramid and display them
for (i, layer) in enumerate(pyramid(image, scale=args["scale"])):
    cv2.imshow("Layer {}".format(i + 1), layer)
    cv2.waitKey(0)
