#import modules
import argparse
import cv2
import numpy as np

#set up argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
    help = "Path to the image file")
args = vars(ap.parse_args())

# Load the image and show it
image = cv2.imread(args["image"])
cv2.imshow("orig", image)

#list of kernel sizes for blurring
kernelSizes = [(3, 3), (9, 9), (15, 15)]

#loop over the kernelSizes and use avg blur
#avg blur uses a simple mean for calculations
for (kX, kY) in kernelSizes:
    blurred = cv2.blur(image, (kX, kY))
    cv2.imshow("Avg ({}, {})".format(kX, kY), blurred)
    cv2.waitKey(0)

#clean up the windows
cv2.destroyAllWindows()

#loop over the kernelSizes and use gaussian blur
#gaussian blur uses a weighted mean for calculations
for (kX, kY) in kernelSizes:
    gaussian = cv2.GaussianBlur(image, (kX, kY), 0)
    cv2.imshow("Gaussian ({}, {})".format(kX, kY), gaussian)
    cv2.waitKey(0)

#clean up the windows
cv2.destroyAllWindows()

#loop over the kernelSizes and use median blur
#median blur is good for removing 'salt and pepper' from images
for k in (3, 9, 15):
    median = cv2.medianBlur(image, k)
    cv2.imshow("median {}".format(k), median)
    cv2.waitKey(0)
