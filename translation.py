#import packages
import numpy as np
import argparse
import cv2
import imutils

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
	help = "Path to the image")
args = vars(ap.parse_args())

# Load the image and show it
image = cv2.imread(args["image"])
cv2.imshow("orig", image)
cv2.waitKey(0)

# Translating (shifting) is given by a numpy matrix
# with the form: [[1, 0, shiftX], [0, 1, shiftY]]
#negative x shifts left, negative y shifts up
#positive x shift right, positive y shifts down
#specify the shift on x and y axis in # px
#below we shift 25px right and 50px down
M = np.float32([[1, 0, 25], [0, 1, 50]])
#shift image
#supply the image, matrix (M)
#and dimensions, width (shape[1]) and height(shape[0])
#recall the image is an array tuple of h, w, and # channels
shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
cv2.imshow("shift down, right", shifted)
cv2.waitKey(0)

#below we shift 50px left and 90px up
M2 = np.float32([[1, 0, -50], [0, 1, -90]])
shifted2 = cv2.warpAffine(image, M2, (image.shape[1], image.shape[0]))
cv2.imshow("shift up, left", shifted2)
cv2.waitKey(0)

#use helper function from imutils package
#shift the image down by 100px
shifted3 = imutils.translate(image, 0, 100)
cv2.imshow("shift down", shifted3)
cv2.waitKey(0)
