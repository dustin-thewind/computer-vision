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

#parameters are: (diameter, color, space) of the filter
#diameter is the pixel neighborhood we want to blur
#larger diameter means more pixels will be included in the calculations
#color is color standard deviation
#the larger color value means more colors will be included when computing blur
#space is standard deviation of the blur
#larger value means pixels farther from center of diameter will influence
#the blurring calculation
params = [(11, 21, 7), (11, 41, 21), (11, 61, 39)]

#loop over the diameter, sigma color, and sigma space
for (diameter, sigma_color, sigma_space) in params:
    blurred = cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)
    title = "Blurred d:{}, sc:{}, ss:{}".format(diameter, sigma_color, sigma_space)
    cv2.imshow(title, blurred)
    cv2.waitKey(0)
