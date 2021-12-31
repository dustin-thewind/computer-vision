
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
cv2.waitKey(0)

#mask allows us to focus on interesting pixels in an image
#mask is same size as the image, but only has 2 pixel values
#0 and 255 (ON or OFF)
#pixels with value 0 are ignored in the original image
#pixels with value 255 are kept
mask = np.zeros(image.shape[:2], dtype="uint8")
cv2.rectangle(mask, (0, 90), (290, 450), 255, -1)
cv2.imshow("mask", mask)

#apply the mask using bitwise_and
#the mask=mask keyword arg sets the bitwiseAND to TRUE
#when the pixel values of the images are equal and
#the mask is >0 at each x,y coordinate
masked = cv2.bitwise_and(image, image, mask=mask)
cv2.imshow("body", masked)
cv2.waitKey(0)

mask = np.zeros(image.shape[:2], dtype="uint8")
cv2.circle(mask, (145, 200), 100, 255, -1)
masked = cv2.bitwise_and(image, image, mask=mask)
cv2.imshow("circle mask", mask)
cv2.imshow("face masked", masked)
cv2.waitKey(0)xs
