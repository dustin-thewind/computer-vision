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

#get dimensions of image and find the center
(h, w) = image.shape[:2]
(cX, cY) = (w / 2, h / 2)

#rotate the image by 45 deg
#specify the nunber of counter clockwise degrees
#we want to rotate by
M = cv2.getRotationMatrix2D((cX, cY), 45, 1.0)
rotated = cv2.warpAffine(image, M, (w, h))
cv2.imshow("rotated 45", rotated)
#cv2.waitKey(0)

#rotate the image by -90 deg
#this will rotate the image clockwise
M = cv2.getRotationMatrix2D((cX, cY), -90, 1.0)
rotated = cv2.warpAffine(image, M, (w, h))
cv2.imshow("rotated -90", rotated)
#cv2.waitKey(0)

#rotate the image by 45 deg about a point rather than the center
#this will rotate the image clockwise
M = cv2.getRotationMatrix2D((cX - 50, cY - 50), 45, 1.0)
rotated = cv2.warpAffine(image, M, (w, h))
cv2.imshow("rotated 45 about a point", rotated)
#cv2.waitKey(0)

#use helper function from imutils
#rotate by 180 degrees
rotated = imutils.rotate(image, 180)
cv2.imshow("rotated 180", rotated)
#cv2.waitKey(0)

#use helper function from imutils
#rotate by 180 degrees
rotated = imutils.rotate(image, 110)
cv2.imshow("rotated -30", rotated)


M = cv2.getRotationMatrix2D((50, 50), 88, 1.0)
rotated2 = cv2.warpAffine(image, M, (w, h))#cv2.imshow("quiz", rotated2)

(b, g, r) = rotated2[10, 10]
print("Pixel at (0, 0) - Red: {r}, Green: {g}, Blue: {b}"
    .format(r=r, g=g, b=b))

cv2.waitKey(0)
