#import modules
import argparse
import cv2
import numpy as np
import imutils

#set up argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
    help = "Path to the image file")
args = vars(ap.parse_args())

# Load the image
image = cv2.imread(args["image"])
#convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#show original image
cv2.imshow("orig", image)

#find all contours in the image and draw all contours on the image
#findContours is destructive since it manipulates the image
#clone the image using copy() if it is needed again
#RETR_LIST ensures all contours are returned
#CHAIN_APPROX_SIMPLE compresses horiz, vert and diag segments into
#only endpoints, which reduces memory consumption without accuracy loss
#findContours returns a tuple of values
#first value is the image itself (in opencv3, opencv2.4 & 4 dont' place image in the tuple)
#second value is the contours themselves (in opencv3, in opencv2.4 & 4 it is the first value)
#third value is heirarchy of the contours  (in opencv2.4 & 4 it is the second value)
cnts = cv2.findContours(gray.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
clone = image.copy()
#draw the contours
#first param is the image we want to draw on
#second param is list of contours from findContours
#third param is the index of the countour inside cnts
#a value of -1 instructs drawContours to draw all contours
#fourth param is the color of the contour we want to draw
#fifth param is thickness of the contour line
cv2.drawContours(clone, cnts, -1, (0, 255, 0), 2)
print("found {} contours".format(len(cnts)))
cv2.imshow("all contours", clone)
cv2.waitKey(0)

clone2 = image.copy()
cv2.destroyAllWindows()
#loop over the contours and draw each
for (i, c) in enumerate(cnts):
    print("drawing contour #{}".format(i + 1))
    cv2.drawContours(clone2, [c], -1, (0, 255, 0), 2)
    cv2.imshow("single contour", clone2)
    cv2.waitKey(0)

clone3 = image.copy()
cv2.destroyAllWindows()

#find only external contours by specifying the
#RETR_EXTERNAL value to findContours
cnts = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cv2.drawContours(clone3, cnts, -1, (0, 255, 0), 2)
print("found {} external contours".format(len(cnts)))
cv2.imshow("all external contours", clone3)
cv2.waitKey(0)

clone4 = image.copy()
cv2.destroyAllWindows()

for c in cnts:
    #build a mask by drawing only the current contour
    mask = np.zeros(gray.shape, dtype="uint8")
    cv2.drawContours(mask, [c], -1, 255, -1)
    #show the images
    cv2.imshow("image", image)
    cv2.imshow("mask", mask)
    cv2.imshow("image + mask", cv2.bitwise_and(image, image, mask=mask))
    cv2.waitKey(0)
