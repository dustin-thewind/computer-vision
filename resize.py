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

#create a new image size of width 150px
#calculate aspect ratio of original image to new image
#use width of new image as factor
#using formula > AR = w/h
#recall imageshape[0] = height and imageshape[1] = width
ratio = 150.0 / image.shape[1]
dim = (150, int(image.shape[0] * ratio))

#resize the image using the calculated aspect ratio
resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
cv2.imshow("Resized Width", resized)
cv2.waitKey(0)

#calculate aspect ratio of original image to new image
#use height of new image as factor
#using formula > AR = w/h
#recall imageshape[0] = height and imageshape[1] = width
ratio = 50 / image.shape[0]
dim = (int(image.shape[1] * ratio), 50)

#resize the image using the calculated aspect ratio
resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
cv2.imshow("Resized Height", resized)
cv2.waitKey(0)

#resize using helper function from imutils
resized = imutils.resize(image, width=100)
cv2.imshow("Resized via Function", resized)
cv2.waitKey(0)

width=image.shape[1]
#what if we wanted to see a 27% increase?
#float_w = float(width * 1.27)
# this doesn't work
resized = imutils.resize(image, width * 1.27, inter=cv2.INTER_NEAREST)
#this works
resized = imutils.resize(image, int(width * 1.27), inter=cv2.INTER_NEAREST)

cv2.imshow("resized with float", resized)
cv2.waitKey(0)

#for the quiz, question #3
ratio1 = 1200.0 / image.shape[1]
dim1 = (1200, int(image.shape[0] * ratio1))
#print(dim1)

#resize the image using the calculated aspect ratio
resized4 = cv2.resize(image, dim1, interpolation=cv2.INTER_CUBIC)
cv2.imshow("Resized Width", resized4)
cv2.waitKey(0)

#get the pixel at y = 367, x = 170
(b, g, r) = resized4[367, 170]
print("Pixel at (0, 0) - Red: {r}, Green: {g}, Blue: {b}"
    .format(r=r, g=g, b=b))

#test all the different interpolation methods
#create a list of all the methods
methods = [
 ("cv2.INTER_NEAREST", cv2.INTER_NEAREST),
 ("cv2.INTER_LINER", cv2.INTER_LINEAR),
 ("cv2.INTER_AREA", cv2.INTER_AREA),
 ("cv2.INTER_CUBIC", cv2.INTER_CUBIC),
 ("cv2.INTER_LANCZOS4", cv2.INTER_LANCZOS4)]

'''
#iterate through the list
for (name, method) in methods:
    #increase the size of the image by 3x
    #using the current interpolation method
    width=image.shape[1]
    resized3 = imutils.resize(image, width=image.shape[1] * 3, inter=method)
    cv2.imshow("method: {}".format(name), resized)
    (b, g, r) = resized3[367, 170]
    print("Pixel at (0, 0) - Red: {r}, Green: {g}, Blue: {b}"
        .format(r=r, g=g, b=b))
    cv2.waitKey(0)
'''
