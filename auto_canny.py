#import modules
import argparse
import cv2
import imutils

#set up argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
    help = "Path to the image file")
args = vars(ap.parse_args())

# Load the image,convert and blur
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (3, 3), 0)

#apply auto canny edge detection using a
#wide threshold, tight threshold and auto-threshold
wide = cv2.Canny(blurred, 10, 200)
tight = cv2.Canny(blurred, 225, 250)
auto = imutils.auto_canny(blurred)

#show the images
cv2.imshow("orig", image)
cv2.imshow("wide", wide)
cv2.imshow("tight", tight)
cv2.imshow("auto", auto)
cv2.waitKey(0)
