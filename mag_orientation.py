#import modules
import argparse
import cv2
import numpy as np

#set up argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
    help = "Path to the image file")
ap.add_argument("-l", "--lower-angle", type=float, default=175.0,
    help="lower orientation angle")
ap.add_argument("-u", "--upper-angle", type=float, default=180.0,
    help="upper orientation angle")
args = vars(ap.parse_args())

# Load the image and show it
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("orig", image)
cv2.waitKey(0)

#compute gradients alont x, y axis
gX = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
gY = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

#compute gradient magnitude and orientation
mag = np.sqrt((gX ** 2) + (gY ** 2))
orientation = np.arctan2(gY, gX) * (180 / np.pi) % 180
orientation2 = np.arctan2(319, 23) * (180 / np.pi) % 180

print(orientation2)

#find all pixels that are within the upper and low angle boundaries
idxs = np.where(orientation >= args["lower_angle"], orientation, -1)
idxs = np.where(orientation <= args["upper_angle"], idxs, -1)
mask = np.zeros(gray.shape, dtype="uint8")
mask[idxs > -1] = 255

cv2.imshow("mask", mask)
cv2.waitKey(0)
