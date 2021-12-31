#1.9
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

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#apply gaussian blur with a 7x7 kernel
blurred = cv2.GaussianBlur(gray, (7, 7), 0)

#basic thresholding using inversing
#first param is the image we want to threshold
#second param - is threshold check
#if pixel is > threshold we set to black, otherwise we set to white
#third param is output value of thresholding
#any pixel greater than threshold then we set to the output value
(T, thresh_inv) = cv2.threshold(blurred, 95, 255, cv2.THRESH_BINARY_INV)
cv2.imshow("thresh binary inv", thresh_inv)

#normal thresholding
(T, thresh) = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
cv2.imshow("thresh binary", thresh)

#visualize only the masked regions in the image
cv2.imshow("output", cv2.bitwise_and(image, image, mask=thresh_inv))
cv2.waitKey(0)
