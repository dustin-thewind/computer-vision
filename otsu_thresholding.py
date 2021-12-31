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

#convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#apply gaussian blur with a 7x7 kernel
blurred = cv2.GaussianBlur(gray, (7, 7), 0)

#use Otsu automatic thresholding
#automatically determines the best threshold 'T'
(T, thresh_inv) = cv2.threshold(blurred, 0, 255,
    cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
cv2.imshow("treshold", thresh_inv)
print("Otsu T Value:{}".format(T))

#visualize only the masked regions in the image
cv2.imshow("output", cv2.bitwise_and(image, image, mask=thresh_inv))
cv2.waitKey(0)
