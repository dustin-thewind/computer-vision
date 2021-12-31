#import modules
from skimage.filters import threshold_local
import argparse
import cv2

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

#apply gaussian blur with a 5x5 kernel
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

#use adaptive thresholding to examine neighborhoods of pixels
#and adaptively threshold each neighboorhood
#calculate the mean value of a 25px neighborhood area
#and threshold based on that value
#constant C is subtracted from the mean calculation (15 below)
thresh = cv2.adaptiveThreshold(blurred, 255,
    cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 15)
cv2.imshow("opencv mean thresh", thresh)

#adaptive thresholding using scikit
T = threshold_local(blurred, 29, offset=5, method="gaussian")
thresh = (blurred < T).astype("uint8") * 255
cv2.imshow("scikit mean thresh", thresh)
cv2.waitKey(0)
