#import modules
import argparse
import cv2

#set up argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
    help = "Path to the image file")
args = vars(ap.parse_args())

# Load the image and show it
image = cv2.imread(args["image"])
#convert the image to greyscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#build rectangular kernel and apply blackhat operation
#which enables us to find dark regions on a light background
#create a 5x13 rectangle for a license plate
#since we know that license plates are usually ~5 wider than tall
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)

#tophate (also called whitehat) allows us to find
# light regions on a dark background
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)

#show output images
cv2.imshow("original", image)
cv2.imshow("blackhat", blackhat)
cv2.imshow("tophat", tophat)
cv2.waitKey(0)
