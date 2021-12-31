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
cv2.imshow("orig", image)

#convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#compute gradients about the x and y axis
gX = cv2.Sobel(gray, ddepth=cv2.CV_64F, dx=1, dy=0)
gY = cv2.Sobel(gray, ddepth=cv2.CV_64F, dx=0, dy=1)

#convert gX and gY values from float to 8 bit unsigned int
#this is so openCV functions can utilize them
gX = cv2.convertScaleAbs(gX)
gY = cv2.convertScaleAbs(gY)

#combine sobel X and Y representations into a single image
sobel_combined = cv2.addWeighted(gX, 0.5, gY, 0.5, 0)

#display the output images
cv2.imshow("sobel x", gX)
cv2.imshow("sobel y", gY)
cv2.imshow("sobel combo", sobel_combined)
cv2.waitKey(0)
