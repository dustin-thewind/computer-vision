#import modules
import argparse
import cv2

#set up argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
    help = "Path to the image file")
args = vars(ap.parse_args())

# Load the image,convert and blur
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

#show the images
cv2.imshow("orig", image)
cv2.imshow("blurred", blurred)

#configure canny images with lower and upper thresholds
wide = cv2.Canny(blurred, 10, 200)
mid = cv2.Canny(blurred, 30, 150)
tight = cv2.Canny(blurred, 240, 250)

#show the edge maps
cv2.imshow("wide", wide)
cv2.imshow("mid", mid)
cv2.imshow("tight", tight)
cv2.waitKey(0)
