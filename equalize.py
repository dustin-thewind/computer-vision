#import modules
import argparse
import cv2

#set up argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
    help = "Path to the image file")
args = vars(ap.parse_args())

# Load the image
image = cv2.imread(args["image"])
#convert to grayscale
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#apply his equalition to stretch the contrast
eq = cv2.equalizeHist(image)

#print RGB value of px at y = 272, x = 146
pixel = eq[272, 146]
print("Pixel at (y272, x146) - {}".format(pixel))
cv2.waitKey(0)

#show the images
cv2.imshow("orig", image)
cv2.imshow("hist equalization", eq)
cv2.waitKey(0)
