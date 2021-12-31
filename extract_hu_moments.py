#usage
#python extract_hu_moments.py -i imagefile

#import packages
import cv2
import imutils
import argparse

#set up argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
    help = "Path to the image file")
args = vars(ap.parse_args())

# Load the image
image = cv2.imread(args["image"])

#convert to grayscale
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#compute the hu moments feature vector for the entire image and display it
moments = cv2.HuMoments(cv2.moments(image)).flatten()
print("orig moments: {}".format(moments))
cv2.imshow("image", image)
cv2.waitKey(0)

#find the contours of the three planes in the image
cnts = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

#loop over each contour
for (i, c) in enumerate(cnts):
    #extract the ROI from the image and
    #compute the hu moments feature for the the ROI
    (x, y, w, h) = cv2.boundingRect(c)
    roi = image[y:y + h, x:x + w]
    moments = cv2.HuMoments(cv2.moments(roi)).flatten()

    #show the moments and ROI
    print("moments for plane #{}: {}".format(i + 1, moments))
    cv2.imshow("ROI #{}".format(i + 1), roi)
    cv2.waitKey(0)
