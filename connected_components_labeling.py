#import modules
from skimage.filters import threshold_local
from skimage import measure
import argparse
import cv2
import numpy as np

#set up argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
    help = "Path to the image file")
args = vars(ap.parse_args())

# Load the image
image = cv2.imread(args["image"])

#extract the value component from HSV color space and apply
#adaptive thresholding to reveal the chars on the plate
V = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))[2]
T = threshold_local(V, 29, offset=15, method="gaussian")
thresh = (V < T).astype("uint8") * 255

cv2.imshow("orig", image)
cv2.imshow("thresh", thresh)

#perform connected components analysis on the threshholded images
# and init the mark to holdy only the "large" components
#perform the check with 8 connectivity (neighbors = 8)
#pixels with 0 value should be disgarded (background)
#label method returns labels, a numpy array with the same
#dimensionality as the threshed image
#each x,y in labels is either 0 (background), or
# > 0 (a connected components)
labels = measure.label(thresh, neighbors=8, background=0)
mask = np.zeros(thresh.shape, dtype="uint8")
print("[INFO] found {} blobls".format(len(np.unique(labels))))

#loop over the components in labels
for (i, label) in enumerate(np.unique(labels)):
    #ignore if it is a background label
    if label == 0:
        print("[INFO] label: 0 (background)")
        continue

    #otherwise, construct the label mask to display only
    #connect components for the current label
    print("[INFO] label: {} (foreground)".format(i))
    labelMask = np.zeros(thresh.shape, dtype="uint8")
    #set all x,y in labelMask that belong to current label in labels
    #two WHITE (255)
    labelMask[labels == label] = 255
    numPixels = cv2.countNonZero(labelMask)

    #if the number of pixels in the component is large
    #add it to our mask of 'large' blobls
    if numPixels > 300 and numPixels < 1500:
        mask = cv2.add(mask, labelMask)

    #show the label mask
    cv2.imshow("label", labelMask)
    cv2.waitKey(0)

cv2.imshow("large blobs", mask)
cv2.waitKey(0)
