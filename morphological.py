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
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.waitKey(0)

#apply a series of erosions
for i in range(0, 3):
    eroded = cv2.erode(gray.copy(), None, iterations=i + 1)
    cv2.imshow("eroded {} times".format(i + 1), eroded)
    cv2.waitKey(0)

#close all windows and clean up
cv2.destroyAllWindows()
cv2.imshow("orig", image)

#apply a series of dilations
for i in range(0, 3):
    dilated = cv2.dilate(gray.copy(), None, iterations=i + 1)
    cv2.imshow("dilated {} times".format(i + 1), dilated)
    cv2.waitKey(0)

#close all windows and clean up
cv2.destroyAllWindows()
cv2.imshow("orig", image)
#initialize list of kernel sizes
#that will be applied to the image
kernelSizes = [(3, 3), (5, 5), (7, 7)]

#loop over kernels and apply 'opening' operation to image
#recall that opening is erosion followed by dilation
for kernelSize in kernelSizes:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
    opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    cv2.imshow("Opening ({}, {})".format(kernelSize[0], kernelSize[1]), opening)
    cv2.waitKey(0)

#close all windows and clean up
cv2.destroyAllWindows()
cv2.imshow("orig", image)

#loop over kernels and apply 'closing' operation to image
#recall that closing is dilation followed by erosion
for kernelSize in kernelSizes:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
    closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    cv2.imshow("Closing ({}, {})".format(kernelSize[0], kernelSize[1]), closing)
    cv2.waitKey(0)

#close all windows and clean up
cv2.destroyAllWindows()
cv2.imshow("orig", image)

#loop over kernels and apply 'closing' operation to image
#recall that closing is dilation followed by erosion
for kernelSize in kernelSizes:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
    gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
    cv2.imshow("Gradient ({}, {})".format(kernelSize[0], kernelSize[1]), gradient)
    cv2.waitKey(0)
