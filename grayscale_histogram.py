#import modules
from matplotlib import pyplot as plt
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

#construct a grayscale histogram
#images - image we want to compute hist for, wrap as a list
#channels - list of indexes, specify the index of the channel
#we want to compute a hist for. to compute for a grayscale
#the list would be [0], for RGB [0,1,2]
#mask - supply a mask, if supplied hist will be for masked region only
#histSize - number of bins we want to use when computing the hist
#this is a list, one for each channel we're computing
#bin sizes do not have to be the same, 32 bins per chan would be [32,32,32]
#ranges - range of possible pixel values. calc
hist = cv2.calcHist([image],[0], None, [256], [0, 256])

#matplotlib expects RGB images, so we convert and display the image
#with matplotlib to avoid GUI conflicts/errors
plt.figure()
plt.axis("off")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))

#plot the historgram
plt.figure()
plt.title("Grayscale Histogram")
plt.xlabel("bins")
plt.ylabel("# of pix")
plt.plot(hist)
plt.xlim([0, 256])

#normalize the Histogram
hist /= hist.sum()

#plot the normalized hist
plt.figure()
plt.title("Grayscale Histogram Normalized")
plt.xlabel("bins")
plt.ylabel("% of pix")
plt.plot(hist)
plt.xlim([0, 256])
plt.show()
