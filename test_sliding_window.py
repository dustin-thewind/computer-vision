# USAGE
# python test_sliding_window.py -i image1.jpg -s 1.5

# import packages
from obj_detect.object_detector.helpers import sliding_window
from obj_detect.object_detector.helpers import pyramid
import argparse
import time
import cv2

# set up argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
ap.add_argument("-w", "--width", required=True, type=int, help="Width of sliding window")
ap.add_argument("-t", "--height", required=True, type=int, help="Height of sliding window")
ap.add_argument("-s", "--scale", type=float, default=1.2, help="scale factor size")
args = vars(ap.parse_args())

# load the image and unpack the command line args
image = cv2.imread(args["image"])
(winW, winH) = (args["width"], args["height"])

# set up a counter for the number of windows
winCount = 0

# loop over the image test_pyramid
for layer in pyramid(image, scale=args["scale"]):
    # loop over the sliding window for each layer of the pyramid
    for (x, y, window) in sliding_window(layer, stepSize=10, windowSize=(winW, winH)):
        # if the current window does not meet our desired size, ignore it
        if window.shape[0] != winH or window.shape[1] != winW:
            continue

        # THIS IS WHERE WE WOULD PROCESS THE WINDOW, EXTRACT HOG FEATURES, AND
		# APPLY A MACHINE LEARNING CLASSIFIER TO PERFORM OBJECT DETECTION

        #increment the counter by 1 for each window
        winCount += 1

        # draw the window
        clone = layer.copy()
        cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
        cv2.imshow("Window", clone)

        #visualize the window
        #cv2.waitKey(1)
        #time.sleep(0.025)

# print out the number of total windows
print("winCount: {}".format(winCount))
