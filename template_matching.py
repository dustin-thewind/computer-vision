# USAGE
# python template_matching.py --source source_01.jpg --template template.jpg

# import packages
import argparse
import cv2

# set up arg parser
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--source", required=True, help="path to the source img")
ap.add_argument("-t", "--template", required=True, help="path to the template img")
args = vars(ap.parse_args())

# load the source and template image
source = cv2.imread(args["source"])
template = cv2.imread(args["template"])
(tempH, tempW) = template.shape[:2]

# find the template in the source image
result = cv2.matchTemplate(source, template, cv2.TM_CCOEFF)
(minVal, maxVal, minLoc, (x, y)) = cv2.minMaxLoc(result)

# draw the bounding box on the source image
cv2.rectangle(source, (x, y), (x + tempW, y + tempH), (0, 255, 0), 2)
print("x, y coord {}, {}".format(x, y))

# show the images
cv2.imshow("source", source)
cv2.imshow("template", template)
cv2.waitKey(0)
