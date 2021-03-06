# USAGE
#  python prepare_image_dataset.py --dataset images/caltech_web_faces \
#	--output output/faces/faces_dataset.txt

# import packages
from imutils import encodings
from imutils import paths
import progressbar
import argparse
import uuid
import cv2

# set up argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help = "path to the directory that holds the images ")
ap.add_argument("-o", "--output", required=True, help = "path to the output file for use on HDFS")
args = vars(ap.parse_args())

# grab the list of image paths in the dataset dir and open the output file for writing
imagePaths = list(paths.list_images(args["dataset"]))
f = open(args["output"], "w")

# init the progress bar
widgets = ["Building Dataset: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(imagePaths), widgets=widgets).start()


# loop over the images in the dataset directory
for (i, path) in enumerate(imagePaths):
    # construct a unique ID for the image, encode the imaage as a string
    # and write the data to a flattened CSV file
    imageID = str(uuid.uuid4())
    image = encodings.base64_encode_image(cv2.imread(path))
    f.write("{}\t{}\t{}\n".format(imageID, path, image))
    pbar.update(i)

# close the output file
pbar.finish()
f.close()
