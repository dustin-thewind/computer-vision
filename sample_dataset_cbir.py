# USAGE
# python sample_dataset_cbir.py -o output/data 

# import packages
from imutils import paths
import random
import shutil
import argparse
import glob
import os


# set up argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
    help = "Path to the directory of image classes")
ap.add_argument("-o", "--output", required=True,
    help = "Path to the output directory to store training and testing images")
ap.add_argument("-t", "--training-size", type=float, default=0.75,
    help = "% of images to be used for training")
args = vars(ap.parse_args())

# if the output directory exists, delete it
if os.path.exists(args["output"]):
    shutil.rmtree(args["output"])

# create the output directories
os.makedirs(args["output"])
os.makedirs("{}/training".format(args["output"]))
os.makedirs("{}/testing".format(args["output"]))

# loop over the image classes in the input directory
for labelPath in glob.glob(args["input"] + "/*"):
    # extract the label from the path and create the sub-directories for the label
    # in the output directory
    label = labelPath[labelPath.rfind("/") + 1:]
    os.makedirs("{}/training/{}".format(args["output"], label))
    os.makedirs("{}/testing/{}".format(args["output"], label))

    # grab the image paths for the current label and shuffle them
    imagePaths = list(paths.list_images(labelPath))
    random.shuffle(imagePaths)
    i = int(len(imagePaths) * args["training_size"])

    # loop over the randomly sampled training paths and copy them
    # to the appropriate output directory
    for imagePath in imagePaths[:i]:
        filename = imagePath[imagePath.rfind("/") + 1:]
        shutil.copy(imagePath, "{}/training/{}/{}".format(args["output"], label, filename))

    # loop over the randomly sampled testing paths and copy them
    # to the appropriate output directory
    for imagePath in imagePaths[i:]:
        filename = imagePath[imagePath.rfind("/") + 1:]
        shutil.copy(imagePath, "{}/testing/{}/{}".format(args["output"], label, filename))
