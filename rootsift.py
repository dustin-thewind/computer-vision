# import packages
import numpy as np
import cv2
import imutils

class RootSIFT:
    def __init__(self):
        # init the SIFT feature extractor for OCV 2.4
        if imutils.is_cv2():
            self.extractor = cv2.DescriptorExtractor_create("SIFT")

        # otherwise use the OCV 3+ version of the SIFT extractor
        else:
            self.extractor = cv2.xfeatures2d.SIFT_create()

    def compute(self, image, kps, eps=1e-7):
        # compute SIFT descriptors for OCV 2.4
        if imutils.is_cv2:
            (kps, descs) = self.extractor.compute(image, kps)

        # otherwise, compute SIFT descriptors for OCV 3+
        else:
            (kps, descs) = self.extractor.detect

        # if there are no keypoints or descriptors, return an empty tuple
        if len(kps) == 0:
            return ([], None)

        # apply the Hellinger kernel by L1-normalizing and taking the
        # square root
        descs /= (descs.sum(axis=1, keepdims=True) + eps)
        descs = np.sqrt(descs)

        #return a tuple of the keypoints and descriptors
        return (kps, descs)
        
