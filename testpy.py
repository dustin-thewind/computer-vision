import numpy as np
import cv2
import sklearn
from scipy.spatial import distance as dists

A = [0.81, 0.84, 0.31, 0.13, 0.96, 0.48, 0.58, 0.65]
B = [0.82, 0.31, 0.50, 0.38, 0.74, 0.59, 0.62, 0.94]

def histogram_intersection(H1, H2):
    return np.sum(np.minimum(H1, H2))

def chi2_distance(histA, histB, eps=1e-10):
    return 0.5 * np.sum(((histA - histB) ** 2) / (histA + histB + eps))

hamming = dists.hamming(A, B)

cosine = dists.cosine(A, B)

#result = histogram_intersection(A, B)
print("result", cosine)
