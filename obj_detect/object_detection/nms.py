# import packages
import numpy as np

def non_max_suppression(boxes, probs, overLapThresh):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []

	# if the bounding boxes are ints, convert to float
	# we do this because we are doing lots of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype(float)

	# init the list of picked indexes
	pick = []

	# grab the coordinates of the bounding boxes
	x1 = boxes[:, 0]
	y1 = boxes[:, 1]
	x2 = boxes[:, 2]
	y2 = boxes[:, 3]

	# compute the area of the bounding boxes and sort the bounding boxes by
	# their associated probabilities
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(probs)

	# keep looping while some indexes still remain on the indexes list
	while len(idxs) > 0:
		# grab the last index in the indices list, add the index
		# value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)

		# find the largest (x, y) coords for the start of the bounding box
		# and smallest x,y coords for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.maximum(x2[i], x2[idxs[:last]])
		yy2 = np.maximum(y2[i], y2[idxs[:last]])

		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)

		# compute the the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]

		# delete all indexes from the index list that have overLap
		# greater than the provided overlap threshhold
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overLapThresh)[0])))

	# return only the boudning boxes that were pickled
	return boxes[pick].astype("int")
