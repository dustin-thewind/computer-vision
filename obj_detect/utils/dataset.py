# import packages
import h5py
import numpy as np

def dump_dataset(data, labels, path, datasetName, writeMethod="w"):
    # open the db, create the dataset, write the data and labels to the dataset
    # and close the DB
    db = h5py.File(path, writeMethod)
    dataset = db.create_dataset(datasetName, (len(data), len(data[0]) + 1), dtype="float")
    dataset[0:len(data)] = np.c_[labels, data]
    db.close()

def load_dataset(path, datasetName):
    # open the DB, grab the labels and data, then close the dataset
    db = h5py.File(path, "r")
    (labels, data) = (db[datasetName][:, 0], db[datasetName][:, 1:])
    db.close()

    # return a tuple of the data and labels
    return (data, labels)
