# import packages
from .baseindexer import BaseIndexer
from scipy import sparse
import numpy as np
import h5py

class BOVWIndexer(BaseIndexer):
    def __init__(self, fvectorSize, dbPath, estNumImages=500, maxBufferSize=500, dbResizeFactor=2,
        verbose=True):
        # call the parent constructor
        super(BOVWIndexer, self).__init__(dbPath, estNumImages=estNumImages,
            maxBufferSize=maxBufferSize, dbResizeFactor=dbResizeFactor,
            verbose=verbose)

        # open the hdf5 DB for writing, init the datasets within the group
        # the BOVW buffer list, and the BOVW index into the dataset
        self.db = h5py.File(self.dbPath, mode="w")
        self.bovwDB = None
        self.bovwBuffer = None
        self.idxs = {"bovw": 0}

        # store the feature vector size of the BOVW, then
        # init the document frequency coutns to be accumulated
        # and actual total # of images in the DB
        self.fvectorSize = fvectorSize
        self._df = np.zeros((fvectorSize,), dtype="float")
        self.totalImages = 0

    def add(self, hist):
        # udpate the BOVW buffer and the document frequency counts
        self.bovwBuffer = BaseIndexer.featureStack(hist, self.bovwBuffer,
            stackMethod=sparse.vstack)
        self._df[np.where(hist.toarray()[0] > 0)] += 1

        # check to see if we have reached the max buffer size
        if self.bovwBuffer.shape[0] >= self.maxBufferSize:
            # if the DB's have not been created yet, create them
            if self.bovwDB is None:
                self._debug("initial buffer full")
                self._createDatasets()

            # write the buffers to file
            self._writeBuffers()

    def _writeBuffers(self):
        # only writ the buffer if thre are entries in the buffer
        if self.bovwBuffer is not None and self.bovwBuffer.shape[0] > 0:
            # write the BOVW buffer to file, increment the index
            # and reset the buffer
            self._writeBuffer(self.bovwDB, "bovw", self.bovwBuffer, "bovw",
                sparse=True)
            self.idxs["bovw"] += self.bovwBuffer.shape[0]
            self.bovwBuffer = None

    def _createDatasets(self):
        #grab the feature veector size and create the dataset
        self._debug("creating datasets...")
        self.bovwDB = self.db.create_dataset("bovw",
            (self.estNumImages, self.fvectorSize),
            maxshape=(None, self.fvectorSize), dtype="float")

    def finish(self):
        # if the DB's have not been init'd then the original
        # buffers were never filled
        if self.bovwDB is None:
            self._debug("minimum init buffer not reached", msgType="[WARN]")
            self._createDatasets()

        # write any unempty buffers to file
        self._debug("writing un-empty buffers...")
        self._writeBuffers()

        # compact datasets
        self._debug("compacting datasets...")
        self._resizeDataset(self.bovwDB, "bovw", finished=self.idxs["bovw"])

        # store the total number of images in the dataset and close the DB
        self.totalImages = self.bovwDB.shape[0]
        self.db.close()

    def df(self, method=None):
        if method == "idf":
            # compute the inverted document frequency
            return np.log(self.totalImages / (1.0 + self._df))

        # otherwise a valid method was supplied, so return the raw
        # document frequency counts
        return self._df
