# import packages
import numpy as np

class RedisQueue:
    def __init__(self, redisDB):
        # store the redis DB object
        self.redisDB = redisDB

    def add(self, imageIdx, hist):
        # init the redist pipeline
        p = self.redisDB.pipeline()

        # loop over all non-zero entries for the hist creating
        # a visual word -> document record for each
        # visual word in the hist
        for i in np.where(hist > 0)[0]:
            p.rpush("vw:{}".format(i), imageIdx)

            # execute the pipeline
            p.execute()

    def finish(self):
        # save the state of the redis DB
        self.redisDB.save()
