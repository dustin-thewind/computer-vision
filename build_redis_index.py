# USAGE
# python build_redis_index.py --bovw-db output/bovw.hdf5

# import packages
from __future__ import print_function
from prod_cbir.db import RedisQueue
from redis import Redis
import argparse
import h5py

# set up argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-b", "--bovw-db", required=True, help = "Path to BOVW database")
args = vars(ap.parse_args())

# connect to redis, init the redis queue, and open the BOVW db
redisDB = Redis(host="localhost", port=6379, db=0)
rq = RedisQueue(redisDB)
bovwDB = h5py.File(args["bovw_db"], mode="r")

# loop over the entries in the BOVW
for (i, hist) in enumerate(bovwDB["bovw"]):
    # check to see if progress should be displayed
    if i > 0 and i % 10 == 0:
        print("[PROGRESS] processed {} entires".format(i))

        # add the image index and hist to the redis server
        rq.add(i, hist)

# close the bovw DB and finish the indexing processing
bovwDB.close()
rq.finish()
