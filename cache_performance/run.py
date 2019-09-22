from embedding_method.mapper import MemFetcher
import time
from cache_performance.rand_data import create_mmap,get_random_dict
import random
import matplotlib.pyplot as plt
import tracemalloc
import sys

results = []
x = []
mmap_t = []
cache_t = []

cache_m = []
mmap_m = []

for i in range(10):
    size = 10000*(i+1)
    x.append(size)
    data = get_random_dict(size)
    create_mmap(data)


    keys = list(data.keys())
    random.shuffle(keys)
    print("cache",sys.getsizeof(data))
    start_time = time.time()
    for key in keys:
        value = data.get(key)
    lookup_time = time.time() - start_time

    mmap = MemFetcher("lookup.json","data.jsonl")
    data = None

    print("mmap", sys.getsizeof(mmap))

    start_time = time.time()
    for key in keys:
        value = mmap(key)
    mmap_time = time.time() - start_time


    #print("size:",size)
    #print("lookup: ",lookup_time)
    #print("mmap: ",mmap_time)

    mmap_t.append(mmap_time)
    cache_t.append(lookup_time)



plt.plot(x,mmap_t)
plt.plot(x,cache_t)
plt.legend(["mmap","cache"])

plt.xlabel("#Data points")
plt.ylabel("Lookup time (seconds)")
plt.title("Lookup time in cache vs memory map")

plt.savefig("cachemmap.pdf")

plt.show()