# Data shuffling

import pickle
import matplotlib.pyplot as plt
import random
import time

from helperfuncs import *

def read_data(phase, num):
    data_list = pickle.load(open(f"D:/parakeet/evaluation_database/{phase}_data_{num}.chess", "rb"))
    return data_list


for P in [0, 1, 2]:
    if P == 0:
        st = "draw"
        num = 500
        rmax = 527
    elif P == 1:
        st = "winning"
        num = 800
        rmax = 395
    elif P == 2:
        st = "adv"
        num = 900
        rmax = 461

    print(st)
    s = time.time()
    l = 0

    while l < num:
        r0, r1, r2, r3 = random.sample(range(0, rmax), 4)

        try:
            p0, f0, e0 = read_data(st, r0)
            p1, f1, e1 = read_data(st, r1)
            p2, f2, e2 = read_data(st, r2)
            p3, f3, e3 = read_data(st, r3)
        except:
            continue
        p = list(p0) + list(p1) + list(p2) + list(p3)
        f = list(f0) + list(f1) + list(f2) + list(f3)
        e = list(e0) + list(e1) + list(e2) + list(e3)
        zipped = list(zip(p, f, e))
        random.shuffle(zipped)
        p, f, e = zip(*zipped)
        p0, f0, e0 = p[:65536], f[:65536], e[:65536]
        p1, f1, e1 = p[65536:131072], f[65536:131072], e[65536:131072]
        p2, f2, e2 = p[131072:196608], f[131072:196608], e[131072:196608]
        p3, f3, e3 = p[196608:262144], f[196608:262144], e[196608:262144]
        pickle.dump([p0, f0, e0], open(f"D:/parakeet/evaluation_database/{st}_data_{r0}.chess", "wb"))
        pickle.dump([p1, f1, e1], open(f"D:/parakeet/evaluation_database/{st}_data_{r1}.chess", "wb"))
        pickle.dump([p2, f2, e2], open(f"D:/parakeet/evaluation_database/{st}_data_{r2}.chess", "wb"))
        pickle.dump([p3, f3, e3], open(f"D:/parakeet/evaluation_database/{st}_data_{r3}.chess", "wb"))
        l += 1
        print(l, time.time() - s, r0, r1, r2, r3)
    print(st, time.time() - s)