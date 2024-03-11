import os
import sys

import numpy as np

dir = sys.argv[1]

files = [x for x in os.listdir(dir) if x.endswith(".out")]

res = []

for file in files:
    line = open(dir + "/" + file, 'r').readlines()[-1]
    if line.strip().startswith("[!]"):
        res.append(float(line.split(" ")[-1]))

print(f"Result: {np.mean(res)} +/ {np.std(res)}")
res.sort()
print(res)