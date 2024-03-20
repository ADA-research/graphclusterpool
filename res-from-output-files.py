import os
import sys

import numpy as np

dir = sys.argv[1]

files = [x for x in os.listdir(dir) if x.endswith(".out")]

res = []
res_name = []
for file in files:
    line = open(dir + "/" + file, 'r').readlines()[-1]
    if line.strip().startswith("[!]"):
        res.append(float(line.split(" ")[-1]))
        res_name.append(file)

print(f"[{len(res)}] Result: {np.mean(res)} +/ {np.std(res)}")

#res.sort()
if len(sys.argv) > 2:
    for f, r in zip(res_name, res):
        print(r, '\t', f)
else:
    res.sort()
    print(res)