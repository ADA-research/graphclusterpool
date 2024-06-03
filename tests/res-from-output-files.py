import os
import sys

import numpy as np

dir = sys.argv[1]

files = [x for x in os.listdir(dir) if x.endswith(".out")]

res = []
res_name = []
for file in files:
    lines = open(dir + "/" + file, 'r').readlines()
    line = lines[-1].strip()
    if line.startswith("[!]"):
        res.append(float(line.split(" ")[-1]))
        res_name.append(file)
    else:
        if line.startswith("Validation"):
            line = lines[-2].strip()
        print("No res in: ", file, ":", line.strip())

print(f"[{len(res)}] Result: {np.mean(res)} +/ {np.std(res)}")

if len(sys.argv) > 2:
    sorted_lists = [[x, y] for y, x in sorted(zip(res, res_name), key=lambda pair: pair[0])]
    for v in sorted_lists:
        print(v[1], '\t', v[0])
else:
    res.sort()
    print(res)