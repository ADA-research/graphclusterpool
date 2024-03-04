from pathlib import Path

pth = Path("results/REDDIT-BINARY")

files = [x for x in pth.iterdir() if x.suffix == ".out"]

for file in files:
    lastline = file.open("r").readlines()[-2]
    print(file)
    print(lastline)