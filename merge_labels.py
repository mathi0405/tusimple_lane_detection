import os

files = [
    "data/TUSimple/train_set/label_data_0313.json",
    "data/TUSimple/train_set/label_data_0531.json",
    "data/TUSimple/train_set/label_data_0601.json"
]

with open("data/train_label.json", "w") as outfile:
    for fname in files:
        with open(fname) as infile:
            for line in infile:
                outfile.write(line)