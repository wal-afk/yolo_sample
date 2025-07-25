# %%
from glob import glob
import csv
import pandas as pd
import os

class LabelCounter:
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.label_list = self.read_label_def(f"{root_dir}/labels.txt")

    def _count_label(self, dir_path):
        label_count = {}
        for path in glob(f"{dir_path}/*.txt"):
            with open(path) as f:
                reader = csv.reader(f, delimiter=" ")
                for line in reader:
                    label = self.label_list[int(line[0])]
                    if label not in label_count:
                        label_count[label] = 0
                    label_count[label] += 1
        return label_count

    @staticmethod
    def read_label_def(path):
        with open(path) as f:
            return [line.strip() for line in f.readlines()]

    def print_labels_count(self):
        subdirs = ["train", "val"]
        data = [
            self._count_label(f"{self.root_dir}/labels/{subdir}") for subdir in subdirs
        ]
        df = pd.DataFrame(data, index=subdirs)
        print(df.T.sort_index())

DIR = os.path.dirname(__file__)
LabelCounter(f"{DIR}/dataset/sample1").print_labels_count()

# %%
