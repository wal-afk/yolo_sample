# %%
from glob import glob
import csv
import pandas as pd
import os


class LabelCounter:
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.label_list = self.read_label_def(f"{root_dir}/labels.txt")

    def _count_label(self, label_dir: str, file_name_list):
        label_count = {}
        for file_name in file_name_list:
            paths = glob(f"{label_dir}/**/{file_name}")
            if len(paths) == 0:
                print(f"ignore: {file_name} not found")
            elif len(paths) >= 2:
                print(f"ignore: {file_name} found more than 1")
            else:
                with open(paths[0]) as f:
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
        image_file_names = {}
        for image_path in glob(f"{self.root_dir}/images/**/*.*", recursive=True):
            group = os.path.relpath(
                os.path.dirname(image_path), start=f"{self.root_dir}/images"
            )
            if group not in image_file_names:
                image_file_names[group] = []
            image_file_names[group].append(os.path.basename(image_path))

        data = {
            subdir: self._count_label(
                f"{self.root_dir}/labels",
                [
                    f"{os.path.splitext(file_name)[0]}.txt"
                    for file_name in file_name_list
                ],
            )
            for subdir, file_name_list in image_file_names.items()
        }
        df = pd.DataFrame(data.values(), index=data.keys())
        print(df.T.sort_index())


# %%
