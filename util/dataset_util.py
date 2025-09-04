# %%
from glob import glob
import csv
import os
import shutil

import yaml
import pandas as pd


class DatasetChecker:
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.label_dir = f"{self.root_dir}/labels"

        self.label_list = self.read_label_def(f"{root_dir}/labels.txt")
        self.image_file_names = self._list_up_iamges()
        self.label_file_names = {
            subdir: [
                f"{os.path.splitext(file_name)[0]}.txt" for file_name in file_name_list
            ]
            for subdir, file_name_list in self.image_file_names.items()
        }

    def _count_label(self, file_name_list):
        label_count = {}
        for file_name in file_name_list:
            lable_path = self._find_label_file(file_name)
            if lable_path is None:
                continue
            with open(lable_path) as f:
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

    def _list_up_iamges(self):
        image_file_names = {}
        for image_path in glob(f"{self.root_dir}/images/**/*.*", recursive=True):
            group = os.path.relpath(
                os.path.dirname(image_path), start=f"{self.root_dir}/images"
            )
            if group not in image_file_names:
                image_file_names[group] = []
            image_file_names[group].append(os.path.basename(image_path))
        return image_file_names

    def print_labels_count(self):
        data = {
            subdir: self._count_label(file_name_list)
            for subdir, file_name_list in self.label_file_names.items()
        }
        df = pd.DataFrame(data.values(), index=data.keys())
        print(df.T.sort_index())

    def _find_label_file(self, file_name):
        lable_paths = glob(f"{self.label_dir}/**/{file_name}", recursive=True)
        if len(lable_paths) >= 2:
            print(f"ignore: {file_name} found more than 1")
        elif len(lable_paths) == 1:
            return lable_paths[0]
        return None

    def relocate_label_files(self):
        for subdir, file_name_list in self.label_file_names.items():
            os.makedirs(f"{self.label_dir}/{subdir}", exist_ok=True)
            for file_name in file_name_list:
                lable_path = self._find_label_file(file_name)
                if lable_path is None:
                    continue
                shutil.move(lable_path, f"{self.label_dir}/{subdir}/{file_name}")

    def craete_custom_yaml(self):
        with open(f"{self.root_dir}/custom.yaml", "w") as f:
            f.write(
                yaml.dump(
                    {
                        "path": self.root_dir,
                        "train": "images/train",
                        "val": "images/val",
                        "nc": len(self.label_list),
                        "names": self.label_list,
                    }
                )
            )


# %%
