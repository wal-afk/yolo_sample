# %%
from glob import glob
import csv
import os
import shutil

import yaml
import pandas as pd

from .show_util import show_all_images, create_yolo_GT_image


class Data:
    def __init__(self, root_dir: str, image_path: str):
        self.root_dir = root_dir
        self.label_dir = f"{root_dir}/labels"
        self.image_dir = f"{root_dir}/images"
        self.image_path = image_path
        self.group = os.path.relpath(os.path.dirname(image_path), start=self.image_dir)

        self.label_file_name = (
            f"{os.path.splitext(os.path.basename(image_path))[0]}.txt"
        )

        _label_path = self.find_label_path()
        self.label_count: dict[int, int] = (
            self._count_label(_label_path) if _label_path is not None else {}
        )

    def find_label_path(self) -> str | None:
        lable_paths = glob(
            f"{self.label_dir}/**/{self.label_file_name}", recursive=True
        )
        if len(lable_paths) >= 2:
            print(f"ignore: {self.label_file_name} found more than 1")
        elif len(lable_paths) == 1:
            return lable_paths[0]
        return None

    def _count_label(self, label_path: str) -> dict[int, int]:
        label_count = {}
        with open(label_path) as f:
            reader = csv.reader(f, delimiter=" ")
            for line in reader:
                label_idx = int(line[0])
                if label_idx not in label_count:
                    label_count[label_idx] = 0
                label_count[label_idx] += 1
        return label_count


class DatasetChecker:
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.label_dir = f"{self.root_dir}/labels"
        self.image_dir = f"{self.root_dir}/images"

        self.label_list = self._read_label_list(f"{root_dir}/labels.txt")
        self.data_list: list[Data] = [
            Data(self.root_dir, image_path)
            for image_path in glob(f"{self.image_dir}/**/*.*", recursive=True)
        ]

        self.label_data_dict: dict[str, list[Data]] = {}
        for data in self.data_list:
            for label_idx, count in data.label_count.items():
                label = self.label_list[label_idx]
                if label not in self.label_data_dict:
                    self.label_data_dict[label] = []
                self.label_data_dict[label].append(data)

    @staticmethod
    def _read_label_list(path):
        with open(path) as f:
            return [line.strip() for line in f.readlines()]

    def show_images_for_each_label(self):
        img_list = []
        for label in self.label_list:
            if label in self.label_data_dict:
                first_data = self.label_data_dict[label][0]
                img_list.append(
                    create_yolo_GT_image(
                        first_data.image_path,
                        first_data.find_label_path(),
                        self.label_list,
                    )
                )
        show_all_images(img_list)

    def print_labels_count(self):
        grouped_label_count = {}
        for data in self.data_list:
            grp = data.group
            if grp not in grouped_label_count:
                grouped_label_count[grp] = {}

            for label_idx, count in data.label_count.items():
                label = self.label_list[label_idx]
                if label not in grouped_label_count[grp]:
                    grouped_label_count[grp][label] = 0
                grouped_label_count[grp][label] += count

        df = pd.DataFrame(
            grouped_label_count.values(), index=grouped_label_count.keys()
        )
        print(df.T.sort_index())

    def relocate_label_files(self):
        copied_paths = []
        for data in self.data_list:
            os.makedirs(f"{self.label_dir}/{data.group}", exist_ok=True)
            lable_path = data.find_label_path()
            if lable_path is None:
                continue
            target_path = f"{self.label_dir}/{data.group}/{data.label_file_name}"
            if os.path.abspath(lable_path) != os.path.abspath(target_path):
                shutil.copy(lable_path, target_path)
                copied_paths.append(lable_path)
        for path in copied_paths:
            os.remove(path)

    def create_custom_yaml(self):
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
