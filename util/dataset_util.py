# %%
from glob import glob
import csv
import os
import shutil
import filecmp
from zipfile import ZipFile

import yaml
import pandas as pd

from .show_util import show_all_images, create_yolo_GT_image


def unzip(dir_path: str, pattern_dict: list[str]):
    shutil.rmtree(dir_path, ignore_errors=True)
    for pattern in pattern_dict.values():
        if len(glob(pattern)) != 1:
            print(f"{pattern}を1つだけアップロードしてください")
            return False

    for key, pattern in pattern_dict.items():
        with ZipFile(glob(pattern)[0]) as zip:
            zip.extractall(f"{dir_path}/{key}")
    return True


class Data:
    """
    画像ファイルに対するアノテーション（ラベルファイル）を関連付けて管理する為のクラス
    次のフォルダ構成を前提とします
    {root_dir}/
        images/
            **/{image_file_name}.*
        labels/
            **/{image_file_name}.txt
    """

    def __init__(self, root_dir: str, image_path: str):
        """
        Args:
            root_dir: データセットのルートディレクトリ
            image_path: 画像ファイルへのパス（必ず{root_dir}/imagesを含むパスであること）
        """
        self.root_dir = root_dir
        self.label_dir = f"{root_dir}/labels"
        self.image_dir = f"{root_dir}/images"
        self.image_path = image_path
        self.group = os.path.relpath(os.path.dirname(image_path), start=self.image_dir)

        self.label_file_name = (
            f"{os.path.splitext(os.path.basename(image_path))[0]}.txt"
        )

        _label_path = self.find_label_path(raise_if_different_labels_exist=True)
        self.label_count: dict[int, int] = (
            self._count_label(_label_path) if _label_path is not None else {}
        )

    def find_label_path(self, raise_if_different_labels_exist=False) -> str | None:
        lable_paths = glob(
            f"{self.label_dir}/**/{self.label_file_name}", recursive=True
        )
        if len(lable_paths) >= 2:
            if raise_if_different_labels_exist:
                for path in lable_paths:
                    if not filecmp.cmp(lable_paths[0], path, shallow=False):
                        raise Exception(
                            f"different {self.label_file_name} found more than 1"
                        )
            return lable_paths[0]
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
    def __init__(self, root_dir: str, *, merge_dataset_dir: str | None = None):
        """
        次のフォルダ構成を前提とします
        {root_dir}/
            images/
                **/{image_file_name}.*
            labels/
                **/{image_file_name}.txt
            labels.txt

        merge_dataset_dirが与えられた場合、クラスのindexが重複しないようにマージします。
        その際、labels/**/*.txtとlabels.txtの内容を書き換えます。

        Args:
            root_dir: データセットのルートディレクトリ
            merge_dataset_dir: 既存のデータセットをマージする場合はそのディレクトリを指定する
        """

        self.root_dir = root_dir
        self.label_dir = f"{self.root_dir}/labels"
        self.image_dir = f"{self.root_dir}/images"

        if merge_dataset_dir is not None:
            # originalファイルの書き換え。
            self.label_list = self._merge_label(root_dir, merge_dataset_dir)

            # ファイルコピーによるマージ
            for extra_image_path in glob(
                f"{merge_dataset_dir}/images/**/*.*", recursive=True
            ):
                target_path = f"{self.image_dir}/{os.path.relpath(
                    extra_image_path,
                    start=f'{merge_dataset_dir}/images')}"
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                shutil.copyfile(extra_image_path, target_path)
            for extra_label_path in glob(
                f"{merge_dataset_dir}/labels/**/*.txt", recursive=True
            ):
                target_path = f"{self.label_dir}/{os.path.relpath(
                    extra_label_path,
                    start=f'{merge_dataset_dir}/labels')}"
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                shutil.copyfile(extra_label_path, target_path)
        else:
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

    @staticmethod
    def _merge_label(root_dir: str, extra_dir: str):
        """
        2つのデータセット中のlabelをマージする為に、
        root_dir中にあるlabels.txtとlabels/**/*.txtの内容を書き換えます。

        Args:
            root_dir: 元のデータセットのルートディレクトリ
            extra_dir: 追加するデータセットのルートディレクトリ
        """

        label_to_idx: dict[str, int] = {}
        labels: list[str] = []

        # labelはextra_dirの方が先に来るようにする。つまり、extra_dirのlabelのindexは0から始まる。
        with open(f"{extra_dir}/labels.txt") as f:
            extra_labels = [line.strip() for line in f.readlines()]

        labels = [*extra_labels]
        label_to_idx = {label: idx for idx, label in enumerate(extra_labels)}

        with open(f"{root_dir}/labels.txt") as f:
            original_labels = [line.strip() for line in f.readlines()]

        for original_label in original_labels:
            if original_label not in labels:
                label_to_idx[original_label] = len(labels)
                labels.append(original_label)

        with open(f"{root_dir}/labels.txt", "w") as f:
            f.write("\n".join(labels))

        for original_label_path in glob(f"{root_dir}/labels/**/*.txt", recursive=True):
            with open(original_label_path) as f:
                lines = f.readlines()
            with open(original_label_path, "w") as f:
                for line in lines:
                    sp = line.split(" ")
                    new_label_idx = label_to_idx[original_labels[int(sp[0])]]
                    updated_line = " ".join([str(new_label_idx), *sp[1:]])
                    f.write(updated_line)

        return labels

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
        copy_dict: dict[str, list[str]] = {}
        preserve: list[str] = []

        for data in self.data_list:
            os.makedirs(f"{self.label_dir}/{data.group}", exist_ok=True)
            lable_path = data.find_label_path()
            if lable_path is None:
                continue
            target_path = f"{self.label_dir}/{data.group}/{data.label_file_name}"
            if os.path.abspath(lable_path) != os.path.abspath(target_path):
                if lable_path not in copy_dict:
                    copy_dict[lable_path] = []
                copy_dict[lable_path].append(target_path)
            else:
                preserve.append(lable_path)

        for lable_path, target_path_list in copy_dict.items():
            for target_path in target_path_list:
                shutil.copy(lable_path, target_path)

        for lable_path in copy_dict:
            if lable_path not in preserve:
                os.remove(lable_path)

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
