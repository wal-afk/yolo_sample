from __future__ import annotations

import os
import shutil
from glob import glob

from .dataset_util import DatasetChecker


def extract(images_zip: str, labels_zip: str, dest: str) -> str:
    """images/labels zip を dest に展開する。"""
    from zipfile import ZipFile

    shutil.rmtree(dest, ignore_errors=True)
    os.makedirs(dest, exist_ok=True)
    with ZipFile(images_zip) as z:
        z.extractall(dest)
    with ZipFile(labels_zip) as z:
        z.extractall(f"{dest}/labels")
    return dest


def available(drive_dir: str = ".") -> list[str]:
    """使えるデータセット名の一覧（<drive_dir>/<name>/ に images*.zip があるフォルダ名）。"""
    return sorted({os.path.basename(os.path.dirname(p))
                   for p in glob(os.path.join(drive_dir, "*", "images*.zip"))})


def prepare(name: str, drive_dir: str = ".", workdir: str = "ds") -> str:
    """<drive_dir>/<name>/ 内の images*.zip と labels*.zip を展開する。
    （ラベリングツール出力の images.zip / labels_*.zip をフォルダに入れたものでもよい。
    無ければ images/ を持つフォルダをそのまま使う）。"""
    img = _glob1(os.path.join(drive_dir, name, "images*.zip"))
    lab = _glob1(os.path.join(drive_dir, name, "labels*.zip"))
    if img and lab:
        return extract(img, lab, os.path.join(workdir, name))
    for d in (os.path.join(drive_dir, name), name):
        if os.path.isdir(os.path.join(d, "images")):
            return d
    raise FileNotFoundError(f"{drive_dir}/{name}/ に images*.zip / labels*.zip が見つかりません")


def _glob1(pattern: str) -> str | None:
    hits = sorted(glob(pattern))
    return hits[0] if hits else None


def combine(sources: list[str], work: str = "dataset", names: list[str] | None = None):
    """
    複数のデータセットを結合し、custom.yaml を作る。
    戻り値: (custom.yaml のパス, DatasetChecker)
    """
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(f"{work}/images", exist_ok=True)
    os.makedirs(f"{work}/labels", exist_ok=True)

    src_names = [_read_names(f"{src}/labels.txt") for src in sources]
    merged = list(names) if names is not None else _union_names(src_names)
    name_to_idx = {name: i for i, name in enumerate(merged)}

    for src, local_names in zip(sources, src_names):
        tag = os.path.basename(os.path.normpath(src))
        remap = _build_remap(local_names, name_to_idx, src)
        for img in glob(f"{src}/images/**/*.*", recursive=True):
            rel = os.path.relpath(img, f"{src}/images")
            # データセット名を接頭辞にして衝突を防ぐ（group 単位で分ける）
            dst = f"{work}/images/{_namespaced(rel, tag)}"
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(img, dst)
        for lab in glob(f"{src}/labels/**/*.txt", recursive=True):
            rel = os.path.relpath(lab, f"{src}/labels")
            dst = f"{work}/labels/{_namespaced(rel, tag)}"
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            _copy_relabeled(lab, dst, remap)

    with open(f"{work}/labels.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(merged))

    ds = DatasetChecker(work)
    ds.relocate_label_files()
    ds.create_custom_yaml()
    return f"{work}/custom.yaml", ds


def _read_names(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _union_names(name_lists: list[list[str]]) -> list[str]:
    """複数のクラス名リストを出現順を保ったまま統合する。"""
    merged: list[str] = []
    for names in name_lists:
        for name in names:
            if name not in merged:
                merged.append(name)
    return merged


def _build_remap(local_names: list[str], name_to_idx: dict[str, int], src: str) -> dict[int, int]:
    """source のローカルID -> 統合後IDの対応表。"""
    remap = {}
    for local_idx, name in enumerate(local_names):
        if name not in name_to_idx:
            raise ValueError(f"{src}: クラス '{name}' が結合先のクラス空間にありません")
        remap[local_idx] = name_to_idx[name]
    return remap


def _copy_relabeled(src_label: str, dst_label: str, remap: dict[int, int]) -> None:
    """ラベルファイルを remap に従ってindexを振り直しつつコピーする。"""
    if all(k == v for k, v in remap.items()):
        shutil.copyfile(src_label, dst_label)
        return
    with open(src_label, encoding="utf-8") as f:
        lines = f.readlines()
    with open(dst_label, "w", encoding="utf-8") as f:
        for line in lines:
            sp = line.split()
            if not sp:
                continue
            f.write(" ".join([str(remap[int(sp[0])]), *sp[1:]]) + "\n")


def _namespaced(rel: str, tag: str) -> str:
    parts = rel.replace("\\", "/").split("/")
    if len(parts) >= 2:          # train|val / group / file  か  train|val / file
        split = parts[0]
        rest = parts[1:]
        group = rest[0] if len(rest) >= 2 else ""
        fname = rest[-1]
        return f"{split}/{tag}__{group}/{fname}"
    return f"{tag}__/{rel}"
