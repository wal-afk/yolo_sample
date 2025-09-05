import math

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import cv2


def show_2_images_path(path1: str, path2: str):
    show_2_images(mpimg.imread(path1), mpimg.imread(path2))


def show_2_images(img1: np.ndarray, img2: np.ndarray):
    fig, axes = plt.subplots(1, 2, figsize=(20, 12))  # 横に2枚、サイズ指定

    axes[0].imshow(img1)
    axes[0].axis("off")
    axes[0].set_title("original")
    axes[1].imshow(img2)
    axes[1].axis("off")
    axes[1].set_title("predict")
    plt.tight_layout()
    plt.show()


def show_all_images(img_list: list[np.ndarray], ncols=4):
    r = 0
    c = 0
    for img in img_list:
        if c == 0:
            fig, axes = plt.subplots(1, ncols, figsize=(8 * ncols, 6))
        axes[c].imshow(img)
        axes[c].axis("off")
        c += 1
        if c == ncols:
            c = 0
            r += 1


def get_distinct_colors_10():
    cmap = plt.get_cmap("tab10")
    colors = []
    for i in range(10):
        color = cmap(i)[:3]  # RGBで取得（0〜1範囲）
        color = tuple(int(c * 255) for c in color)  # 0〜255に変換
        colors.append(color)
    return colors


def create_yolo_GT_image(image_path, label_path, class_names):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, _ = image.shape
    with open(label_path, "r") as f:
        lines = f.readlines()

    cols = get_distinct_colors_10()

    for line in lines:
        cls, x_center, y_center, w, h = map(float, line.strip().split())
        x1 = int((x_center - w / 2) * width)
        y1 = int((y_center - h / 2) * height)
        x2 = int((x_center + w / 2) * width)
        y2 = int((y_center + h / 2) * height)

        line_color = cols[int(cls) % 10]
        font_color = (255, 255, 255)
        thickness = math.ceil(height / 400)
        cv2.rectangle(image, (x1, y1), (x2, y2), line_color, 3 * thickness)
        label = class_names[int(cls)]
        face = cv2.FONT_HERSHEY_SIMPLEX
        font_size = int(height / 20)
        scale = cv2.getFontScaleFromHeight(face, font_size)
        size, below_baseline = cv2.getTextSize(label, face, scale, thickness)

        y_baseline = (
            y1 - below_baseline if y1 >= below_baseline + size[1] else y2 + size[1]
        )

        cv2.rectangle(
            image,
            (x1, y_baseline - size[1]),
            (x1 + size[0], y_baseline +below_baseline),
            line_color,
            cv2.FILLED,
        )

        cv2.putText(image, label, (x1, y_baseline), face, scale, font_color, thickness)
    return image
