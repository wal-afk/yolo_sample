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


def create_yolo_GT_image(image_path, label_path, class_names):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, _ = image.shape
    print(type(image), image.shape, image.dtype)
    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        cls, x_center, y_center, w, h = map(float, line.strip().split())
        x1 = int((x_center - w / 2) * width)
        y1 = int((y_center - h / 2) * height)
        x2 = int((x_center + w / 2) * width)
        y2 = int((y_center + h / 2) * height)

        color = (255, 0, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = class_names[int(cls)]
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return image
