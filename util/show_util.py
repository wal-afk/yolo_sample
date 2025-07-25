import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


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
