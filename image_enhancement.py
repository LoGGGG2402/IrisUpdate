import cv2
from iris_nomalization import iris_normalization


def image_enhancement(img, image_path):
    img = iris_normalization(img, image_path)
    if img is None:
        return None
    img = cv2.convertScaleAbs(img, 1.5, 2)
    return img
