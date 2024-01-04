import segmentation
import cv2
import numpy as np


def iris_normalization(image, image_path):
    segment = segmentation.iris_segmentation(image, image_path)
    if segment is None:
        return None
    segmented_iris, inner_center, inner_radius, outer_center, outer_radius = segment
    # Convert the image to grayscale if it's not already
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    thetas = np.arange(0, 2 * np.pi, 2 * np.pi / 180)
    d = outer_radius - inner_radius
    normalized_image = np.zeros((d, 180))

    for (i, theta) in enumerate(thetas):
        for j in range(d):
            r = j / d

            xP = inner_center[0] + inner_radius * np.cos(theta)
            yP = inner_center[1] + inner_radius * np.sin(theta)
            xI = inner_center[0] + outer_radius * np.cos(theta)
            yI = inner_center[1] + outer_radius * np.sin(theta)

            Xc = int((1 - r) * xP + r * xI)
            Yc = int((1 - r) * yP + r * yI)

            normalized_image[j, i] = image[Yc, Xc]
    return cv2.resize(normalized_image, (30, 180))
