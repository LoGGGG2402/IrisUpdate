from scipy.spatial import distance
import cv2
import numpy as np


def iris_segmentation(img=None, img_path=None):
    if img is None:
        img = cv2.imread(img_path)
    pupil = find_pupil(img)
    if pupil is None:
        return None
    else:
        iris = find_iris(img, pupil)
        if iris is None:
            return None

    return img, (pupil[0], pupil[1]), pupil[2], (iris[0], iris[1]), iris[2]


def find_pupil(img):
    # Convert to gray scale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Remove noise by blurring the image
    img = cv2.medianBlur(img, 9)
    # Increase the contrast
    img = cv2.convertScaleAbs(img, alpha=1.1, beta=10)

    circles = cv2.HoughCircles(img, method=cv2.HOUGH_GRADIENT, dp=1, minDist=50, param1=100, param2=30, minRadius=5,
                               maxRadius=30)
    if circles is not None:
        circles = np.uint16(np.around(circles))

        for i in circles[0, :]:
            copy = img.copy()
            cv2.circle(copy, (i[0], i[1]), i[2], (0, 255, 0), 2)

        # Find the smallest circle (assuming it's the pupil)
        min_circle = min(circles[0, :], key=lambda x: x[2])
        return min_circle
    else:
        return None


def find_iris(img, inner_circle):
    # Convert to gray scale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Remove noise by blurring the image
    img = cv2.medianBlur(img, 9)
    # Increase the constrast
    img = cv2.convertScaleAbs(img, alpha=1.1, beta=10)

    circles = cv2.HoughCircles(img, method=cv2.HOUGH_GRADIENT, dp=1, minDist=1, param1=30, param2=1,
                               minRadius=round(inner_circle[2] * 1.6), maxRadius=round(inner_circle[2] * 2.7))

    if circles is not None:
        circles = np.uint16(np.around(circles))
        # Find the circle that neighbours the inner circle

        neighbours = [x for x in circles[0, :] if
                      distance.euclidean((x[0], x[1]), (inner_circle[0], inner_circle[1])) < 4]
        if len(neighbours) == 0:
            return None
        max_circle = max(neighbours, key=lambda x: x[2])
        return max_circle

    else:
        return None
