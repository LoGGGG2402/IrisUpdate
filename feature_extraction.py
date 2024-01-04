import cv2
import numpy as np
from image_enhancement import image_enhancement


def gaborconvolve_f(img, minw_length, sigma_f):
    """
    Convolve each row of an imgage with 1D log-Gabor filters.
    """
    rows, ndata = img.shape
    logGabor_f = np.zeros(ndata)
    filterb = np.zeros([rows, ndata], dtype=complex)

    radius = np.arange(ndata / 2 + 1) / (ndata / 2) / 2
    radius[0] = 1

    # filter wavelength
    wavelength = minw_length

    # radial filter component 
    fo = 1 / wavelength
    logGabor_f[0: int(ndata / 2) + 1] = np.exp((-(np.log(radius / fo)) ** 2) /
                                               (2 * np.log(sigma_f) ** 2))
    logGabor_f[0] = 0

    # convolution for each row
    for r in range(rows):
        signal = img[r, 0:ndata]
        imagefft = np.fft.fft(signal)
        filterb[r, :] = np.fft.ifft(imagefft * logGabor_f)

    return filterb


def encode_iris(image_path):
    image = cv2.imread(image_path)
    normalized_image = image_enhancement(image, image_path)
    # if normalized_image is None:
    if normalized_image is None:
        return None

    row, column = normalized_image.shape
    encode_matrix = list()
    convolved_image = gaborconvolve_f(normalized_image, 8, 0.5)
    real_part = np.real(convolved_image)
    imag_part = np.imag(convolved_image)
    for i in range(row):
        encode_row = []
        for j in range(column):
            if real_part[i][j] >= 0:
                if imag_part[i][j] > 0:
                    encode_row.append(1)
                    encode_row.append(1)
                else:
                    encode_row.append(1)
                    encode_row.append(0)
            else:
                if imag_part[i][j] > 0:
                    encode_row.append(0)
                    encode_row.append(1)
                else:
                    encode_row.append(0)
                    encode_row.append(0)
        encode_matrix.append(encode_row)
        cv2.imwrite("imageshow/EncodedImage.bmp", (np.array(encode_matrix) * 255).astype(np.uint8))
    return encode_matrix
