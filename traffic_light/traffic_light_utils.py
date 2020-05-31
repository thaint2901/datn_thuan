import cv2
import numpy as np


def chi2_distance(histA, histB, eps=1e-10):
    """tính toán khoảng cách của các hist
    
    Keyword Arguments:
        eps {float} -- Tránh việc chia cho 0 (default: {1e-10})
    
    Returns:
        float -- khoảng cách tính toán được
    """
    d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
        for (a, b) in zip(histA, histB)])
    return d


def describe_hist(image, bins=[8, 8, 8]):
    """Tính toán histogram theo BIN (thùng)
    
    Arguments:
        image {array}
    
    Keyword Arguments:
        bins {list} -- list int, các bins theo 3 kênh màu (default: {[8, 8, 8]})
    
    Returns:
        array -- ma trận hist được flatten()
    """
    hist = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist,hist)

    return hist.flatten()