import cv2
from matplotlib import pyplot as plt
import numpy as np

def ransac(src, dst, num_ite = 100, threshold = 5, worst = 100):
    inner = []
    max_M = 0
    size = dst.shape[0]
    for ite in range(num_ite):
        new_inner = []
        random = np.random.randint(0, size-1, 4)
        M = cv2.findHomography(src[random], dst[random], 0)[0]
        new = cv2.perspectiveTransform(src, M)
        for i in range(size):
            norm = np.linalg.norm(new[i]- dst[i])
            if norm < threshold:
                new_inner.append(i)
        if len(new_inner) > len(inner):
            inner = new_inner
            max_M = M
    return inner, max_M