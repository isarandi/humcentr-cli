import functools

import cv2
import numpy as np


def white_balance(img, a=None, b=None):
    result = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    avg_a = a if a is not None else np.mean(result[..., 1])
    avg_b = b if b is not None else np.mean(result[..., 2])
    result[..., 1] = result[..., 1] - ((avg_a - 128) * (result[..., 0] / 255.0) * 1.1)
    result[..., 2] = result[..., 2] - ((avg_b - 128) * (result[..., 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2RGB)
    return result


def adjust_gamma(image, gamma, inplace=False):
    if inplace:
        cv2.LUT(image, _get_gamma_lookup_table(gamma), dst=image)
        return image

    return cv2.LUT(image, _get_gamma_lookup_table(gamma))


@functools.lru_cache()
def _get_gamma_lookup_table(gamma):
    return (np.linspace(0, 1, 256) ** gamma * 255).astype(np.uint8)
