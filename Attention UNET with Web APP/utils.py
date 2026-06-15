import cv2
import numpy as np


def create_overlay(original, mask):

    overlay = original.copy()

    overlay[mask == 1] = [255, 0, 0]

    blended = cv2.addWeighted(
        original,
        0.7,
        overlay,
        0.3,
        0
    )

    return blended