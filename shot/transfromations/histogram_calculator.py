import numpy as np
import cv2

class HistogramCalculator:
    def __init__(self, bin_count:int = 256, normalize:bool = True, channel:int = 0,
                 range_:int = 256):
        self.bin_count = bin_count
        self.normalize = normalize
        self.channel = channel
        self.range = range_

    def process(self, image:np.ndarray) -> np.ndarray:
        if image is None or not isinstance(image, np.ndarray):
            raise ValueError("Input must be a valid image as a numpy array.")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Input image must be a 3-channel image.")

        hist = cv2.calcHist([image], [self.channel], None, [self.bin_count], [0, self.range])
        if self.normalize:
            hist = cv2.normalize(hist, hist).flatten()
        return hist