import cv2
import numpy as np

class ImageSplitter:
    def __init__(self, size:int):
        self.size = size

    def process(self, image:np.ndarray) -> np.ndarray:
        width, height = image.shape
        self._validate(height, width)
        sub_images = self._extract(height, image, width)
        return np.array(sub_images)

    def _extract(self, height, image, width):
        sub_images = []
        for i in range(0, width, self.size):
            for j in range(0, height, self.size):
                sub_image = image[i:i + self.size, j:j + self.size]
                sub_images.append(sub_image)
        return sub_images

    def _validate(self, height, width):
        is_valid_size_h = height % self.size == 0
        is_valid_size_w = width % self.size == 0
        if not is_valid_size_h or not is_valid_size_w:
            raise ValueError(f"Image size {width}x{height} is not divisible by {self.size}")
