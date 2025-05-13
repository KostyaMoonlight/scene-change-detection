import numpy as np
import cv2

class ColorManipulation:

    def __init__(self, method: str = 'increase_brightness'):
        self.method = method
        self.valid_methods = ['increase_brightness', 'increase_darkness']
        if method not in self.valid_methods:
            raise ValueError(f"Unsupported color manipulation method: {method}. Supported methods are: {self.valid_methods}")

    def process(self, image: np.ndarray) -> np.ndarray:
        self.__validate(image)
        return self._apply_color_manipulation(image)


    def _apply_color_manipulation(self, image: np.ndarray) -> np.ndarray:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        if self.method == 'increase_brightness':
            new_image =  self._increase_brightness(image)
        elif self.method == 'increase_darkness':
            new_image = self._increase_darkness(image)
        else:
            raise ValueError(f"Unsupported color manipulation method: {self.method}")
        return cv2.cvtColor(new_image, cv2.COLOR_HSV2BGR)

    def _increase_brightness(self, hsv):
        h, s, v = cv2.split(hsv)
        limit = 40
        s[s < limit] = 0
        mask = (s >= limit) & (s < 170)
        s[mask] -= 40
        return cv2.merge((h, s, v))

    def _increase_darkness(self, hsv):
        h, s, v = cv2.split(hsv)
        lim = 30
        v[v < lim] = 0
        mask = (v >= lim) & (v < 100)
        v[mask] -= 30
        return cv2.merge((h, s, v))

    def __validate(self, image):
        if image is None or not isinstance(image, np.ndarray):
            raise ValueError("Input must be a valid image as a numpy array.")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Input image must be a 3-channel image.")
