import numpy as np
import cv2


class SmoothMethods:
    def __init__(self, method: str = 'gaussian', **kwargs):
        """
        :param method: ['gaussian', 'median', 'bilateral']
        :param kwargs:
            if method == 'gaussian':
                kernel_size: int
                sigma: int
            if method == 'median':
                kernel_size: int
            if method == 'bilateral':
                diameter: int
                sigma_color: int
                sigma_space: int
        """
        self.method = method
        self.valid_methods = ['gaussian', 'median', 'bilateral']
        if method not in self.valid_methods:
            raise ValueError(f"Unsupported smoothing method: {method}. Supported methods are: {self.valid_methods}")
        self.kwargs = kwargs

    def process(self, image: np.ndarray) -> np.ndarray:
        if image is None or not isinstance(image, np.ndarray):
            raise ValueError("Input must be a valid image as a numpy array.")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Input image must be a 3-channel image.")
        return self.smooth(image)

    def smooth(self, image: np.ndarray) -> np.ndarray:
        if self.method == 'gaussian':
            return self._gaussian_smoothing(image)
        elif self.method == 'median':
            return self._median_smoothing(image)
        elif self.method == 'bilateral':
            return self._bilateral_smoothing(image)
        else:
            raise ValueError(f"Unsupported smoothing method: {self.method}")

    def _gaussian_smoothing(self, image: np.ndarray) -> np.ndarray:
        kernel_size = self.kwargs.get('kernel_size', 5)
        sigma = self.kwargs.get('sigma', 1)
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

    def _median_smoothing(self, image: np.ndarray) -> np.ndarray:
        kernel_size = self.kwargs.get('kernel_size', 5)
        return cv2.medianBlur(image, kernel_size)

    def _bilateral_smoothing(self, image: np.ndarray) -> np.ndarray:
        diameter = self.kwargs.get('diameter', 9)
        sigma_color = self.kwargs.get('sigma_color', 75)
        sigma_space = self.kwargs.get('sigma_space', 75)
        return cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)
