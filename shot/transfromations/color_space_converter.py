import cv2
import numpy as np

class ColorSpaceConverter:


    def __init__(self, target_color_space: str, **kwargs):
        self.target_color_spaces = target_color_space
        self.color_speces = {
            'RGB': self._convert_to_rgb,
            'GRAY': self._convert_to_gray,
            'YUV': self._convert_to_yuv,
            'YCrCb': self._convert_to_ycrcb,
            'HSV': self._convert_to_hsv,
            'LAB': self._convert_to_lab,
            'Sobel': self._convert_to_sobel,
            'Sobel2': self._convert_to_sobel2,
        }
        if target_color_space not in self.color_speces:
            raise ValueError(f"Unsupported color space: {target_color_space}")
        self.kwargs = kwargs

    def process(self, image: np.ndarray) -> np.ndarray:
        if image is None or not isinstance(image, np.ndarray):
            raise ValueError("Input must be a valid image as a numpy array.")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Input image must be a 3-channel image.")
        return self.color_speces[self.target_color_spaces](image)

    def _convert_to_hsv(self, image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    def _convert_to_lab(self, image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

    def _convert_to_rgb(self, image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _convert_to_gray(self, image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def _convert_to_yuv(self, image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

    def _convert_to_ycrcb(self, image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    #TEST VISUAL
    def _convert_to_sobel(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ksize = self.kwargs.get('ksize', 5)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
        sobel = np.sqrt(sobel_x**2 + sobel_y**2)
        return np.uint8(sobel)

    def _convert_to_sobel2(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ksize = self.kwargs.get('ksize', 5)
        grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=ksize, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
        grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=ksize, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        return cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)