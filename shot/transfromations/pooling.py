import cv2
import numpy as np

class Pooling:
    def __init__(self, pool_size: int = 2, stride: int = 2, mode: str = 'max', input_type: str = 'image'):
        self.pool_size = pool_size
        self.stride = stride
        self.mode = mode
        self.valid_modes = ['max', 'average']
        if mode not in self.valid_modes:
            raise ValueError(f"Unsupported pooling mode: {mode}. Supported modes are: {self.valid_modes}")

        self.input_type = input_type
        self.valid_input_types = ['image', 'histogram']
        if input_type not in self.valid_input_types:
            raise ValueError(f"Unsupported input type: {input_type}. Supported types are: {self.valid_input_types}")

    def process(self, input_data):
        if self.input_type == 'image':
            return self.pool2d(input_data)
        elif self.input_type == 'histogram':
            return self.pool1d(input_data)
        else:
            raise ValueError(f"Unsupported input type: {self.input_type}")



    def pool2d(self, image):
        if self.mode == 'max':
            return self.max_pooling(image)
        elif self.mode == 'average':
            return self.average_pooling(image)
        else:
            raise ValueError(f"Unsupported pooling mode: {self.mode}")

    def pool1d(self, histogram):
        output_len = int((len(histogram) - self.pool_size) / self.stride) + 1
        pooled_histogram = np.zeros(output_len)
        for i in range(output_len):
            start = i * self.stride
            end = start + self.pool_size
            if self.mode == 'max':
                pooled_histogram[i] = np.max(histogram[start:end])
            elif self.mode == 'average':
                pooled_histogram[i] = np.mean(histogram[start:end])
            else:
                raise ValueError(f"Unsupported pooling mode: {self.mode}")

    def max_pooling(self, image):
        self.__validate_image(image)
        pooled_image = cv2.resize(image, (image.shape[1] // self.pool_size, image.shape[0] // self.pool_size), interpolation=cv2.INTER_MAX)
        return pooled_image

    def average_pooling(self, image):
        self.__validate_image(image)
        pooled_image = cv2.resize(image, (image.shape[1] // self.pool_size, image.shape[0] // self.pool_size), interpolation=cv2.INTER_LINEAR)
        return pooled_image

    def __validate_image(self, image):
        if image is None or not isinstance(image, np.ndarray):
            raise ValueError("Input must be a valid image as a numpy array.")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Input image must be a 3-channel image.")
