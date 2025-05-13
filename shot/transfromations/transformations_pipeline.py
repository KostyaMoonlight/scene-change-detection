import numpy as np
import cv2
from shot.transfromations.color_manipulation import ColorManipulation
from shot.transfromations.histogram_calculator import HistogramCalculator
from shot.transfromations.color_space_converter import ColorSpaceConverter
from shot.transfromations.image_splitter import ImageSplitter
from shot.transfromations.pooling import Pooling
from shot.transfromations.smooth_methods import SmoothMethods


class TransformationPipeline:
    def __init__(self, transformations: list):
        self. transformations = self._initialize_transformations(transformations)


    def process_image(self, image: np.ndarray) -> np.ndarray:
        for transformation in self.transformations:
            image = transformation.process(image)
        return image

    def get_transformations_dict(self):
        return {
            'smooth_methods': SmoothMethods, # 3 channels (h, w, c)
            'color_manipulation': ColorManipulation, # 3 channels (h, w, c)
            'color_space_converter': ColorSpaceConverter, # 3 channels (h, w, c)
            'image_splitter': ImageSplitter, # 4 channels (n, h, w, c)
            'pooling': Pooling, # 4 channels (n, h, w, c)
            'histogram_calculator': HistogramCalculator, #3 channels (n, c, b)
        }

    def _initialize_transformations(self, transformations):
        transformations_dict = self.get_transformations_dict()
        transformations = []
        for name, params in transformations:
            transformations.append(transformations_dict[name](**params))
        return transformations
