import numpy as np
import cv2

class HDRFilter:
    def __init__(self, method: str = 'clahe', strength: float = 1.0, gamma: float = 1.0, **kwargs):
        """
        Initialize HDR filter transformation.
        
        Args:
            method (str): HDR fusion method ('clahe', 'debevec', 'robertson', 'mertens')
            strength (float): Strength of the HDR effect (0.0 to 2.0)
            gamma (float): Gamma correction value (0.1 to 2.0)
            **kwargs: Additional parameters for specific methods:
                - debevec: samples (int), lambda_ (float)
                - robertson: samples (int), lambda_ (float)
                - mertens: contrast_weight (float), saturation_weight (float), exposure_weight (float)
        """
        self.method = method
        self.valid_methods = ['clahe', 'debevec', 'robertson', 'mertens']
        if method not in self.valid_methods:
            raise ValueError(f"Unsupported HDR method: {method}. Supported methods are: {self.valid_methods}")
            
        self.strength = np.clip(strength, 0.0, 2.0)
        self.gamma = np.clip(gamma, 0.1, 2.0)
        self.kwargs = kwargs

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Apply HDR effect to the input image.
        
        Args:
            image (np.ndarray): Input image in BGR format
            
        Returns:
            np.ndarray: Processed image with HDR effect
        """
        if image is None or not isinstance(image, np.ndarray):
            raise ValueError("Input must be a valid image as a numpy array.")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Input image must be a 3-channel image.")
            
        if self.method == 'clahe':
            return self._apply_clahe(image)
        elif self.method == 'debevec':
            return self._apply_debevec(image)
        elif self.method == 'robertson':
            return self._apply_robertson(image)
        elif self.method == 'mertens':
            return self._apply_mertens(image)
        else:
            raise ValueError(f"Unsupported HDR method: {self.method}")

    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        
        # Apply gamma correction
        l = np.power(l / 255.0, self.gamma) * 255.0
        l = l.astype(np.uint8)
        
        # Merge channels and convert back to BGR
        enhanced_lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # Blend with original image based on strength
        result = cv2.addWeighted(image, 1 - self.strength, enhanced, self.strength, 0)
        return result

    def _apply_debevec(self, image: np.ndarray) -> np.ndarray:
        # Create exposure times array
        times = np.array([1/30.0, 1/15.0, 1/8.0, 1/4.0, 1/2.0, 1.0, 2.0, 4.0, 8.0, 15.0, 30.0], dtype=np.float32)
        
        # Create synthetic exposures
        images = []
        for t in times:
            # Simulate different exposures
            exposed = np.clip(image * t, 0, 255).astype(np.uint8)
            images.append(exposed)
        
        # Convert to float32 for HDR processing
        images = [img.astype(np.float32) for img in images]
        
        # Create Debevec HDR
        samples = self.kwargs.get('samples', 100)
        lambda_ = self.kwargs.get('lambda_', 10.0)
        
        # Create response curve
        response = cv2.createCalibrateDebevec()
        response.calibrate(images, times, samples=samples, lambda_=lambda_)
        
        # Merge exposures
        merge_debevec = cv2.createMergeDebevec()
        hdr = merge_debevec.process(images, times, response)
        
        # Tone mapping
        tonemap = cv2.createTonemap(gamma=self.gamma)
        ldr = tonemap.process(hdr)
        
        # Convert to 8-bit
        result = np.clip(ldr * 255, 0, 255).astype(np.uint8)
        
        # Blend with original image based on strength
        result = cv2.addWeighted(image, 1 - self.strength, result, self.strength, 0)
        return result

    def _apply_robertson(self, image: np.ndarray) -> np.ndarray:
        # Create exposure times array
        times = np.array([1/30.0, 1/15.0, 1/8.0, 1/4.0, 1/2.0, 1.0, 2.0, 4.0, 8.0, 15.0, 30.0], dtype=np.float32)
        
        # Create synthetic exposures
        images = []
        for t in times:
            # Simulate different exposures
            exposed = np.clip(image * t, 0, 255).astype(np.uint8)
            images.append(exposed)
        
        # Convert to float32 for HDR processing
        images = [img.astype(np.float32) for img in images]
        
        # Create Robertson HDR
        samples = self.kwargs.get('samples', 100)
        lambda_ = self.kwargs.get('lambda_', 10.0)
        
        # Create response curve
        response = cv2.createCalibrateRobertson()
        response.calibrate(images, times, samples=samples, lambda_=lambda_)
        
        # Merge exposures
        merge_robertson = cv2.createMergeRobertson()
        hdr = merge_robertson.process(images, times, response)
        
        # Tone mapping
        tonemap = cv2.createTonemap(gamma=self.gamma)
        ldr = tonemap.process(hdr)
        
        # Convert to 8-bit
        result = np.clip(ldr * 255, 0, 255).astype(np.uint8)
        
        # Blend with original image based on strength
        result = cv2.addWeighted(image, 1 - self.strength, result, self.strength, 0)
        return result

    def _apply_mertens(self, image: np.ndarray) -> np.ndarray:
        # Create exposure times array
        times = np.array([1/30.0, 1/15.0, 1/8.0, 1/4.0, 1/2.0, 1.0, 2.0, 4.0, 8.0, 15.0, 30.0], dtype=np.float32)
        
        # Create synthetic exposures
        images = []
        for t in times:
            # Simulate different exposures
            exposed = np.clip(image * t, 0, 255).astype(np.uint8)
            images.append(exposed)
        
        # Get Mertens fusion parameters
        contrast_weight = self.kwargs.get('contrast_weight', 1.0)
        saturation_weight = self.kwargs.get('saturation_weight', 1.0)
        exposure_weight = self.kwargs.get('exposure_weight', 1.0)
        
        # Create Mertens fusion
        merge_mertens = cv2.createMergeMertens(
            contrast_weight=contrast_weight,
            saturation_weight=saturation_weight,
            exposure_weight=exposure_weight
        )
        
        # Process the images
        result = merge_mertens.process(images)
        
        # Convert to 8-bit
        result = np.clip(result * 255, 0, 255).astype(np.uint8)
        
        # Blend with original image based on strength
        result = cv2.addWeighted(image, 1 - self.strength, result, self.strength, 0)
        return result 