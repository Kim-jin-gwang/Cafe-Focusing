from abc import ABC, abstractmethod
from typing import Tuple, Dict, Optional
import cv2
import numpy as np

class BaseBackgroundGenerator(ABC):
    """
    Abstract base class for background processing (blurring, desaturating, etc.).
    """
    @abstractmethod
    def generate(self, img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Processes the background of the image.
        
        Args:
            img: Input image in BGR format.
            mask: Binary mask where 255 represents foreground.
            
        Returns:
            A tuple of (processed_bg, steps) where:
                - processed_bg is the fully processed background image.
                - steps is a dictionary of intermediate steps.
        """
        pass


class BlurBackgroundGenerator(BaseBackgroundGenerator):
    """
    Blurs the background using Average Blur, Gaussian Blur, or a
    disc-kernel Bokeh blur that mimics real camera lens defocus
    (bright points spread into circular highlights).
    """
    def __init__(
        self,
        blur_type: str = 'average',
        blur_size: Tuple[int, int] = (13, 13)
    ):
        if blur_type not in ('average', 'gaussian', 'bokeh'):
            raise ValueError("blur_type must be 'average', 'gaussian', or 'bokeh'")
        self.blur_type = blur_type
        self.blur_size = blur_size

    @staticmethod
    def _disc_kernel(diameter: int) -> np.ndarray:
        """Circular (disc) convolution kernel — the aperture shape of a lens."""
        diameter = max(3, diameter | 1)  # odd, >= 3
        r = diameter // 2
        y, x = np.ogrid[-r:r + 1, -r:r + 1]
        kernel = ((x * x + y * y) <= r * r).astype(np.float32)
        return kernel / kernel.sum()

    def generate(self, img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        steps = {}

        if self.blur_type == 'average':
            blurred_bg = cv2.blur(img, self.blur_size)
        elif self.blur_type == 'bokeh':
            # Boost highlights in near-linear light before the disc filter so
            # bright spots bloom into circles instead of averaging away.
            kernel = self._disc_kernel(self.blur_size[0])
            linear = (img.astype(np.float32) / 255.0) ** 3.0
            blurred = cv2.filter2D(linear, -1, kernel)
            blurred_bg = (np.clip(blurred, 0.0, 1.0) ** (1.0 / 3.0) * 255.0).astype(np.uint8)
        else: # gaussian
            # Ensure odd numbers for gaussian kernel size
            kernel_x = self.blur_size[0] + 1 if self.blur_size[0] % 2 == 0 else self.blur_size[0]
            kernel_y = self.blur_size[1] + 1 if self.blur_size[1] % 2 == 0 else self.blur_size[1]
            blurred_bg = cv2.GaussianBlur(img, (kernel_x, kernel_y), 0)

        steps['img_blur'] = blurred_bg.copy()
        return blurred_bg, steps


class DesaturateBackgroundGenerator(BaseBackgroundGenerator):
    """
    Reduces the saturation (colorfulness) of the background.
    Can be chained with an optional blur generator.
    """
    def __init__(
        self,
        saturation_factor: float = 0.3,
        blur_generator: Optional[BaseBackgroundGenerator] = None
    ):
        self.saturation_factor = np.clip(saturation_factor, 0.0, 1.0)
        self.blur_generator = blur_generator

    def generate(self, img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        steps = {}
        
        # Apply blur first if a blur generator is provided
        current_img = img
        if self.blur_generator is not None:
            current_img, blur_steps = self.blur_generator.generate(img, mask)
            steps.update(blur_steps)
            
        # Convert to grayscale
        gray = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)
        gray_3ch = cv2.merge([gray, gray, gray])
        
        # Linearly interpolate between original (or blurred) and grayscale
        desaturated = cv2.addWeighted(current_img, self.saturation_factor, gray_3ch, 1.0 - self.saturation_factor, 0)
        steps['bg_desaturate'] = desaturated.copy()
        
        return desaturated, steps


class DarkenBackgroundGenerator(BaseBackgroundGenerator):
    """
    Reduces the brightness of the background to make the foreground pop even more.
    Can be chained with an optional blur generator.
    """
    def __init__(
        self,
        brightness_factor: float = 0.6,
        blur_generator: Optional[BaseBackgroundGenerator] = None
    ):
        self.brightness_factor = np.clip(brightness_factor, 0.0, 1.0)
        self.blur_generator = blur_generator

    def generate(self, img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        steps = {}
        
        # Apply blur first if a blur generator is provided
        current_img = img
        if self.blur_generator is not None:
            current_img, blur_steps = self.blur_generator.generate(img, mask)
            steps.update(blur_steps)
            
        # Darken the image
        darkened = (current_img.astype(np.float32) * self.brightness_factor)
        darkened = np.clip(darkened, 0, 255).astype(np.uint8)
        steps['bg_darken'] = darkened.copy()
        
        return darkened, steps
