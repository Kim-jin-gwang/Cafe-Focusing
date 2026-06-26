from abc import ABC, abstractmethod
from typing import Tuple, Dict
import cv2
import numpy as np

class BaseBlender(ABC):
    """
    Abstract base class for blending foreground and background images.
    """
    @abstractmethod
    def blend(
        self, 
        img: np.ndarray, 
        bg_img: np.ndarray, 
        mask: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Blends the foreground of the original image with the processed background.
        
        Args:
            img: Original input image in BGR format.
            bg_img: Processed background image in BGR format.
            mask: Smoothed mask (0 to 255).
            
        Returns:
            A tuple of (mixed_image, steps) where:
                - mixed_image is the final blended output image.
                - steps is a dictionary of intermediate steps.
        """
        pass


class AlphaBlender(BaseBlender):
    """
    Blends the foreground and background using the smoothed mask as an alpha channel.
    Provides natural transition boundaries and avoids edge artifacts.
    """
    def blend(
        self, 
        img: np.ndarray, 
        bg_img: np.ndarray, 
        mask: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        steps = {}
        # Normalize mask to 0.0 - 1.0 alpha channel
        alpha = np.dstack([mask] * 3).astype('float32') / 255.0
        
        img_float = img.astype('float32')
        bg_float = bg_img.astype('float32')
        
        # Linear interpolation: alpha * FG + (1 - alpha) * BG
        mixed = (alpha * img_float) + ((1 - alpha) * bg_float)
        mixed = np.clip(mixed, 0, 255).astype('uint8')
        
        # Compatibility steps
        steps['only_coffee'] = (alpha * img_float + (1 - alpha) * 255.0).astype('uint8')
        steps['img_blur'] = bg_img.copy()
        
        return mixed, steps


class LegacyAndBlender(BaseBlender):
    """
    Legacy blending method using bitwise_and operations.
    Extracts object and background independently onto a colored canvas,
    blurs the background, and merges them using bitwise AND.
    """
    def __init__(
        self,
        mask_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        bg_blur_size: Tuple[int, int] = (13, 13)
    ):
        self.mask_color = mask_color
        self.bg_blur_size = bg_blur_size

    def blend(
        self, 
        img: np.ndarray, 
        bg_img: np.ndarray, 
        mask: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        steps = {}
        mask_stack = np.dstack([mask] * 3).astype('float32') / 255.0
        img_float = img.astype('float32') / 255.0
        
        # Object extraction (foreground)
        masked = (mask_stack * img_float) + ((1 - mask_stack) * self.mask_color)
        masked = (masked * 255).astype('uint8')
        steps['only_coffee'] = masked.copy()
        
        # Background extraction
        back_masked = ((1 - mask_stack) * img_float) + (mask_stack * self.mask_color)
        back_masked = (back_masked * 255).astype('uint8')
        steps['background'] = back_masked.copy()
        
        # Blur background (legacy style: blur the masked background, not the full image)
        img_blur = cv2.blur(back_masked, self.bg_blur_size)
        steps['img_blur'] = img_blur.copy()
        
        # Composite via bitwise AND
        mixed = cv2.bitwise_and(img_blur, masked)
        
        return mixed, steps
