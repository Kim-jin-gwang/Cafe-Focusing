from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any
import cv2
import numpy as np

class BaseForegroundDetector(ABC):
    """
    Abstract base class for foreground object detection and mask generation.
    """
    @abstractmethod
    def detect(self, img: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Detects the foreground object and returns a binary mask.
        
        Args:
            img: Input image in BGR format (numpy array).
            
        Returns:
            A tuple of (mask, steps) where:
                - mask is a 2D numpy array (0 to 255) representing the foreground.
                - steps is a dictionary containing intermediate step images.
        """
        pass


class ContourForegroundDetector(BaseForegroundDetector):
    """
    Contour-based foreground detector.
    Uses Canny edge detection, morphology operations, contour analysis, 
    and mask smoothing.
    """
    def __init__(
        self,
        canny_low: int = 40,
        canny_high: int = 150,
        dilate_iter: int = 1,
        erode_iter: int = 1,
        mask_dilate_iter: int = 10,
        mask_erode_iter: int = 10,
        mask_blur_size: Tuple[int, int] = (21, 21),
        use_convex_poly: bool = True
    ):
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.dilate_iter = dilate_iter
        self.erode_iter = erode_iter
        self.mask_dilate_iter = mask_dilate_iter
        self.mask_erode_iter = mask_erode_iter
        self.mask_blur_size = mask_blur_size
        self.use_convex_poly = use_convex_poly

    def detect(self, img: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        steps = {}
        
        # 1. Grayscale conversion
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        steps['gray'] = gray.copy()
        
        # 2. Canny Edge Detection
        edges = cv2.Canny(gray, self.canny_low, self.canny_high)
        steps['canny_edge'] = edges.copy()
        
        # 3. Edge Dilation (Morphology)
        if self.dilate_iter > 0:
            edges = cv2.dilate(edges, None, iterations=self.dilate_iter)
            steps['canny_dilate'] = edges.copy()
            
        # 4. Edge Erosion (Morphology)
        if self.erode_iter > 0:
            edges = cv2.erode(edges, None, iterations=self.erode_iter)
            steps['canny_erode'] = edges.copy()
            
        # 5. Find Contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            raise ValueError("No contours found in the image. Try adjusting Canny thresholds.")
            
        # Find the largest contour by area
        contour_info = sorted([(c, cv2.contourArea(c)) for c in contours], key=lambda x: x[1], reverse=True)
        max_contour = contour_info[0][0]
        
        # Draw contour step
        img_draw_contour = cv2.drawContours(gray.copy(), [max_contour], -1, (0, 255, 0), 3)
        steps['img_draw_contour'] = img_draw_contour
        
        # 6. Create Mask
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        if self.use_convex_poly:
            cv2.fillConvexPoly(mask, max_contour, 255)
        else:
            cv2.drawContours(mask, [max_contour], -1, 255, -1)
        steps['fill_mask'] = mask.copy()
        
        # 7. Mask Dilation
        if self.mask_dilate_iter > 0:
            mask = cv2.dilate(mask, None, iterations=self.mask_dilate_iter)
            steps['mask_dilate'] = mask.copy()
            
        # 8. Mask Erosion
        if self.mask_erode_iter > 0:
            mask = cv2.erode(mask, None, iterations=self.mask_erode_iter)
            steps['mask_erode'] = mask.copy()
            
        # 9. Smooth Mask (Gaussian Blur)
        if self.mask_blur_size[0] > 0 and self.mask_blur_size[1] > 0:
            mask = cv2.GaussianBlur(mask, self.mask_blur_size, 0)
            steps['mask_gaussian'] = mask.copy()
            
        return mask, steps


class OtsuForegroundDetector(BaseForegroundDetector):
    """
    Otsu thresholding-based foreground detector.
    Useful for objects that have high contrast with their background.
    """
    def __init__(
        self,
        blur_kernel: Tuple[int, int] = (5, 5),
        dilate_iter: int = 2,
        erode_iter: int = 2,
        mask_blur_size: Tuple[int, int] = (21, 21)
    ):
        self.blur_kernel = blur_kernel
        self.dilate_iter = dilate_iter
        self.erode_iter = erode_iter
        self.mask_blur_size = mask_blur_size

    def detect(self, img: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        steps = {}
        
        # 1. Grayscale conversion
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        steps['gray'] = gray.copy()
        
        # 2. Gaussian Blur to reduce noise
        blurred = cv2.GaussianBlur(gray, self.blur_kernel, 0)
        steps['otsu_blur'] = blurred.copy()
        
        # 3. Otsu Thresholding
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        steps['otsu_threshold'] = thresh.copy()
        
        # 4. Find Contours on thresholded image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            # Fallback to inverse threshold
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not contours:
                raise ValueError("No contours found using Otsu thresholding.")
                
        # Find the largest contour by area
        contour_info = sorted([(c, cv2.contourArea(c)) for c in contours], key=lambda x: x[1], reverse=True)
        max_contour = contour_info[0][0]
        
        # Draw contour step
        img_draw_contour = cv2.drawContours(gray.copy(), [max_contour], -1, (0, 255, 0), 3)
        steps['img_draw_contour'] = img_draw_contour
        
        # 5. Create Mask
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [max_contour], -1, 255, -1)
        steps['fill_mask'] = mask.copy()
        
        # 6. Mask Morphology
        if self.dilate_iter > 0:
            mask = cv2.dilate(mask, None, iterations=self.dilate_iter)
            steps['mask_dilate'] = mask.copy()
        if self.erode_iter > 0:
            mask = cv2.erode(mask, None, iterations=self.erode_iter)
            steps['mask_erode'] = mask.copy()
            
        # 7. Smooth Mask
        if self.mask_blur_size[0] > 0 and self.mask_blur_size[1] > 0:
            mask = cv2.GaussianBlur(mask, self.mask_blur_size, 0)
            steps['mask_gaussian'] = mask.copy()
            
        return mask, steps
