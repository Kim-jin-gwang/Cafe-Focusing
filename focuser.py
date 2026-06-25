import cv2
import numpy as np
import os
from typing import Tuple, Dict, Any, Union

class CafeFocuser:
    """
    OpenCV-based image out-focusing processor.
    Extracts the main foreground object (specifically for cafe food/drinks) 
    using edge detection and contour analysis, then blurs the background.
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
        bg_blur_size: Tuple[int, int] = (13, 13),
        mask_color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    ):
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.dilate_iter = dilate_iter
        self.erode_iter = erode_iter
        self.mask_dilate_iter = mask_dilate_iter
        self.mask_erode_iter = mask_erode_iter
        self.mask_blur_size = mask_blur_size
        self.bg_blur_size = bg_blur_size
        self.mask_color = mask_color

    def detect_edges(self, gray_img: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Applies Canny edge detection, dilation, and erosion."""
        steps = {}
        
        # 1. Edge detection
        edges = cv2.Canny(gray_img, self.canny_low, self.canny_high)
        steps['canny_edge'] = edges.copy()
        
        # 2. Dilation
        if self.dilate_iter > 0:
            edges = cv2.dilate(edges, None, iterations=self.dilate_iter)
            steps['canny_dilate'] = edges.copy()
            
        # 3. Erosion
        if self.erode_iter > 0:
            edges = cv2.erode(edges, None, iterations=self.erode_iter)
            steps['canny_erode'] = edges.copy()
            
        return edges, steps

    def find_largest_contour(self, edges: np.ndarray) -> np.ndarray:
        """Finds all external contours and returns the largest one by area."""
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            raise ValueError("No contours found in the image. Try adjusting Canny thresholds.")
            
        contour_info = []
        for c in contours:
            contour_info.append((c, cv2.contourArea(c)))
        
        # Sort by area in descending order
        contour_info = sorted(contour_info, key=lambda x: x[1], reverse=True)
        return contour_info[0][0]

    def create_and_smooth_mask(
        self, 
        image_shape: Tuple[int, ...], 
        contour: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Creates a binary mask of the contour and applies smoothing/blurring."""
        steps = {}
        
        # 1. Create empty mask and fill the polygon
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, contour, 255)
        steps['fill_mask'] = mask.copy()
        
        # 2. Dilation
        if self.mask_dilate_iter > 0:
            mask = cv2.dilate(mask, None, iterations=self.mask_dilate_iter)
            steps['mask_dilate'] = mask.copy()
            
        # 3. Erosion
        if self.mask_erode_iter > 0:
            mask = cv2.erode(mask, None, iterations=self.mask_erode_iter)
            steps['mask_erode'] = mask.copy()
            
        # 4. Gaussian Blur to soften edges
        if self.mask_blur_size[0] > 0 and self.mask_blur_size[1] > 0:
            mask = cv2.GaussianBlur(mask, self.mask_blur_size, 0)
            steps['mask_gaussian'] = mask.copy()
            
        return mask, steps

    def blend_legacy(
        self, 
        img: np.ndarray, 
        mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """
        Original blend logic using BGR image and mask:
        - Extracts object (background set to MASK_COLOR).
        - Extracts background (object set to MASK_COLOR).
        - Blurs the background image.
        - Blends them using bitwise_and.
        """
        steps = {}
        mask_stack = np.dstack([mask] * 3).astype('float32') / 255.0
        img_float = img.astype('float32') / 255.0
        
        # Object extraction
        masked = (mask_stack * img_float) + ((1 - mask_stack) * self.mask_color)
        masked = (masked * 255).astype('uint8')
        steps['only_coffee'] = masked.copy()
        
        # Background extraction
        back_masked = ((1 - mask_stack) * img_float) + (mask_stack * self.mask_color)
        back_masked = (back_masked * 255).astype('uint8')
        steps['background'] = back_masked.copy()
        
        # Blur background
        img_blur = cv2.blur(back_masked, self.bg_blur_size)
        steps['img_blur'] = img_blur.copy()
        
        # Composite via bitwise AND
        mixed = cv2.bitwise_and(img_blur, masked)
        
        return mixed, masked, img_blur, steps

    def blend_alpha(
        self, 
        img: np.ndarray, 
        mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """
        Improved blend logic using direct Alpha Blending:
        - Blurs the entire original image.
        - Blends original image (foreground) and blurred image (background)
          using the smoothed mask as the alpha channel.
        This avoids hard edge artifacts and color bleeding from bitwise_and.
        """
        steps = {}
        # Normalize mask to 0.0 - 1.0
        alpha = np.dstack([mask] * 3).astype('float32') / 255.0
        
        # Blur the entire original image
        blurred_bg = cv2.blur(img, self.bg_blur_size)
        steps['img_blur'] = blurred_bg.copy()
        
        # Alpha blend: output = alpha * foreground + (1 - alpha) * background
        img_float = img.astype('float32')
        bg_float = blurred_bg.astype('float32')
        
        mixed = (alpha * img_float) + ((1 - alpha) * bg_float)
        mixed = np.clip(mixed, 0, 255).astype('uint8')
        
        # For compatibility/informational steps
        steps['only_coffee'] = (alpha * img_float + (1 - alpha) * 255.0).astype('uint8')
        
        return mixed, blurred_bg, steps

    def process(
        self, 
        image_input: Union[str, np.ndarray], 
        use_alpha_blend: bool = True,
        save_steps: bool = False,
        output_dir: str = 'steps'
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Executes the full focusing pipeline.
        
        Args:
            image_input: Path to the image file, or pre-loaded BGR image numpy array.
            use_alpha_blend: If True, uses the new natural alpha blending.
                             If False, uses the legacy bitwise_and method.
            save_steps: If True, saves all intermediate images to output_dir.
            output_dir: Directory path where intermediate steps will be saved.
            
        Returns:
            A tuple of (final_mixed_image, dict_of_all_steps_images)
        """
        # Load image
        if isinstance(image_input, str):
            img = cv2.imread(image_input)
            if img is None:
                raise FileNotFoundError(f"Could not read image from path: {image_input}")
        else:
            img = image_input.copy()
            
        all_steps = {'original': img.copy()}
        
        # Gray scale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        all_steps['gray'] = gray.copy()
        
        # Edge Detection
        edges, edge_steps = self.detect_edges(gray)
        all_steps.update(edge_steps)
        
        # Find Contour
        max_contour = self.find_largest_contour(edges)
        
        # Draw contour step
        img_draw_contour = cv2.drawContours(gray.copy(), [max_contour], -1, (0, 255, 0), 3)
        all_steps['img_draw_contour'] = img_draw_contour
        
        # Create and smooth mask
        mask, mask_steps = self.create_and_smooth_mask(img.shape, max_contour)
        all_steps.update(mask_steps)
        
        # Blend
        if use_alpha_blend:
            mixed, blurred_bg, blend_steps = self.blend_alpha(img, mask)
            all_steps.update(blend_steps)
        else:
            mixed, masked, blurred_bg, blend_steps = self.blend_legacy(img, mask)
            all_steps.update(blend_steps)
            
        all_steps['mixed'] = mixed
        
        # Save steps if requested
        if save_steps:
            os.makedirs(output_dir, exist_ok=True)
            for step_name, step_img in all_steps.items():
                out_path = os.path.join(output_dir, f"{step_name}.png")
                cv2.imwrite(out_path, step_img)
                
        return mixed, all_steps
