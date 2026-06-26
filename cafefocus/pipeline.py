import os
import cv2
import numpy as np
from typing import Tuple, Dict, Union
from .detector import BaseForegroundDetector
from .background import BaseBackgroundGenerator
from .blender import BaseBlender

class ImageFocusPipeline:
    """
    Extensible image focusing pipeline.
    Orchestrates the steps:
      1. Foreground object detection and mask generation.
      2. Background processing (blurring, desaturating, etc.).
      3. Blending foreground and background.
    """
    def __init__(
        self,
        detector: BaseForegroundDetector,
        bg_generator: BaseBackgroundGenerator,
        blender: BaseBlender
    ):
        self.detector = detector
        self.bg_generator = bg_generator
        self.blender = blender

    def process(
        self,
        image_input: Union[str, np.ndarray],
        save_steps: bool = False,
        output_dir: str = 'steps'
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Executes the full pipeline.
        
        Args:
            image_input: Path to the image file, or pre-loaded BGR image numpy array.
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
        
        # 1. Detect foreground mask
        mask, detect_steps = self.detector.detect(img)
        all_steps.update(detect_steps)
        
        # 2. Process background
        bg_img, bg_steps = self.bg_generator.generate(img, mask)
        all_steps.update(bg_steps)
        
        # 3. Blend foreground and background
        mixed, blend_steps = self.blender.blend(img, bg_img, mask)
        all_steps.update(blend_steps)
        
        all_steps['mixed'] = mixed
        
        # Save steps if requested
        if save_steps:
            os.makedirs(output_dir, exist_ok=True)
            for step_name, step_img in all_steps.items():
                # Make sure the step image is valid and has correct dimensions
                if step_img is not None and isinstance(step_img, np.ndarray):
                    out_path = os.path.join(output_dir, f"{step_name}.png")
                    cv2.imwrite(out_path, step_img)
                    
        return mixed, all_steps
