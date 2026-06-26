import argparse
import sys
import os
import cv2

from cafefocus.detector import ContourForegroundDetector, OtsuForegroundDetector
from cafefocus.background import BlurBackgroundGenerator, DesaturateBackgroundGenerator, DarkenBackgroundGenerator
from cafefocus.blender import AlphaBlender, LegacyAndBlender
from cafefocus.pipeline import ImageFocusPipeline

def main():
    parser = argparse.ArgumentParser(
        description="Cafe-Focusing: Extensible OpenCV-based image out-focusing utility."
    )
    
    # Required arguments
    parser.add_argument(
        "input", 
        type=str, 
        help="Path to the input image file (e.g. food_solo.png)"
    )
    
    # Optional arguments
    parser.add_argument(
        "-o", "--output", 
        type=str, 
        default="focused_result.png", 
        help="Path to save the final blended image (default: focused_result.png)"
    )
    parser.add_argument(
        "--legacy", 
        action="store_true", 
        help="Use the legacy bitwise_and blending method instead of natural alpha blending"
    )
    parser.add_argument(
        "--save-steps", 
        action="store_true", 
        help="Save intermediate steps (canny, mask, blur, etc.) to a directory"
    )
    parser.add_argument(
        "--steps-dir", 
        type=str, 
        default="processing_steps", 
        help="Directory to save intermediate steps if --save-steps is set"
    )
    
    # Detector configuration
    parser.add_argument(
        "--detector",
        type=str,
        choices=["contour", "otsu"],
        default="contour",
        help="Foreground detection method to use (default: contour)"
    )
    
    # Background effect configuration
    parser.add_argument(
        "--bg-effect",
        type=str,
        choices=["blur", "desaturate", "darken"],
        default="blur",
        help="Background effect to apply (default: blur)"
    )
    parser.add_argument(
        "--bg-blur-type",
        type=str,
        choices=["average", "gaussian"],
        default="average",
        help="Type of blur to apply to background (default: average)"
    )
    parser.add_argument(
        "--saturation",
        type=float,
        default=0.3,
        help="Saturation level for desaturated background (0.0 to 1.0, default: 0.3)"
    )
    parser.add_argument(
        "--brightness",
        type=float,
        default=0.6,
        help="Brightness level for darkened background (0.0 to 1.0, default: 0.6)"
    )
    
    # Advanced parameter tweaking for Contour detector
    parser.add_argument(
        "--canny-low", 
        type=int, 
        default=40, 
        help="Canny edge detection low threshold (default: 40)"
    )
    parser.add_argument(
        "--canny-high", 
        type=int, 
        default=150, 
        help="Canny edge detection high threshold (default: 150)"
    )
    parser.add_argument(
        "--mask-dilate", 
        type=int, 
        default=10, 
        help="Iterations for mask dilation (default: 10)"
    )
    parser.add_argument(
        "--mask-erode", 
        type=int, 
        default=10, 
        help="Iterations for mask erosion (default: 10)"
    )
    parser.add_argument(
        "--mask-blur", 
        type=int, 
        default=21, 
        help="Kernel size for mask Gaussian blur (must be odd, default: 21)"
    )
    parser.add_argument(
        "--bg-blur", 
        type=int, 
        default=13, 
        help="Kernel size for background blur (default: 13)"
    )

    args = parser.parse_args()

    # Input validation
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist.", file=sys.stderr)
        sys.exit(1)
        
    if args.mask_blur % 2 == 0:
        print("Error: --mask-blur must be an odd integer.", file=sys.stderr)
        sys.exit(1)

    print(f"Initializing CafeFocus Pipeline...")
    
    # 1. Setup Detector
    if args.detector == "contour":
        print(f"  - Detector: ContourForegroundDetector")
        print(f"    * Canny thresholds: ({args.canny_low}, {args.canny_high})")
        print(f"    * Mask dilation/erosion iterations: {args.mask_dilate}/{args.mask_erode}")
        print(f"    * Mask blur size: ({args.mask_blur}, {args.mask_blur})")
        detector = ContourForegroundDetector(
            canny_low=args.canny_low,
            canny_high=args.canny_high,
            mask_dilate_iter=args.mask_dilate,
            mask_erode_iter=args.mask_erode,
            mask_blur_size=(args.mask_blur, args.mask_blur)
        )
    else: # otsu
        print(f"  - Detector: OtsuForegroundDetector")
        print(f"    * Mask blur size: ({args.mask_blur}, {args.mask_blur})")
        detector = OtsuForegroundDetector(
            mask_blur_size=(args.mask_blur, args.mask_blur)
        )
        
    # 2. Setup Background Generator
    base_blur_generator = BlurBackgroundGenerator(
        blur_type=args.bg_blur_type,
        blur_size=(args.bg_blur, args.bg_blur)
    )
    
    if args.bg_effect == "blur":
        print(f"  - Background Effect: Blur ({args.bg_blur_type})")
        print(f"    * Blur size: ({args.bg_blur}, {args.bg_blur})")
        bg_generator = base_blur_generator
    elif args.bg_effect == "desaturate":
        print(f"  - Background Effect: Desaturate (Saturation factor: {args.saturation}) + Blur")
        bg_generator = DesaturateBackgroundGenerator(
            saturation_factor=args.saturation,
            blur_generator=base_blur_generator
        )
    else: # darken
        print(f"  - Background Effect: Darken (Brightness factor: {args.brightness}) + Blur")
        bg_generator = DarkenBackgroundGenerator(
            brightness_factor=args.brightness,
            blur_generator=base_blur_generator
        )
        
    # 3. Setup Blender
    if args.legacy:
        print(f"  - Blender: LegacyAndBlender (Bitwise AND)")
        blender = LegacyAndBlender(
            bg_blur_size=(args.bg_blur, args.bg_blur)
        )
    else:
        print(f"  - Blender: AlphaBlender (Natural Alpha Blend)")
        blender = AlphaBlender()

    # 4. Construct and Run Pipeline
    print(f"Processing '{args.input}'...")
    try:
        pipeline = ImageFocusPipeline(
            detector=detector,
            bg_generator=bg_generator,
            blender=blender
        )
        
        mixed, all_steps = pipeline.process(
            image_input=args.input,
            save_steps=args.save_steps,
            output_dir=args.steps_dir
        )
        
        # Save final result
        cv2.imwrite(args.output, mixed)
        print(f"Success! Saved final image to '{args.output}'")
        
        if args.save_steps:
            print(f"Saved intermediate step images in directory: '{args.steps_dir}'")
            
    except Exception as e:
        print(f"An error occurred during image processing: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
