import argparse
import sys
import os
from focuser import CafeFocuser

def main():
    parser = argparse.ArgumentParser(
        description="Cafe-Focusing: OpenCV-based image out-focusing utility."
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
    
    # Advanced parameter tweaking
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
        help="Kernel size for background average blur (default: 13)"
    )

    args = parser.parse_args()

    # Input validation
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist.", file=sys.stderr)
        sys.exit(1)
        
    if args.mask_blur % 2 == 0:
        print("Error: --mask-blur must be an odd integer.", file=sys.stderr)
        sys.exit(1)

    print(f"Initializing CafeFocuser with settings:")
    print(f"  - Canny thresholds: ({args.canny_low}, {args.canny_high})")
    print(f"  - Mask dilation/erosion iterations: {args.mask_dilate}/{args.mask_erode}")
    print(f"  - Mask blur size: ({args.mask_blur}, {args.mask_blur})")
    print(f"  - Background blur size: ({args.bg_blur}, {args.bg_blur})")
    print(f"  - Blending mode: {'Legacy (Bitwise AND)' if args.legacy else 'Natural (Alpha Blend)'}")
    print(f"Processing '{args.input}'...")

    try:
        # Instantiate focuser
        focuser = CafeFocuser(
            canny_low=args.canny_low,
            canny_high=args.canny_high,
            mask_dilate_iter=args.mask_dilate,
            mask_erode_iter=args.mask_erode,
            mask_blur_size=(args.mask_blur, args.mask_blur),
            bg_blur_size=(args.bg_blur, args.bg_blur)
        )
        
        # Execute pipeline
        use_alpha = not args.legacy
        mixed, all_steps = focuser.process(
            image_input=args.input,
            use_alpha_blend=use_alpha,
            save_steps=args.save_steps,
            output_dir=args.steps_dir
        )
        
        # Save final result
        import cv2
        cv2.imwrite(args.output, mixed)
        print(f"Success! Saved final image to '{args.output}'")
        
        if args.save_steps:
            print(f"Saved intermediate step images in directory: '{args.steps_dir}'")
            
    except Exception as e:
        print(f"An error occurred during image processing: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
