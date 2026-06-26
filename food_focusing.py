#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Cafe-Focusing Tutorial Demonstration Script
===========================================
이 스크립트는 기존 `food_focusing.ipynb` 주피터 노트북을 파이썬 스크립트(`.py`)로 
변환하고, 새로 리팩터링된 `cafefocus` 패키지의 확장 구조를 적용한 데모 프로그램입니다.

본 예제는 다음을 수행합니다:
1. 기본 OpenCV 모듈과 새로 설계된 `cafefocus` 모듈 로드.
2. 예제 이미지 (`example_process_img/food_solo.png`) 로드.
3. 기존 `CafeFocuser` 및 새로운 `ImageFocusPipeline`을 사용한 이미지 처리.
4. 처리 방식 비교 결과와 내부 공정 단계를 Matplotlib 차트로 생성하고 파일로 저장.
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 패키지 및 모듈 가져오기
from cafefocus.detector import ContourForegroundDetector, OtsuForegroundDetector
from cafefocus.background import BlurBackgroundGenerator, DesaturateBackgroundGenerator
from cafefocus.blender import AlphaBlender, LegacyAndBlender
from cafefocus.pipeline import ImageFocusPipeline

def main():
    print("=== Cafe-Focusing 확장 모듈 데모 및 시각화 스크립트 ===")
    
    # 2. 입력 이미지 준비
    img_path = 'example_process_img/food_solo.png'
    if not os.path.exists(img_path):
        img_path = 'food_solo.png'
        
    if not os.path.exists(img_path):
        print(f"오류: 테스트용 예제 이미지 '{img_path}'를 찾을 수 없습니다.")
        sys.exit(1)
        
    print(f"이미지 로드 중: {img_path}")
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # 원본 이미지 시각화용 단독 창 저장
    plt.figure(figsize=(6, 6))
    plt.imshow(img_rgb)
    plt.title('Original Image')
    plt.axis('off')
    original_plot_path = os.path.join('example_process_img', 'original_image_plot.png')
    plt.savefig(original_plot_path, bbox_inches='tight')
    plt.close()
    print(f"-> 원본 이미지 차트가 저장되었습니다: {original_plot_path}")

    # 3. 아웃포커싱 처리 실행 (모듈형 ImageFocusPipeline 사용)
    print("공통 전경 탐지기 및 배경 블러 생성기 설정 중...")
    detector = ContourForegroundDetector(
        canny_low=40,
        canny_high=150,
        mask_dilate_iter=10,
        mask_erode_iter=10,
        mask_blur_size=(21, 21)
    )
    bg_generator = BlurBackgroundGenerator(
        blur_type='average',
        blur_size=(13, 13)
    )

    # 1) 레거시 AND 마스크 방식 파이프라인 실행
    print("레거시 (Bitwise AND) 합성 파이프라인 구동...")
    legacy_pipeline = ImageFocusPipeline(
        detector=detector,
        bg_generator=bg_generator,
        blender=LegacyAndBlender(mask_color=(1.0, 1.0, 1.0), bg_blur_size=(13, 13))
    )
    mixed_legacy, steps_legacy = legacy_pipeline.process(img_bgr)
    mixed_legacy_rgb = cv2.cvtColor(mixed_legacy, cv2.COLOR_BGR2RGB)

    # 2) 개선된 알파 블렌딩 방식 파이프라인 실행
    print("개선된 (Alpha Blend) 합성 파이프라인 구동...")
    alpha_pipeline = ImageFocusPipeline(
        detector=detector,
        bg_generator=bg_generator,
        blender=AlphaBlender()
    )
    mixed_alpha, steps_alpha = alpha_pipeline.process(img_bgr)
    mixed_alpha_rgb = cv2.cvtColor(mixed_alpha, cv2.COLOR_BGR2RGB)

    # 4. 새로 리팩터링된 Extensible Pipeline 데모 (예: Otsu 검출기 + 채도 감쇄 배경)
    print("\n새로운 확장형 파이프라인 (Otsu Detector + Desaturated Background) 실행 중...")
    
    custom_pipeline = ImageFocusPipeline(
        detector=OtsuForegroundDetector(
            blur_kernel=(5, 5),
            dilate_iter=2,
            erode_iter=2,
            mask_blur_size=(21, 21)
        ),
        bg_generator=DesaturateBackgroundGenerator(
            saturation_factor=0.2, # 배경 채도를 20%로 낮춤 (거의 흑백)
            blur_generator=BlurBackgroundGenerator(blur_type='gaussian', blur_size=(15, 15))
        ),
        blender=AlphaBlender()
    )
    
    mixed_custom, steps_custom = custom_pipeline.process(img_bgr)
    mixed_custom_rgb = cv2.cvtColor(mixed_custom, cv2.COLOR_BGR2RGB)

    # 5. 아웃포커싱 결과 비교 시각화 차트 생성
    print("\n결과 비교 시각화 이미지 생성 중...")
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(img_rgb)
    axes[0].set_title('1. Original Image')
    axes[0].axis('off')

    axes[1].imshow(mixed_legacy_rgb)
    axes[1].set_title('2. Legacy Focusing\n(Bitwise AND)')
    axes[1].axis('off')

    axes[2].imshow(mixed_alpha_rgb)
    axes[2].set_title('3. Improved Focusing\n(Alpha Blend)')
    axes[2].axis('off')

    axes[3].imshow(mixed_custom_rgb)
    axes[3].set_title('4. Custom Extensible Pipeline\n(Otsu + Desaturated BG)')
    axes[3].axis('off')

    plt.tight_layout()
    comparison_plot_path = os.path.join('example_process_img', 'focusing_comparison_result.png')
    plt.savefig(comparison_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"-> 아웃포커싱 비교 결과 그래프가 저장되었습니다: {comparison_plot_path}")

    # 6. 알파 블렌딩 파이프라인 상세 처리 단계 시각화
    print("알파 블렌딩 상세 처리 단계 시각화 이미지 생성 중...")
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.ravel()

    # 시각화할 단계 매핑
    visual_steps = [
        ('gray', '1. Grayscale', 'gray'),
        ('canny_edge', '2. Canny Edge', 'gray'),
        ('canny_dilate', '3. Dilated Edge', 'gray'),
        ('img_draw_contour', '4. Contour Detected', 'gray'),
        ('fill_mask', '5. Convex Mask', 'gray'),
        ('mask_gaussian', '6. Smoothed Mask', 'gray'),
        ('img_blur', '7. Blurred Background (Alpha)', None),
        ('mixed', '8. Final Composite (Alpha)', None)
    ]

    for i, (key, title, cmap) in enumerate(visual_steps):
        if key in steps_alpha:
            img_step = steps_alpha[key]
            # RGB 변환 처리 (컬러 채널이 있는 경우)
            if len(img_step.shape) == 3:
                img_step = cv2.cvtColor(img_step, cv2.COLOR_BGR2RGB)
                
            axes[i].imshow(img_step, cmap=cmap)
            axes[i].set_title(title)
            axes[i].axis('off')
        else:
            axes[i].text(0.5, 0.5, f"Step '{key}'\nNot Found", ha='center', va='center')
            axes[i].axis('off')

    plt.tight_layout()
    steps_plot_path = os.path.join('example_process_img', 'pipeline_processing_steps.png')
    plt.savefig(steps_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"-> 상세 파이프라인 단계 그래프가 저장되었습니다: {steps_plot_path}")
    print("\n데모 실행이 완료되었습니다! 모든 결과가 이미지 파일로 저장되었습니다.")

if __name__ == "__main__":
    main()
