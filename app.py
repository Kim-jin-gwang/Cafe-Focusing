"""Gradio live demo for Cafe-Focusing.

Wraps the cafefocus pipeline in a web UI: upload an image, tune the
detection/background parameters, and get an out-focused result with
optional intermediate step visualization.

Runs locally (`python app.py`) and on Hugging Face Spaces as-is.
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import cv2
import numpy as np
import gradio as gr

try:
    # Hugging Face ZeroGPU 환경에서는 @spaces.GPU 데코레이터가 최소 1개 필요
    import spaces
    _gpu = spaces.GPU
except ImportError:  # 로컬 실행: 데코레이터를 no-op으로 대체
    def _gpu(fn):
        return fn

from cafefocus.detector import ContourForegroundDetector, OtsuForegroundDetector
from cafefocus.background import (
    BlurBackgroundGenerator,
    DesaturateBackgroundGenerator,
    DarkenBackgroundGenerator,
)
from cafefocus.blender import AlphaBlender, LegacyAndBlender
from cafefocus.pipeline import ImageFocusPipeline

MAX_SIDE = 1600  # 데모 서버 보호: 큰 이미지는 축소 처리

STEP_LABELS = {
    "original": "원본",
    "gray": "그레이스케일",
    "canny": "Canny 엣지",
    "edges": "Canny 엣지",
    "dilate": "팽창(dilate)",
    "erode": "침식(erode)",
    "contour": "윤곽선",
    "fill_mask": "채운 마스크",
    "mask_dilate": "마스크 팽창",
    "mask_erode": "마스크 침식",
    "mask_blur": "마스크 블러",
    "mask": "최종 마스크",
    "otsu": "Otsu 이진화",
    "blur": "배경 블러",
    "bg": "배경 처리 결과",
    "mixed": "최종 결과",
}


def _to_rgb(img: np.ndarray) -> np.ndarray:
    if img is None:
        return None
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


@_gpu
def focus_image(
    image,
    detector_name,
    bg_effect,
    blur_type,
    canny_low,
    canny_high,
    mask_dilate,
    mask_erode,
    mask_blur,
    bg_blur,
    saturation,
    brightness,
    legacy_blend,
    show_steps,
):
    if image is None:
        raise gr.Error("이미지를 먼저 업로드해주세요.")

    # Gradio는 RGB, 파이프라인은 BGR
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    h, w = bgr.shape[:2]
    if max(h, w) > MAX_SIDE:
        scale = MAX_SIDE / max(h, w)
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    mask_blur = int(mask_blur)
    if mask_blur % 2 == 0:
        mask_blur += 1
    bg_blur = int(bg_blur)

    if detector_name == "윤곽선 (Contour)":
        detector = ContourForegroundDetector(
            canny_low=int(canny_low),
            canny_high=int(canny_high),
            mask_dilate_iter=int(mask_dilate),
            mask_erode_iter=int(mask_erode),
            mask_blur_size=(mask_blur, mask_blur),
        )
    else:
        detector = OtsuForegroundDetector(mask_blur_size=(mask_blur, mask_blur))

    base_blur = BlurBackgroundGenerator(
        blur_type="gaussian" if blur_type == "가우시안" else "average",
        blur_size=(bg_blur, bg_blur),
    )
    if bg_effect == "블러":
        bg_generator = base_blur
    elif bg_effect == "채도 감소":
        bg_generator = DesaturateBackgroundGenerator(
            saturation_factor=float(saturation), blur_generator=base_blur
        )
    else:  # 어둡게
        bg_generator = DarkenBackgroundGenerator(
            brightness_factor=float(brightness), blur_generator=base_blur
        )

    blender = LegacyAndBlender(bg_blur_size=(bg_blur, bg_blur)) if legacy_blend else AlphaBlender()

    pipeline = ImageFocusPipeline(detector=detector, bg_generator=bg_generator, blender=blender)
    mixed, all_steps = pipeline.process(image_input=bgr)

    gallery = []
    if show_steps:
        for name, step_img in all_steps.items():
            if step_img is None or not isinstance(step_img, np.ndarray):
                continue
            label = STEP_LABELS.get(name, name)
            gallery.append((_to_rgb(step_img), label))

    return _to_rgb(mixed), gallery


with gr.Blocks(title="Cafe-Focusing") as demo:
    gr.Markdown(
        """
        # ☕ Cafe-Focusing
        OpenCV 윤곽선 분석으로 피사체의 정확한 외곽을 찾아 **배경만 블러 처리**하는 아웃포커싱 데모입니다.
        휴대폰 카메라의 고정된 원형 포커스 필터와 달리, 불규칙한 형태의 피사체(음료, 디저트 등)에 정밀하게 맞춰집니다.

        📎 [GitHub 저장소](https://github.com/Kim-jin-gwang/Cafe-Focusing) ·
        🌐 [전체 프로젝트 보기](https://demo-gateway.trealight112.workers.dev/)
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(label="입력 이미지", type="numpy")
            detector_name = gr.Radio(
                ["윤곽선 (Contour)", "Otsu 이진화"], value="윤곽선 (Contour)", label="피사체 검출 방식"
            )
            bg_effect = gr.Radio(["블러", "채도 감소", "어둡게"], value="블러", label="배경 효과")
            run_btn = gr.Button("아웃포커싱 적용", variant="primary")

            with gr.Accordion("고급 설정", open=False):
                blur_type = gr.Radio(["평균", "가우시안"], value="평균", label="블러 종류")
                canny_low = gr.Slider(0, 255, value=40, step=1, label="Canny 하한 임계값")
                canny_high = gr.Slider(0, 255, value=150, step=1, label="Canny 상한 임계값")
                mask_dilate = gr.Slider(0, 30, value=10, step=1, label="마스크 팽창 반복 횟수")
                mask_erode = gr.Slider(0, 30, value=10, step=1, label="마스크 침식 반복 횟수")
                mask_blur = gr.Slider(1, 51, value=21, step=2, label="마스크 블러 커널 (홀수)")
                bg_blur = gr.Slider(3, 51, value=13, step=2, label="배경 블러 커널")
                saturation = gr.Slider(0.0, 1.0, value=0.3, step=0.05, label="채도 (채도 감소 효과)")
                brightness = gr.Slider(0.0, 1.0, value=0.6, step=0.05, label="밝기 (어둡게 효과)")
                legacy_blend = gr.Checkbox(value=False, label="레거시 블렌딩 (bitwise AND)")
                show_steps = gr.Checkbox(value=True, label="중간 처리 단계 보기")

        with gr.Column(scale=1):
            output_image = gr.Image(label="아웃포커싱 결과")
            steps_gallery = gr.Gallery(label="처리 단계", columns=4, height=300)

    inputs = [
        input_image, detector_name, bg_effect, blur_type,
        canny_low, canny_high, mask_dilate, mask_erode, mask_blur, bg_blur,
        saturation, brightness, legacy_blend, show_steps,
    ]
    run_btn.click(focus_image, inputs=inputs, outputs=[output_image, steps_gallery])

    gr.Examples(
        examples=[[os.path.join(BASE_DIR, "example_process_img", "food_solo.png")]],
        inputs=[input_image],
        label="예시 이미지",
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
