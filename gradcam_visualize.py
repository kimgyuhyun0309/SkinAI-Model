"""
gradcam_visualize.py
- Grad-CAM 생성 및 시각화 모듈
- inference.py + gradcam_setup.py와 함께 사용
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch

from gradcam_setup import (
    gradcam_activations,
    gradcam_gradients,
    register_gradcam_hooks
)
from inference import val_tf, load_model, device


# ============================================================
# Grad-CAM 생성 함수
# ============================================================
def generate_gradcam(
    img_pil,
    model,
    class_idx=None,
    alpha=0.5
):
    """
    이미지(PIL)와 모델을 받아 Grad-CAM heatmap 생성 및 시각화를 진행한다.

    Args:
        img_pil : PIL.Image (원본 이미지)
        model   : EfficientNet 모델
        class_idx : 타겟 클래스 인덱스(없으면 예측 클래스 사용)
        alpha   : Heatmap 투명도

    Returns:
        cam_resized : 원본 이미지 크기에 맞게 리사이즈된 heatmap array
    """
    global gradcam_activations, gradcam_gradients

    # 입력 텐서 변환
    x = val_tf(img_pil).unsqueeze(0).to(device)

    # Grad-CAM 저장 변수 초기화
    gradcam_activations = None
    gradcam_gradients = None

    model.zero_grad()

    # Forward
    logits = model(x)

    # 타깃 클래스 자동 결정
    if class_idx is None:
        class_idx = logits.argmax(1).item()

    # Backward
    score = logits[0, class_idx]
    score.backward()

    # A: activation map, G: gradients
    A = gradcam_activations[0]      # shape = [C, H, W]
    G = gradcam_gradients[0]        # shape = [C, H, W]

    # Weight = 평균 gradient (채널 차원만 평균)
    weights = torch.mean(G, dim=(1, 2), keepdim=True)  # [C, 1, 1]

    # CAM 생성 (weighted sum)
    cam = torch.sum(weights * A, dim=0)   # [H, W]
    cam = torch.relu(cam)

    # 정규화
    cam -= cam.min()
    cam /= (cam.max() + 1e-8)

    cam = cam.cpu().numpy()

    # 원본 이미지 크기로 resize
    cam_resized = cv2.resize(cam, img_pil.size[::-1])

    # 시각화
    plt.figure(figsize=(12, 5))

    # 원본
    plt.subplot(1, 2, 1)
    plt.imshow(img_pil)
    plt.axis("off")
    plt.title("원본 이미지")

    # Grad-CAM overlay
    plt.subplot(1, 2, 2)
    plt.imshow(img_pil)
    plt.imshow(cam_resized, cmap="jet", alpha=alpha)
    plt.axis("off")
    plt.title("Grad-CAM 시각화")

    plt.tight_layout()
    plt.show()

    return cam_resized


# ============================================================
# 모듈 테스트 (직접 실행 시)
# ============================================================
if __name__ == "__main__":
    from inference import predict_image

    # 모델 로드 + hook 등록
    model = load_model()
    register_gradcam_hooks(model)

    # 테스트 이미지 지정
    sample_path = PROJECT_ROOT / "sample.jpg"

    if not sample_path.exists():
        print(f"⚠ 테스트 이미지 없음: {sample_path}")
    else:
        # 예측
        result = predict_image(str(sample_path), model)

        # Grad-CAM 생성
        class_idx = CLASSES.index(result["pred_class"])
        generate_gradcam(result["image"], model, class_idx=class_idx, alpha=0.5)