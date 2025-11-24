"""
inference.py
- 학습된 EfficientNet-B2 모델을 로드하여
  단일 이미지에 대해 피부 상태를 예측하고,
  한국어 추천 문구까지 반환하는 추론 스크립트
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
import timm

from config import PROJECT_ROOT, CLASSES
from recommendations import RECOMMENDATIONS
from train_fast_b2 import val_tf, device  # 동일한 전처리와 device 재사용


# ============================================================
# 1) 모델 로딩 함수
# ============================================================
def load_model(
    model_path: Path = PROJECT_ROOT / "models" / "effb2_skin_fast.pth",
    model_name: str = "tf_efficientnet_b2_ns",
):
    """
    저장된 가중치를 불러와서 EfficientNet-B2 모델을 생성하고
    eval 모드로 반환한다.
    """
    model = timm.create_model(
        model_name,
        pretrained=False,
        num_classes=len(CLASSES),
    ).to(device)

    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    print(f"📌 모델 로드 완료: {model_path}")
    return model


# ============================================================
# 2) 단일 이미지 예측 함수
# ============================================================
def predict_image(
    image_path: str,
    model=None,
):
    """
    image_path: 예측할 이미지 파일 경로 (str 또는 Path)
    model: 이미 로드된 모델 객체 (None이면 내부에서 자동 로드)

    반환:
      dict = {
        "image": 원본 PIL.Image,
        "pred_class": 예측 클래스명 (str),
        "confidence": 신뢰도 (float),
        "probs": 전체 클래스 softmax 확률 (np.ndarray),
        "advice_ko": 한국어 추천 문구 (str)
      }
    """
    if model is None:
        model = load_model()

    img_path = Path(image_path)
    img = Image.open(img_path).convert("RGB")

    # 전처리
    x = val_tf(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    pred_idx = int(np.argmax(probs))
    pred_class = CLASSES[pred_idx]
    confidence = float(probs[pred_idx])

    # 추천 문구
    advice_ko = RECOMMENDATIONS.get(
        pred_class,
        "해당 유형에 대한 추천 문구가 아직 준비되지 않았습니다.",
    )

    return {
        "image": img,
        "pred_class": pred_class,
        "confidence": confidence,
        "probs": probs,
        "advice_ko": advice_ko,
    }


# ============================================================
# 3) 테스트용 실행부
# ============================================================
if __name__ == "__main__":
    # 예시 경로 (직접 수정해서 사용)
    sample_image = PROJECT_ROOT / "sample.jpg"

    if not sample_image.exists():
        print(f"⚠ 테스트용 이미지가 없습니다: {sample_image}")
    else:
        model = load_model()
        result = predict_image(str(sample_image), model)

        print("\n=== 예측 결과 ===")
        print("예측 클래스 :", result["pred_class"])
        print("신뢰도      :", f"{result['confidence']:.3f}")
        print("\n[추천 문구]\n")
        print(result["advice_ko"])