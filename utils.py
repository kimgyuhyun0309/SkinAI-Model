"""
utils.py
- SkinAI 프로젝트 공용 유틸리티 함수 모음
  1) 이미지 SHA-1 계산
  2) 이미지 크기 정규화
  3) 흐릿한 이미지 필터(라플라시안 분산)
  4) 안전한 이미지 로딩 함수
"""

import hashlib
from pathlib import Path
from PIL import Image
import numpy as np
import cv2


# ============================================================
# SHA-1 계산
# ============================================================
def sha1(path: Path) -> str:
    """파일 내용을 기반으로 SHA-1 해시 생성"""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ============================================================
# 이미지 안전 로딩
# ============================================================
def safe_load_pil(path: Path):
    """
    PIL로 이미지를 안전하게 로드.
    로딩 실패 시 None 반환.
    """
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


# ============================================================
# 이미지 크기 정규화 함수
# ============================================================
def resize_image(pil_img, min_size=256, max_size=768):
    """
    입력 이미지(PIL.Image)를 조건에 맞게 축소/정규화
    - min_size: 최소 변 길이가 이보다 작은 이미지는 None 반환
    - max_size: 최대 변 길이가 이보다 크면 축소
    """
    w, h = pil_img.size

    # 작은 이미지 제거
    if min(w, h) < min_size:
        return None

    # 너무 큰 이미지는 축소
    scale = max_size / max(w, h)
    if scale < 1.0:
        pil_img = pil_img.resize((int(w * scale), int(h * scale)))

    return pil_img


# ============================================================
# 흐릿한 이미지 제거 — 라플라시안 분산
# ============================================================
def variance_of_laplacian(image):
    """
    라플라시안 분산 계산
    값이 낮을수록 이미지가 흐릿함
    """
    return cv2.Laplacian(image, cv2.CV_64F).var()


def is_blurry_image(path: Path, threshold=20.0):
    """
    이미지가 흐릿한지 검사
    True  → blurry (제거해야 함)
    False → 정상
    """
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return True  # 로드 실패 시 제거

    score = variance_of_laplacian(img)
    return score < threshold


# ============================================================
# 디버깅용: 이미지 크기 출력
# ============================================================
def print_image_info(path: Path):
    """이미지 크기 및 기타 정보를 출력"""
    img = safe_load_pil(path)
    if img is None:
        print(f"[X] 이미지 로딩 실패: {path}")
        return
    print(f"[OK] {path} | size={img.size}, mode={img.mode}")