"""
metadata_builder.py
- SkinAI 프로젝트의 원본 이미지로부터
  1) manifest_raw.tsv : 파일 경로 + 클래스 + SHA-1 해시
  2) normalized 이미지 저장 + manifest_normalized.tsv
  3) blur(흐릿함) 제거 + manifest_clean.tsv

  이 모든 과정을 수행하는 데이터 전처리 스크립트입니다.
"""

import csv
import hashlib
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import cv2

from config import PROJECT_ROOT, CLASSES


# ============================================================
# 1) 기본 경로 설정
# ============================================================
SOURCE_DIR = PROJECT_ROOT / "images"
MANIFEST_RAW = PROJECT_ROOT / "manifest_raw.tsv"
MANIFEST_NORMALIZED = PROJECT_ROOT / "manifest_normalized.tsv"
MANIFEST_CLEAN = PROJECT_ROOT / "manifest_clean.tsv"

NORMALIZED_DIR = PROJECT_ROOT / "normalized"
NORMALIZED_DIR.mkdir(exist_ok=True)


# ============================================================
# 2) SHA-1 해시 함수
# ============================================================
def sha1(path: Path) -> str:
    """파일 내용을 기반으로 SHA-1 해시 생성"""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ============================================================
# 3) Step 1 — manifest_raw.tsv 생성
# ============================================================
def build_manifest_raw():

    rows = []
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    for cls in CLASSES:
        cls_dir = SOURCE_DIR / cls
        for ext in exts:
            for p in cls_dir.rglob(f"*{ext}"):
                rows.append({
                    "filepath": str(p),
                    "class": cls,
                    "hash": sha1(p)
                })

    with open(MANIFEST_RAW, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["filepath", "class", "hash"], delimiter="\t"
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ Step1: manifest_raw.tsv 생성 완료 ({len(rows)}개)")


# ============================================================
# 4) Step 2 — 이미지 정규화 & manifest_normalized.tsv
# ============================================================
def normalize_images(min_size=256, target_max=768):

    df = pd.read_csv(MANIFEST_RAW, sep="\t")
    out_rows = []

    for _, row in df.iterrows():
        src = Path(row.filepath)
        if not src.exists():
            continue

        try:
            im = Image.open(src).convert("RGB")
            w, h = im.size

            # 작은 이미지 제거
            if min(w, h) < min_size:
                continue

            # 이미지 크기 조절
            scale = target_max / max(w, h)
            if scale < 1.0:
                im = im.resize((int(w * scale), int(h * scale)))

            # 저장 경로
            cls_dir = NORMALIZED_DIR / row["class"]
            cls_dir.mkdir(parents=True, exist_ok=True)
            out_path = cls_dir / f"{row['hash']}.jpg"

            im.save(out_path, "JPEG", quality=92, optimize=True)

            out_rows.append({
                "filepath": str(out_path),
                "class": row["class"],
                "hash": row["hash"]
            })

        except Exception:
            continue

    pd.DataFrame(out_rows).to_csv(MANIFEST_NORMALIZED, sep="\t", index=False)

    print(f"✅ Step2: normalized 이미지 생성 완료 ({len(out_rows)}개)")
    print(f"📄 파일: {MANIFEST_NORMALIZED}")


# ============================================================
# 5) Step 3 — 흐릿한 이미지 제거 + manifest_clean.tsv
# ============================================================
def variance_of_laplacian(image):
    """흐릿한 이미지 판별용 라플라시안 분산 계산"""
    return cv2.Laplacian(image, cv2.CV_64F).var()


def clean_blurry_images(blur_threshold=20.0):

    df = pd.read_csv(MANIFEST_NORMALIZED, sep="\t")
    print(f"정규화된 이미지 개수: {len(df)}")

    # 해시 기준 중복 제거
    df = df.drop_duplicates(subset="hash", keep="first")
    print(f"중복 제거 후: {len(df)}")

    keep_rows = []

    for _, row in df.iterrows():
        p = row.filepath
        img = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # 흐릿한 이미지 제거
        if variance_of_laplacian(img) < blur_threshold:
            continue

        keep_rows.append(row)

    pd.DataFrame(keep_rows).to_csv(MANIFEST_CLEAN, sep="\t", index=False)

    print(f"✅ Step3: blur 제거 후 남은 이미지: {len(keep_rows)}")
    print(f"📄 파일: {MANIFEST_CLEAN}")


# ============================================================
# 6) Main — 전체 파이프라인 실행
# ============================================================
if __name__ == "__main__":
    print("🔧 SkinAI 데이터 전처리 시작...")
    build_manifest_raw()
    normalize_images()
    clean_blurry_images()
    print("🎉 모든 단계 완료!")