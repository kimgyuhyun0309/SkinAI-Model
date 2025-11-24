"""
dataset_split.py
- SkinAI 데이터셋을 train / val / test로 계층적(Stratified) 분할하는 스크립트
- manifest_clean.tsv(최종 정제된 이미지 목록)를 기반으로 분할
- rare class(사진 적은 클래스)는 모두 train으로 이동
"""

import shutil
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from config import PROJECT_ROOT, CLASSES


# ============================================================
# 경로 설정
# ============================================================
CLEAN_MANIFEST = PROJECT_ROOT / "manifest_clean.tsv"
SPLIT_ROOT = PROJECT_ROOT / "splits"

# train/val/test 폴더 생성
for split in ["train", "val", "test"]:
    for c in CLASSES:
        (SPLIT_ROOT / split / c).mkdir(parents=True, exist_ok=True)


# ============================================================
# 데이터 로드
# ============================================================
def load_manifest():
    df = pd.read_csv(CLEAN_MANIFEST, sep="\t")
    df = df[df["class"].isin(CLASSES)].reset_index(drop=True)
    return df


# ============================================================
# 파일 복사 함수
# ============================================================
def copy_rows(rows: pd.DataFrame, split: str):
    """각 split/train|val|test 폴더로 이미지 복사"""
    for _, r in rows.iterrows():
        src = Path(r.filepath)
        dst = SPLIT_ROOT / split / r["class"] / src.name
        if not dst.exists():
            shutil.copy2(src, dst)


# ============================================================
# 메인 분할 함수
# ============================================================
def split_dataset(
    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15,
    min_samples_for_split=6
):
    """
    계층적 분할:
    - 이미지가 많은 클래스 → Stratified split
    - 너무 적은 클래스(<6장) → 모두 train으로 이동
    """

    df = load_manifest()
    print(f"📂 전체 이미지 수: {len(df)}")

    # 클래스별 개수
    counts = df["class"].value_counts()
    print("\n클래스별 이미지 수:")
    print(counts)

    # 희귀 클래스 분리
    rare_classes = set(counts[counts < min_samples_for_split].index)
    print(f"\n⚠ 희귀 클래스(모두 train으로 이동): {list(rare_classes)}")

    df_main = df[~df["class"].isin(rare_classes)].reset_index(drop=True)
    df_rare = df[df["class"].isin(rare_classes)].reset_index(drop=True)

    # ============================
    # 1차 Split: train vs temp(val+test)
    # ============================
    sss1 = StratifiedShuffleSplit(
        n_splits=1,
        test_size=(val_ratio + test_ratio),
        random_state=42
    )
    (train_idx, temp_idx), = sss1.split(df_main, df_main["class"])

    train_df = df_main.iloc[train_idx].copy()
    temp_df = df_main.iloc[temp_idx].copy()

    # ============================
    # 2차 Split: val vs test
    # ============================
    sss2 = StratifiedShuffleSplit(
        n_splits=1,
        test_size=test_ratio / (val_ratio + test_ratio),
        random_state=42
    )
    (val_idx, test_idx), = sss2.split(temp_df, temp_df["class"])

    val_df = temp_df.iloc[val_idx].copy()
    test_df = temp_df.iloc[test_idx].copy()

    # ============================
    # 희귀 클래스는 모두 train에 추가
    # ============================
    if len(df_rare) > 0:
        train_df = pd.concat([train_df, df_rare], ignore_index=True)

    # ============================
    # 파일 복사
    # ============================
    print("\n📁 파일 복사 중...")

    copy_rows(train_df, "train")
    copy_rows(val_df, "val")
    copy_rows(test_df, "test")

    # ============================
    # 결과 출력
    # ============================
    def count_split(split):
        return {
            c: len(list((SPLIT_ROOT / split / c).glob("*.jpg")))
            for c in CLASSES
        }

    train_count = count_split("train")
    val_count = count_split("val")
    test_count = count_split("test")

    print("\n🎉 분할 완료!")
    print(f"Train 총합: {sum(train_count.values())}")
    print(f"Val   총합: {sum(val_count.values())}")
    print(f"Test  총합: {sum(test_count.values())}")

    print("\n클래스별 데이터 분포 (train / val / test):")
    for c in CLASSES:
        print(f"{c:14s}: {train_count[c]:4d} / {val_count[c]:4d} / {test_count[c]:4d}")


# ============================================================
# 실행부
# ============================================================
if __name__ == "__main__":
    print("🔧 Stratified Dataset Split 시작...")
    split_dataset()