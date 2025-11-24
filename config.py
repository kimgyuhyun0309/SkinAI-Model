from pathlib import Path

# 프로젝트 루트 경로 (이 파일이 위치한 폴더를 기준)
PROJECT_ROOT = Path(__file__).resolve().parent

# 피부 클래스 목록
CLASSES = [
    "acne_mild",
    "acne_moderate",
    "acne_severe",
    "dry_skin",
    "pigmentation",
    "redness",
    "blackheads",
    "pores",
    "clear_skin",
]