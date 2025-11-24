"""
evaluation.py
- Validation dataset에 대해 분류 성능을 평가하는 스크립트
- Classification Report + Confusion Matrix 출력
"""

import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from config import PROJECT_ROOT, CLASSES
from dataset_split import SPLIT_ROOT


# ============================================================
# Device 설정
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥 Device: {device}")


# ============================================================
# Dataset 정의
# ============================================================
class ImageFolderCustom(Dataset):
    def __init__(self, root, classes, tf):
        self.paths = []
        self.targets = []
        self.tf = tf

        for i, c in enumerate(classes):
            for p in (root / c).glob("*.jpg"):
                self.paths.append(p)
                self.targets.append(i)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.tf(img), self.targets[idx]


# ============================================================
# 검증용 Transform
# ============================================================
val_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.485, 0.456, 0.406),
        (0.229, 0.224, 0.225)
    )
])


# ============================================================
# Dataset / DataLoader 로드
# ============================================================
val_ds = ImageFolderCustom(SPLIT_ROOT / "val", CLASSES, val_tf)
val_dl = DataLoader(
    val_ds,
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)


# ============================================================
# 모델 로드
# ============================================================
best_model_path = PROJECT_ROOT / "models" / "effb2_skin_fast.pth"

import timm
model = timm.create_model(
    "tf_efficientnet_b2_ns",
    pretrained=False,
    num_classes=len(CLASSES)
).to(device)

state = torch.load(best_model_path, map_location=device)
model.load_state_dict(state)
model.eval()

print(f"📌 모델 로드 완료: {best_model_path}")


# ============================================================
# 평가 수행
# ============================================================
y_true = []
y_pred = []

with torch.no_grad(), torch.amp.autocast(device.type):
    for x, y in val_dl:
        x = x.to(device)

        logits = model(x)
        pred = logits.argmax(1).cpu().tolist()

        y_pred += pred
        y_true += y.tolist()


# ============================================================
# Classification Report 출력
# ============================================================
print("\n📊 Classification Report")
print(classification_report(y_true, y_pred, target_names=CLASSES, digits=3))


# ============================================================
# Confusion Matrix 시각화
# ============================================================
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASSES,
    yticklabels=CLASSES,
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix — Validation Set")
plt.tight_layout()
plt.show()