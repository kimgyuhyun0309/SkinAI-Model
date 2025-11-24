"""
train_advanced.py
- 얼굴 크롭 + 강화된 증강 + 클래스 불균형 보정
- EfficientNet-B0 고급 학습 파이프라인
"""

import os
import numpy as np
import torch
import torch.nn as nn
import cv2
from pathlib import Path
from PIL import Image
from collections import Counter
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import timm

from config import PROJECT_ROOT, CLASSES
from dataset_split import SPLIT_ROOT


# ============================================================
# Device 설정
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥 Device: {device}")


# ============================================================
# 얼굴 검출 (Haar Cascade)
# ============================================================
haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(haar_path)


def crop_face_pil(im_pil: Image.Image, min_face_frac=0.25):
    """
    얼굴이 충분히 크게 있을 경우 얼굴 중심으로 크롭.
    실패 시 원본 이미지를 그대로 반환.
    """
    try:
        img = np.array(im_pil.convert("RGB"))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(64, 64)
        )

        if len(faces) == 0:
            return im_pil

        # 가장 큰 얼굴 선택
        x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
        H, W = gray.shape[:2]

        if w < W * min_face_frac or h < H * min_face_frac:
            return im_pil

        pad = int(0.15 * max(w, h))

        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(W, x + w + pad)
        y1 = min(H, y + h + pad)

        face = img[y0:y1, x0:x1]

        if face.size == 0:
            return im_pil

        return Image.fromarray(face)

    except Exception:
        return im_pil


# ============================================================
# 데이터 증강 정의
# ============================================================
train_tf = transforms.Compose([
    transforms.Resize(288),
    transforms.Lambda(crop_face_pil),  # 얼굴 크롭 적용
    transforms.Resize(256),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(20),
    transforms.ColorJitter(
        brightness=0.25,
        contrast=0.25,
        saturation=0.2,
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.485, 0.456, 0.406),
        (0.229, 0.224, 0.225)
    ),
])

val_tf = transforms.Compose([
    transforms.Resize(288),
    transforms.Lambda(crop_face_pil),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.485, 0.456, 0.406),
        (0.229, 0.224, 0.225),
    ),
])


# ============================================================
# Dataset
# ============================================================
class ImageFolderCustom(Dataset):
    def __init__(self, root: Path, classes, tf):
        self.paths = []
        self.targets = []
        self.classes = classes
        self.tf = tf

        for i, c in enumerate(classes):
            ps = sorted((root / c).glob("*.jpg"))
            self.paths += ps
            self.targets += [i] * len(ps)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.tf(img), self.targets[idx]


train_ds = ImageFolderCustom(SPLIT_ROOT / "train", CLASSES, train_tf)
val_ds = ImageFolderCustom(SPLIT_ROOT / "val", CLASSES, val_tf)


# ============================================================
# 클래스 불균형 보정 (WeightedRandomSampler)
# ============================================================
counts = Counter(train_ds.targets)
class_counts = np.array([counts[i] for i in range(len(CLASSES))], dtype=np.float32)
class_weight = class_counts.sum() / np.clip(class_counts, 1, None)

sample_weight = np.array([class_weight[t] for t in train_ds.targets], dtype=np.float32)

sampler = WeightedRandomSampler(
    weights=torch.tensor(sample_weight, dtype=torch.double),
    num_samples=len(sample_weight),
    replacement=True
)

train_dl = DataLoader(
    train_ds,
    batch_size=32,
    sampler=sampler,
    num_workers=2,
    pin_memory=True
)
val_dl = DataLoader(
    val_ds,
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

print("Train size:", len(train_ds))
print("Class distribution:", counts)


# ============================================================
# 모델 설정
# ============================================================
model = timm.create_model(
    "tf_efficientnet_b0_ns",
    pretrained=True,
    num_classes=len(CLASSES)
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2
)


# ============================================================
# 평가 함수
# ============================================================
def evaluate():
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0

    with torch.no_grad():
        for x, y in val_dl:
            x, y = x.to(device), y.to(device)

            logits = model(x)
            loss = criterion(logits, y)
            loss_sum += loss.item() * y.size(0)

            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    return loss_sum / total, correct / total


# ============================================================
# 학습 루프
# ============================================================
BEST_DIR = PROJECT_ROOT / "models"
BEST_DIR.mkdir(exist_ok=True)
best_path = BEST_DIR / "effb0_skin_best.pth"

best_acc = -1.0

for epoch in range(12):
    model.train()

    for x, y in train_dl:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

    val_loss, val_acc = evaluate()
    scheduler.step(val_acc)

    print(f"Epoch {epoch + 1:02d} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), best_path)
        print("✨ 개선된 모델 저장!")

print("📌 최종 최고 모델 저장:", best_path)