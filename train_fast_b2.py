"""
train_fast_b2.py
- EfficientNet-B2 기반 고속 학습 (AMP + CosineAnnealing + EarlyStopping + tqdm)
- SkinAI 데이터셋 고성능 학습 스크립트
"""

import gc
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
from collections import Counter
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm

from config import PROJECT_ROOT, CLASSES
from dataset_split import SPLIT_ROOT


# ============================================================
# GPU 초기화 / 최적화
# ============================================================
gc.collect()
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True

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
# 증강 설정
# ============================================================
IMG_SIZE = 224
BATCH = 16
EPOCHS = 10
PATIENCE = 4
LR = 3e-4

train_tf = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 1.1)),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.85, 1.0)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.15, 0.15, 0.10),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.485, 0.456, 0.406),
        (0.229, 0.224, 0.225)
    ),
])

val_tf = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 1.1)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.485, 0.456, 0.406),
        (0.229, 0.224, 0.225)
    ),
])


# ============================================================
# Dataset / Dataloader
# ============================================================
train_ds = ImageFolderCustom(SPLIT_ROOT / "train", CLASSES, train_tf)
val_ds = ImageFolderCustom(SPLIT_ROOT / "val", CLASSES, val_tf)

train_dl = DataLoader(
    train_ds,
    batch_size=BATCH,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_dl = DataLoader(
    val_ds,
    batch_size=BATCH,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

print(f"Train size: {len(train_ds)} | Val size: {len(val_ds)}")


# ============================================================
# 클래스 가중치 + label smoothing
# ============================================================
cnt = Counter(train_ds.targets)
class_counts = np.array([cnt[i] for i in range(len(CLASSES))], dtype=np.float32)
weights = class_counts.sum() / np.clip(class_counts, 1, None)
weights = torch.tensor(weights / weights.mean(), dtype=torch.float32, device=device)

criterion = nn.CrossEntropyLoss(
    weight=weights,
    label_smoothing=0.05
)


# ============================================================
# EfficientNet-B2 모델
# ============================================================
model = timm.create_model(
    "tf_efficientnet_b2_ns",
    pretrained=True,
    num_classes=len(CLASSES)
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS,
    eta_min=1e-5
)

# AMP scaler
scaler = torch.amp.GradScaler(device.type)


# ============================================================
# 평가 함수
# ============================================================
def evaluate():
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0

    with torch.no_grad(), torch.amp.autocast(device.type):
        for x, y in tqdm(val_dl, desc="검증", leave=False):
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)
            loss_sum += loss.item() * y.size(0)

            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    return loss_sum / total, correct / total


# ============================================================
# 학습 루프 (EarlyStopping 포함)
# ============================================================
BEST_DIR = PROJECT_ROOT / "models"
BEST_DIR.mkdir(exist_ok=True)
best_path = BEST_DIR / "effb2_skin_fast.pth"

best_acc = -1.0
no_improve = 0

for epoch in range(1, EPOCHS + 1):
    model.train()
    pbar = tqdm(train_dl, desc=f"학습 {epoch}/{EPOCHS}")

    for x, y in pbar:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device.type):
            logits = model(x)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    scheduler.step()
    val_loss, val_acc = evaluate()

    print(f"Epoch {epoch:02d} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | best={best_acc:.4f}")

    # Best model 저장
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), best_path)
        no_improve = 0
        print("✨ 개선된 모델 저장!")
    else:
        no_improve += 1

    # Early Stopping
    if no_improve >= PATIENCE:
        print("⏹ Early Stopping 발동!")
        break

print("📌 최종 최고 모델 저장:", best_path)