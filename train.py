"""
train.py
- EfficientNet-B0 기반 기본 학습 스크립트
- dataset_split.py로 생성된 /splits/train, /splits/val 데이터를 사용
"""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm

from config import PROJECT_ROOT, CLASSES
from dataset_split import SPLIT_ROOT  # splits 폴더 경로 재사용


# ============================================================
# 1) Device 설정
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥 Device: {device}")


# ============================================================
# 2) Dataset 정의
# ============================================================
class ImageFolderCustom(Dataset):
    """
    /splits/train/{class}/*.jpg
    /splits/val/{class}/*.jpg
    구조를 읽어서 (이미지, 라벨) 형태로 반환하는 Dataset
    """

    def __init__(self, root: Path, classes, tf):
        self.paths = []
        self.targets = []
        self.classes = classes
        self.tf = tf

        for idx, c in enumerate(classes):
            class_dir = root / c
            if not class_dir.exists():
                continue
            for p in class_dir.glob("*.jpg"):
                self.paths.append(p)
                self.targets.append(idx)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        img = self.tf(img)
        label = self.targets[i]
        return img, label


# ============================================================
# 3) 데이터 전처리 / 증강 정의
# ============================================================
def get_transforms():
    train_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.1
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])

    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])

    return train_tf, val_tf


# ============================================================
# 4) 평가 함수
# ============================================================
def evaluate(model, val_dl, criterion):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0

    with torch.no_grad():
        for x, y in val_dl:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)
            loss_sum += loss.item() * y.size(0)

            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    avg_loss = loss_sum / total if total > 0 else 0.0
    acc = correct / total if total > 0 else 0.0

    return avg_loss, acc


# ============================================================
# 5) 학습 루프
# ============================================================
def train_model(
    model_name: str = "tf_efficientnet_b0_ns",
    batch_size: int = 32,
    lr: float = 3e-4,
    epochs: int = 5,
):

    train_tf, val_tf = get_transforms()

    # Dataset / DataLoader
    train_root = SPLIT_ROOT / "train"
    val_root = SPLIT_ROOT / "val"

    train_ds = ImageFolderCustom(train_root, CLASSES, train_tf)
    val_ds = ImageFolderCustom(val_root, CLASSES, val_tf)

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    print(f"📦 Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")
    print(f"📦 Train batches: {len(train_dl)} | Val batches: {len(val_dl)}")

    # 모델 생성
    model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=len(CLASSES),
    ).to(device)

    # 옵티마이저 / 손실 함수
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4,
    )
    criterion = nn.CrossEntropyLoss()

    # 학습
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for x, y in train_dl:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * y.size(0)

        avg_train_loss = running_loss / len(train_ds) if len(train_ds) > 0 else 0.0
        val_loss, val_acc = evaluate(model, val_dl, criterion)

        print(
            f"[Epoch {epoch}/{epochs}] "
            f"train_loss={avg_train_loss:.4f} "
            f"| val_loss={val_loss:.4f} "
            f"| val_acc={val_acc:.4f}"
        )

    # 모델 저장
    model_dir = PROJECT_ROOT / "models"
    model_dir.mkdir(exist_ok=True)

    save_path = model_dir / "effb0_skin.pth"
    torch.save(model.state_dict(), save_path)
    print(f"✅ 학습 완료, 모델 저장 경로: {save_path}")

    return model, save_path


# ============================================================
# 6) 실행부
# ============================================================
if __name__ == "__main__":
    train_model(
        model_name="tf_efficientnet_b0_ns",
        batch_size=32,
        lr=3e-4,
        epochs=5,
    )