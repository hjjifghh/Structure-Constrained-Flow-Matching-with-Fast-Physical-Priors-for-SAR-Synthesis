import os
import re
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import pandas as pd
from tqdm import tqdm

# ==========================================
# 0. 全局路径与超参数配置
# ==========================================
ROOT_DIR = os.path.abspath(".")
REAL_DIR = os.path.join(ROOT_DIR, "data", "real")

# ✨ 定义你的 7 组对比训练集路径 (追加了 K=All)
TRAIN_EXPERIMENTS = {
    "1. Real (Baseline)": os.path.join(ROOT_DIR, "data", "real"),
    "2. Sim (Old 64x64)": os.path.join(ROOT_DIR, "data", "sim1_64x64"),
    "3. Sim (New Bone)": os.path.join(ROOT_DIR, "data", "sim"),
    "4. Ours (K=1, Ep3000)": os.path.join(ROOT_DIR, "final_test_results1"),
    "5. Ours (K=4, Ep3000)": os.path.join(ROOT_DIR, "final_test_results4"),
    "6. Ours (K=16, Ep2700)": os.path.join(ROOT_DIR, "final_test_results16"),
    "7. Ours (K=All)": os.path.join(ROOT_DIR, "final_test_resultsall") # ✨ 请确保该文件夹名与你本地一致
}

# 🚀 训练超参数
BATCH_SIZE = 32
EPOCHS = 400
LEARNING_RATE = 0.0005
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# ✨ 核心修复 1：全局随机种子锁死，确保 Baseline 实验绝对可复现
# ==========================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==========================================
# 1. 动态智能数据集构建
# ==========================================
class SAR_ATR_Dataset(Dataset):
    def __init__(self, data_dir, is_train=True, transform=None, class_to_idx=None):
        self.data_dir = data_dir
        self.transform = transform
        self.is_train = is_train
        self.samples = []

        if class_to_idx is None:
            raise ValueError("必须提供类别的映射字典 class_to_idx!")
        self.class_to_idx = class_to_idx

        self._scan_dataset()

    def _scan_dataset(self):
        for root, dirs, files in os.walk(self.data_dir):
            # 路线挟持：避开 scatter_points 等干扰文件夹
            if "crop_original" in dirs:
                dirs[:] = ["crop_original"]

            for f in files:
                if not f.lower().endswith(('.png', '.jpg')): continue

                class_name = f.split('_')[0].lower()
                if class_name not in self.class_to_idx: continue

                name_no_ext = os.path.splitext(f)[0]
                el = None

                match_new = re.search(r'El([\d\.]+)', name_no_ext, re.IGNORECASE)
                if match_new: el = float(match_new.group(1).rstrip('.'))

                if el is None:
                    match_old = re.search(r'elevDeg_(\d+)', name_no_ext, re.IGNORECASE)
                    if match_old: el = float(match_old.group(1))

                if el is None: continue

                full_path = os.path.join(root, f)

                try:
                    with Image.open(full_path) as test_img:
                        test_img.verify()
                except Exception:
                    continue

                is_17_deg = abs(el - 17.0) < 0.5

                if self.is_train and is_17_deg:
                    self.samples.append((full_path, self.class_to_idx[class_name]))
                elif not self.is_train and not is_17_deg:
                    self.samples.append((full_path, self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('L')

        # ✨ 核心修复 2：智能识别拼接长图并动态截取中央区域
        width, height = img.size
        if width == 192 and height == 64:
            # 左(64), 上(0), 右(128), 下(64)
            img = img.crop((64, 0, 128, 64))
        elif width == 128 and height == 64:
            img = img.crop((64, 0, 128, 64))

        if self.transform: img = self.transform(img)
        return img, label


# ==========================================
# 2. ResNet-18 单通道魔改版 (✨ 架构泛化性优化)
# ==========================================
def get_resnet18_gray(num_classes):
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # 加入 Dropout 层，防止网络对生成的伪影产生过拟合
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(num_ftrs, num_classes)
    )
    return model.to(DEVICE)


# ==========================================
# 3. 核心训练与评估流水线 (✨ 训练策略泛化性优化)
# ==========================================
def train_and_evaluate(train_dir, test_loader, num_classes, class_to_idx, exp_name):
    print(f"\n" + "=" * 50)
    print(f"▶ 正在运行实验: {exp_name}")
    print("=" * 50)

    # 增强策略：加入了轻微的随机旋转 (±10度)，提升跨视角识别的鲁棒性
    train_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    train_dataset = SAR_ATR_Dataset(train_dir, is_train=True, transform=train_transform, class_to_idx=class_to_idx)

    if len(train_dataset) == 0:
        print(f"❌ 警告: 训练集 {train_dir} 中未找到 17° 图片，跳过本实验！")
        return 0.0

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    print(f"📦 载入训练数据 (17°): {len(train_dataset)} 张")

    model = get_resnet18_gray(num_classes)
    criterion = nn.CrossEntropyLoss()

    # 优化器加入了 L2 正则化 (weight_decay) 压制过大的权重
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    # 加入余弦退火学习率，让模型在后期能够更稳定地收敛
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    model.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()  # 更新学习率

        if (epoch + 1) % 10 == 0:
            print(
                f"   [Epoch {epoch + 1}/{EPOCHS}] Loss: {running_loss / len(train_loader):.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100.0 * correct / total
    print(f"🎯 [{exp_name}] 最终非 17° 测试集准确率: {acc:.2f}%")
    return acc


# ==========================================
# 4. 主程序入口
# ==========================================
if __name__ == "__main__":
    # 开启绝对稳态模式
    set_seed(42)

    print("🔥 SAR ATR 下游分类测试基准启动 (泛化增强版) 🔥")
    print(f"🖥️  正在使用的计算设备: {DEVICE}")

    print("\n🔍 正在扫描分类目录...")
    classes = set()
    for f in os.listdir(REAL_DIR):
        if f.endswith(('.png', '.jpg')): classes.add(f.split('_')[0].lower())
    for d in os.listdir(REAL_DIR):
        if os.path.isdir(os.path.join(REAL_DIR, d)): classes.add(d.lower())

    classes = sorted(list(classes))
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
    num_classes = len(classes)
    print(f"🏷️ 识别到 {num_classes} 个类别: {classes}")

    test_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    test_dataset = SAR_ATR_Dataset(REAL_DIR, is_train=False, transform=test_transform, class_to_idx=class_to_idx)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    print(f"📦 锁定全局测试集 (非17°真实数据): 共 {len(test_dataset)} 张")

    results = []
    for exp_name, train_dir in TRAIN_EXPERIMENTS.items():
        if not os.path.exists(train_dir):
            print(f"\n⚠️ 找不到路径 {train_dir}，如果你还没准备好该数据请先创建。")
            continue

        acc = train_and_evaluate(train_dir, test_loader, num_classes, class_to_idx, exp_name)
        results.append({"Experiment": exp_name, "Accuracy (%)": acc})

    if results:
        df = pd.DataFrame(results)
        print("\n" + "★" * 50)
        print("🏆 最终分类准确率汇总表 (ResNet-18)")
        print("★" * 50)
        print(df.to_string(index=False))

        df.to_csv("final_classification_results.csv", index=False)
        print(f"\n📝 表格已自动保存至 final_classification_results.csv")