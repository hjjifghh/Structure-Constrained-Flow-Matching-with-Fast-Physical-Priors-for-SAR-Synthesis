import os
import random
import re
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from diffusers import UNet2DModel, DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from tqdm import tqdm
from torch.amp import autocast, GradScaler


# ==========================================
# ⚡️ EMA (指数移动平均) 权重平滑器，用于防止过拟合
# ==========================================
class EMAModel:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def step(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] -= (1.0 - self.decay) * (self.shadow[name] - param.data)

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]


# ==========================================
# 0. 数据扫描与 OOD 划分
# ==========================================
def get_train_val_pairs(sim_dir, real_dir, k_shot_per_class=None):
    val_pairs = []
    train_class_dict = {}

    print(f"[*] 正在扫描数据集，执行划分: 非17°为训练集，17°为测试集...")
    if k_shot_per_class is not None:
        print(f"[*] 🚀 启用 Few-Shot 采样: K={k_shot_per_class}")

    if os.path.exists(sim_dir):
        for class_folder in sorted(os.listdir(sim_dir)):
            sim_class_dir = os.path.join(sim_dir, class_folder)

            if not os.path.isdir(sim_class_dir):
                sim_class_dir = sim_dir
                real_class_dir = real_dir
                class_name = "default_class"
                is_root_dir = True
            else:
                real_class_dir = os.path.join(real_dir, class_folder, "crop_original")
                class_name = class_folder
                is_root_dir = False

            if class_name not in train_class_dict:
                train_class_dict[class_name] = []

            if not os.path.exists(real_class_dir):
                continue

            for img_name in sorted(os.listdir(sim_class_dir)):
                if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    continue

                sim_path = os.path.join(sim_class_dir, img_name)
                real_path = os.path.join(real_class_dir, img_name)

                if os.path.exists(real_path):
                    el_match = re.search(r'el([\d\.]+)', img_name, re.IGNORECASE)

                    if el_match:
                        el_deg = float(el_match.group(1))
                        if abs(el_deg - 17.0) < 0.5:
                            val_pairs.append((sim_path, real_path))
                        else:
                            train_class_dict[class_name].append((sim_path, real_path))

            if is_root_dir:
                break

    train_pairs = []
    class_names = list(train_class_dict.keys())
    num_classes = len(class_names)

    for cls_idx, cls_name in enumerate(class_names):
        pairs = train_class_dict[cls_name]
        if len(pairs) == 0:
            continue

        if k_shot_per_class is None or len(pairs) <= k_shot_per_class:
            train_pairs.extend(pairs)
        else:
            interval = len(pairs) / k_shot_per_class
            shift = (cls_idx / num_classes) * interval
            indices = [int((shift + i * interval)) % len(pairs) for i in range(k_shot_per_class)]
            sampled_pairs = [pairs[i] for i in indices]
            train_pairs.extend(sampled_pairs)

    random.shuffle(train_pairs)

    print(f"[*] 数据集划分完成: 训练集 {len(train_pairs)} 对, 测试集(17°) {len(val_pairs)} 对。")
    return train_pairs, val_pairs


# ==========================================
# 1. 自定义配对数据集
# ==========================================
class PairedSARDataset(Dataset):
    def __init__(self, pairs, img_size=64, is_train=True):
        self.pairs = pairs
        self.is_train = is_train

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size), Image.Resampling.BILINEAR),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        sim_path, real_path = self.pairs[idx]

        sim_img = Image.open(sim_path).convert("L")
        real_img = Image.open(real_path).convert("L")

        if self.is_train and random.random() > 0.6:
            sim_img = sim_img.transpose(Image.FLIP_LEFT_RIGHT)
            real_img = real_img.transpose(Image.FLIP_LEFT_RIGHT)

        sim_tensor = self.transform(sim_img)
        real_tensor = self.transform(real_img)

        return {
            "sim": sim_tensor,
            "real": real_tensor
        }


# ==========================================
# 2. 模型与调度器初始化 (✨ 引入官方 DDPMScheduler)
# ==========================================
def get_model(img_size=64):
    unet = UNet2DModel(
        sample_size=img_size,
        in_channels=2,  # Condition (1) + Noisy Image (1) = 2
        out_channels=1,  # 预测单通道噪声
        layers_per_block=2,
        block_out_channels=(64, 128, 256, 512),
        down_block_types=(
            "AttnDownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "AttnUpBlock2D",
        ),
        dropout=0.5
    )

    # 标准 DDPM 调度器 (1000 步线性时间表)
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule="linear",
        prediction_type="epsilon"  # 预测噪声
    )

    return unet, noise_scheduler


# ==========================================
# 3. 图像生成评估逻辑 (✨ 标准 DDPM 采样)
# ==========================================
@torch.no_grad()
def evaluate_and_save_samples(unet, noise_scheduler, dataset, device, epoch, save_dir, prefix="val"):
    unet.eval()
    os.makedirs(save_dir, exist_ok=True)

    num_samples = min(4, len(dataset))
    if num_samples == 0:
        return

    samples = [dataset[i] for i in range(num_samples)]
    conditions = torch.stack([s["sim"] for s in samples]).to(device)
    reals = torch.stack([s["real"] for s in samples]).to(device)

    # 初始状态 z_T (纯噪声)
    z_t = torch.randn_like(conditions)

    # 设置推理步数 (为了效率，评估时设为 50 步)
    noise_scheduler.set_timesteps(50)

    for t in noise_scheduler.timesteps:
        with autocast('cuda'):
            # 将当前隐状态与条件图像拼接
            unet_input = torch.cat([z_t, conditions], dim=1)
            # 预测噪声
            noise_pred = unet(unet_input, t).sample

        # 根据调度器公式更新到前一个时间步 z_{t-1}
        z_t = noise_scheduler.step(noise_pred, t, z_t).prev_sample

    # z_0 即为最终生成的图像
    conditions_vis = (conditions / 2 + 0.5).clamp(0, 1)
    reals_vis = (reals / 2 + 0.5).clamp(0, 1)
    generated_vis = (z_t / 2 + 0.5).clamp(0, 1)

    grid_rows = []
    for i in range(num_samples):
        row = torch.cat([conditions_vis[i], generated_vis[i], reals_vis[i]], dim=2)
        grid_rows.append(row)

    final_grid = torch.cat(grid_rows, dim=1)
    save_path = os.path.join(save_dir, f"{prefix}_epoch_{epoch}.png")
    save_image(final_grid, save_path)
    unet.train()


# ==========================================
# 4. 训练逻辑 (✨ 标准 DDPM 前向与损失)
# ==========================================
def train(k_shot_per_class=None, exp_name="K_All"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_pairs, val_pairs = get_train_val_pairs(sim_dir="./data/sim", real_dir="./data/real",
                                                 k_shot_per_class=k_shot_per_class)

    if len(train_pairs) == 0:
        print(f"❌ 错误：{exp_name} 训练集为空！")
        return None

    train_dataset = PairedSARDataset(train_pairs, img_size=64, is_train=True)
    val_dataset = PairedSARDataset(val_pairs, img_size=64, is_train=False)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True,
                                  persistent_workers=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True,
                                persistent_workers=True)

    unet, noise_scheduler = get_model(img_size=64)
    unet.to(device)

    optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-4, weight_decay=1e-3)
    scaler = GradScaler('cuda')

    use_ema = (k_shot_per_class is None)
    if use_ema:
        ema = EMAModel(unet, decay=0.995)
        print("   🌟 检测到全量数据训练，已开启 EMA 权重平滑。")
    else:
        print(f"   ⚡ 检测到小样本 ({exp_name})，已关闭 EMA以加快初始拟合。")

    if k_shot_per_class is not None and k_shot_per_class <= 16:
        epochs = 3000
        print(f"   🔥 K={k_shot_per_class} (<=16)，设定训练轮数为 {epochs} 轮。")
    else:
        epochs = 1500
        print(f"   🔥 设定训练轮数为 {epochs} 轮。")

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=(len(train_dataloader) * epochs)
    )

    # ✨ 统一与规范化输出路径
    vis_save_dir = os.path.join(".", "output", "training_visualizations", exp_name)
    model_save_dir = os.path.join(".", "output", "saved_models", exp_name)
    os.makedirs(vis_save_dir, exist_ok=True)
    os.makedirs(model_save_dir, exist_ok=True)

    best_val_loss = float('inf')
    best_model_path = os.path.join(model_save_dir, "unet_best.pth")

    for epoch in range(1, epochs + 1):
        unet.train()
        progress_bar = tqdm(train_dataloader, desc=f"[{exp_name}] Epoch {epoch}/{epochs}")
        train_loss = 0.0

        for batch in progress_bar:
            clean_real = batch["real"].to(device)  # x_0
            condition_sim = batch["sim"].to(device)
            bsz = clean_real.shape[0]

            # 数据增强：对条件图加入微小噪声增强鲁棒性
            if k_shot_per_class is not None and k_shot_per_class <= 4:
                if torch.rand(1).item() < 0.7:
                    condition_sim = (condition_sim + torch.randn_like(condition_sim) * 0.15).clamp(-1, 1)
            else:
                if torch.rand(1).item() < 0.5:
                    condition_sim = (condition_sim + torch.randn_like(condition_sim) * 0.05).clamp(-1, 1)

            # ----------------------------------------------------
            # ✨ 标准 DDPM 核心前向构造
            # ----------------------------------------------------
            # 1. 采样真实的随机噪声
            noise = torch.randn_like(clean_real)

            # 2. 随机采样时间步 t ~ U[0, 1000)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device).long()

            # 3. 根据 DDPM 公式加噪，获取 x_t
            noisy_real = noise_scheduler.add_noise(clean_real, noise, timesteps)

            with autocast('cuda'):
                # U-Net 输入：拼接加噪图像和条件图像
                unet_input = torch.cat([noisy_real, condition_sim], dim=1)

                # 4. 预测原本加入的噪声 epsilon
                noise_pred = unet(unet_input, timesteps).sample

                # 5. 计算损失 (最标准的均方误差 MSE 损失)
                loss = F.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            lr_scheduler.step()

            if use_ema:
                ema.step()

            train_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss / len(train_dataloader)

        if use_ema:
            ema.apply_shadow()

        # ✨ 验证阶段：计算 Validation Loss
        unet.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_dataloader:
                clean_real = batch["real"].to(device)
                condition_sim = batch["sim"].to(device)
                bsz = clean_real.shape[0]

                noise = torch.randn_like(clean_real)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device).long()
                noisy_real = noise_scheduler.add_noise(clean_real, noise, timesteps)

                with autocast('cuda'):
                    unet_input = torch.cat([noisy_real, condition_sim], dim=1)
                    noise_pred = unet(unet_input, timesteps).sample
                    v_loss = F.mse_loss(noise_pred, noise)
                    val_loss += v_loss.item()

        avg_val_loss = val_loss / len(val_dataloader) if len(val_dataloader) > 0 else 0.0

        if epoch % 10 == 0 or epoch == 1:
            print(f"✅ {exp_name} Epoch {epoch} | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f}")

        if avg_val_loss < best_val_loss and avg_val_loss > 0:
            best_val_loss = avg_val_loss
            torch.save(unet.state_dict(), best_model_path)

        if epoch % 100 == 0:
            evaluate_and_save_samples(unet, noise_scheduler, train_dataset, device, epoch, save_dir=vis_save_dir,
                                      prefix="train")
            evaluate_and_save_samples(unet, noise_scheduler, val_dataset, device, epoch, save_dir=vis_save_dir,
                                      prefix="val")

        if epoch % 300 == 0:
            torch.save(unet.state_dict(), os.path.join(model_save_dir, f"unet_epoch_{epoch}.pth"))

        if use_ema:
            ema.restore()

    return best_model_path


# ==========================================
# 5. 批量推理函数 (✨ 标准 DDPM 采样)
# ==========================================
@torch.no_grad()
def batch_inference(model_path, output_dir, device="cuda", batch_size=32):
    _, val_pairs = get_train_val_pairs(sim_dir="./data/sim", real_dir="./data/real", k_shot_per_class=1)
    os.makedirs(output_dir, exist_ok=True)

    unet, noise_scheduler = get_model(img_size=64)
    unet.load_state_dict(torch.load(model_path, map_location=device))
    unet.to(device)
    unet.eval()

    transform = transforms.Compose([
        transforms.Resize((64, 64), Image.Resampling.BILINEAR),
        transforms.Grayscale(1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # 推理步数设置
    noise_scheduler.set_timesteps(50)

    for i in tqdm(range(0, len(val_pairs), batch_size), desc=f"Inferencing {os.path.basename(output_dir)}"):
        batch_pairs = val_pairs[i:i + batch_size]
        current_bsz = len(batch_pairs)

        cond_list = [transform(Image.open(item[0])) for item in batch_pairs]
        condition_sim = torch.stack(cond_list).to(device)

        # 初始纯噪声 z_T
        z_t = torch.randn((current_bsz, 1, 64, 64), device=device)

        # DDPM 迭代去噪
        for t in noise_scheduler.timesteps:
            with autocast('cuda'):
                unet_input = torch.cat([z_t, condition_sim], dim=1)
                noise_pred = unet(unet_input, t).sample

            # 通过调度器降噪一步
            z_t = noise_scheduler.step(noise_pred, t, z_t).prev_sample

        # 采样结束时，映射回 0~255 的像素值
        image = (z_t / 2 + 0.5).clamp(0, 1)
        image_np = (image.cpu().numpy() * 255.0).astype(np.uint8)

        for j, item in enumerate(batch_pairs):
            img_array = image_np[j, 0]
            sim_path = item[0]

            class_name = os.path.basename(os.path.dirname(sim_path))
            if class_name == "sim":
                class_name = "default"
            filename = f"{class_name}_{os.path.basename(sim_path)}"

            save_path = os.path.join(output_dir, filename)
            Image.fromarray(img_array).save(save_path)


# ==========================================
# 6. 流水线任务中心
# ==========================================
if __name__ == "__main__":
    experiments = [None]

    for k in experiments:
        exp_name = f"K_{k}" if k is not None else "K_All"
        print("\n" + "=" * 60)
        print(f"🚀 开始执行流水线任务: {exp_name}")
        print("=" * 60)

        best_model_path = train(k_shot_per_class=k, exp_name=exp_name)

        if best_model_path and os.path.exists(best_model_path):
            # ✨ 统一推理结果保存路径
            out_dir = os.path.join(".", "output", "inference_results", exp_name)
            print(f"\n🎬 训练完成！开始生成测试集，保存至 {out_dir} ...")
            batch_inference(best_model_path, out_dir)
        else:
            print(f"❌ {exp_name} 训练失败或未找到模型，跳过推理。")

    print("\n🎉🎉 所有流水线任务已全部执行完毕！")