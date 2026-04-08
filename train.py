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
from diffusers import UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from pytorch_msssim import ssim  # ✨ 新增：引入 MSSSIM 库用于结构约束计算


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
# 2. 模型初始化 (✨ 移除旧 Scheduler，直接预测速度场)
# ==========================================
def get_model(img_size=64):
    unet = UNet2DModel(
        sample_size=img_size,
        in_channels=2,
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(64, 128, 256, 512),
        down_block_types=(
            "AttnDownBlock2D",  # 最高分辨率直接引入全局注意力感知背景与目标
            "AttnDownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "AttnUpBlock2D",  # 解码阶段对称恢复全局结构
        ),
        dropout=0.5
    )
    return unet


# ==========================================
# 3. 图像生成评估逻辑 (✨ 彻底替换为 Euler ODE 采样)
# ==========================================
@torch.no_grad()
def evaluate_and_save_samples(unet, dataset, device, epoch, save_dir, prefix="val"):
    unet.eval()
    os.makedirs(save_dir, exist_ok=True)

    num_samples = min(4, len(dataset))
    if num_samples == 0:
        return

    samples = [dataset[i] for i in range(num_samples)]
    conditions = torch.stack([s["sim"] for s in samples]).to(device)
    reals = torch.stack([s["real"] for s in samples]).to(device)

    # 初始状态 z_0 (纯噪声)
    z_t = torch.randn_like(conditions)

    # 欧拉法步进采样参数
    num_steps = 50
    dt = 1.0 / num_steps

    for step in range(num_steps):
        # 当前时间 t ∈ [0, 1)
        t_val = step * dt
        # 扩展 time 为 UNet 支持的维度，并映射到 [0, 1000] 以利用原有的 Time Embedding
        t_tensor = torch.full((z_t.shape[0],), t_val * 1000.0, device=device)

        with autocast('cuda'):
            unet_input = torch.cat([z_t, conditions], dim=1)
            # 预测速度场 v
            pred_v = unet(unet_input, t_tensor).sample

        # 欧拉积分步进: z_{t + dt} = z_t + v * dt
        z_t = z_t + pred_v * dt

    # z_1 即为最终生成的图像
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
# 4. 训练逻辑 (✨ Flow Matching 重构)
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

    unet = get_model(img_size=64)
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

    vis_save_dir = f"training_visualizations/{exp_name}"
    model_save_dir = f"saved_models/{exp_name}"
    os.makedirs(vis_save_dir, exist_ok=True)
    os.makedirs(model_save_dir, exist_ok=True)

    best_val_loss = float('inf')
    best_model_path = os.path.join(model_save_dir, "unet_best.pth")

    if k_shot_per_class is not None and k_shot_per_class <= 16:
        l1_switch_epoch = 200
        initial_l1_weight = 0.1
        final_l1_weight = 0.02
    else:
        l1_switch_epoch = 500
        initial_l1_weight = 0.3
        final_l1_weight = 0.06

    for epoch in range(1, epochs + 1):
        unet.train()
        progress_bar = tqdm(train_dataloader, desc=f"[{exp_name}] Epoch {epoch}/{epochs}")
        train_loss = 0.0

        current_l1_weight = initial_l1_weight if epoch <= l1_switch_epoch else final_l1_weight

        for batch in progress_bar:
            clean_real = batch["real"].to(device)  # 数据 x_1
            condition_sim = batch["sim"].to(device)
            bsz = clean_real.shape[0]

            if k_shot_per_class is not None and k_shot_per_class <= 4:
                if torch.rand(1).item() < 0.7:
                    cond_noise = torch.randn_like(condition_sim) * 0.15
                    condition_sim = (condition_sim + cond_noise).clamp(-1, 1)
            else:
                if torch.rand(1).item() < 0.5:
                    cond_noise = torch.randn_like(condition_sim) * 0.05
                    condition_sim = (condition_sim + cond_noise).clamp(-1, 1)

            # 构造基础噪声 x_0
            base_noise = torch.randn_like(clean_real)
            random_offset_scale = torch.empty(bsz, 1, 1, 1, device=device).uniform_(0.0, 0.15)
            offset_noise = torch.randn(bsz, clean_real.shape[1], 1, 1, device=device) * random_offset_scale
            x_0 = base_noise - offset_noise

            # ----------------------------------------------------
            # ✨ Flow Matching 核心前向构造
            # ----------------------------------------------------
            # 1. 随机采样时间 t ~ U[0, 1]
            t = torch.rand((bsz,), device=device)
            t_expand = t.view(-1, 1, 1, 1)

            # 2. 计算当前状态 z_t = t * x_1 + (1 - t) * x_0
            z_t = t_expand * clean_real + (1.0 - t_expand) * x_0

            # 3. 计算目标速度场 target_v = x_1 - x_0
            target_v = clean_real - x_0

            # 映射时间步以适配 UNet Embedding (放缩到 [0, 1000])
            timesteps = (t * 1000.0).long()

            with autocast('cuda'):
                unet_input = torch.cat([z_t, condition_sim], dim=1)
                # 预测速度场 v
                pred_v = unet(unet_input, timesteps).sample

                # 4. 损失计算
                loss_mse = F.mse_loss(pred_v, target_v)
                loss_l1 = F.l1_loss(pred_v, target_v)

                # 动态还原当前步预测的清晰图像 (用于 SSIM)
                # 推导: z_t = t * x_1 + (1-t) * x_0; v = x_1 - x_0 => x_1 = z_t + (1-t) * v
                pred_x1 = z_t + (1.0 - t_expand) * pred_v

                loss_ssim = 1 - ssim(pred_x1, clean_real, data_range=2.0, size_average=True)

                loss = loss_mse + current_l1_weight * loss_l1 + 0.1 * loss_ssim

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            lr_scheduler.step()

            if use_ema:
                ema.step()

            train_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}", "l1_w": current_l1_weight})

        avg_train_loss = train_loss / len(train_dataloader)

        if use_ema:
            ema.apply_shadow()

        # ✨ 验证阶段同步调整为 Flow Matching
        unet.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_dataloader:
                clean_real = batch["real"].to(device)
                condition_sim = batch["sim"].to(device)
                bsz = clean_real.shape[0]

                base_noise = torch.randn_like(clean_real)
                offset_noise = torch.randn(bsz, clean_real.shape[1], 1, 1, device=device) * 0.075
                x_0 = base_noise - offset_noise

                t = torch.rand((bsz,), device=device)
                t_expand = t.view(-1, 1, 1, 1)

                z_t = t_expand * clean_real + (1.0 - t_expand) * x_0
                target_v = clean_real - x_0
                timesteps = (t * 1000.0).long()

                with autocast('cuda'):
                    unet_input = torch.cat([z_t, condition_sim], dim=1)
                    pred_v = unet(unet_input, timesteps).sample

                    loss_mse = F.mse_loss(pred_v, target_v)
                    loss_l1 = F.l1_loss(pred_v, target_v)

                    pred_x1 = z_t + (1.0 - t_expand) * pred_v
                    loss_ssim = 1 - ssim(pred_x1, clean_real, data_range=2.0, size_average=True)

                    v_loss = loss_mse + current_l1_weight * loss_l1 + 0.1 * loss_ssim
                    val_loss += v_loss.item()

        avg_val_loss = val_loss / len(val_dataloader) if len(val_dataloader) > 0 else 0.0

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"✅ {exp_name} Epoch {epoch} | Train: {avg_train_loss:.5f} | Val: {avg_val_loss:.5f} | L1_W: {current_l1_weight}")

        if avg_val_loss < best_val_loss and avg_val_loss > 0:
            best_val_loss = avg_val_loss
            torch.save(unet.state_dict(), best_model_path)

        if epoch % 100 == 0:
            evaluate_and_save_samples(unet, train_dataset, device, epoch, save_dir=vis_save_dir, prefix="train")
            evaluate_and_save_samples(unet, val_dataset, device, epoch, save_dir=vis_save_dir, prefix="val")

        if epoch % 300 == 0:
            torch.save(unet.state_dict(), os.path.join(model_save_dir, f"unet_epoch_{epoch}.pth"))

        if use_ema:
            ema.restore()

    return best_model_path


# ==========================================
# 5. 批量推理函数 (✨ 彻底替换为 Euler ODE 采样)
# ==========================================
@torch.no_grad()
def batch_inference(model_path, output_dir, device="cuda", batch_size=32):
    _, val_pairs = get_train_val_pairs(sim_dir="./data/sim", real_dir="./data/real", k_shot_per_class=1)
    os.makedirs(output_dir, exist_ok=True)

    unet = get_model(img_size=64)
    unet.load_state_dict(torch.load(model_path, map_location=device))
    unet.to(device)
    unet.eval()

    transform = transforms.Compose([
        transforms.Resize((64, 64), Image.Resampling.BILINEAR),
        transforms.Grayscale(1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    num_steps = 50
    dt = 1.0 / num_steps

    for i in tqdm(range(0, len(val_pairs), batch_size), desc=f"Inferencing {os.path.basename(output_dir)}"):
        batch_pairs = val_pairs[i:i + batch_size]
        current_bsz = len(batch_pairs)

        cond_list = [transform(Image.open(item[0])) for item in batch_pairs]
        condition_sim = torch.stack(cond_list).to(device)

        # 初始噪声 z_0
        z_t = torch.randn((current_bsz, 1, 64, 64), device=device)

        # 欧拉步进积分
        for step in range(num_steps):
            t_val = step * dt
            t_tensor = torch.full((current_bsz,), t_val * 1000.0, device=device)

            with autocast('cuda'):
                pred_v = unet(torch.cat([z_t, condition_sim], dim=1), t_tensor).sample

            # z_{t + dt} = z_t + v * dt
            z_t = z_t + pred_v * dt

        # 采样结束时 z_1 即为生成的图像
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
    experiments = [3]

    for k in experiments:
        exp_name = f"K_{k}" if k is not None else "K_All"
        print("\n" + "=" * 60)
        print(f"🚀 开始执行流水线任务: {exp_name}")
        print("=" * 60)

        best_model_path = train(k_shot_per_class=k, exp_name=exp_name)

        if best_model_path and os.path.exists(best_model_path):
            out_dir = f"./inference_results/{exp_name}"
            print(f"\n🎬 训练完成！开始生成测试集，保存至 {out_dir} ...")
            batch_inference(best_model_path, out_dir)
        else:
            print(f"❌ {exp_name} 训练失败或未找到模型，跳过推理。")

    print("\n🎉🎉 所有流水线任务已全部执行完毕！")