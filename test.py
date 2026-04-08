import os
import re
import random
import shutil
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from diffusers import UNet2DModel
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# ✨ 引入学术界最权威的官方 pytorch-fid 库
from pytorch_fid.fid_score import calculate_fid_given_paths

# ==========================================
# 0. 全局路径与超参数配置
# ==========================================
# 数据集路径
SIM_DIR = "./data/sim"
REAL_DIR = "./data/real"

# 模型权重路径 (填入你跑出的 best 模型)
WEIGHT_PATH = "saved_models/K_3/unet_epoch_3000.pth"

# 生成结果的保存目录
OUTPUT_DIR = "./final_test_results3"

# 生成步数 (Flow Matching 的欧拉积分步数，建议 50~100 步)
NUM_INFERENCE_STEPS = 50

# 批量处理大小 (根据你的显存大小调节，24G显存设为 32 或 64 毫无压力)
BATCH_SIZE = 32

# 图片保存比例 (0.05 即随机保存 5% 的图片作为视觉抽查)
SAVE_RATIO = 1


# ==========================================
# 1. 网络结构定义 (✨ 已同步为 GFC-Net 优化后的全局注意力架构)
# ==========================================
def get_pure_unet(img_size=64):
    unet = UNet2DModel(
        sample_size=img_size,
        in_channels=2,
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(64, 128, 256, 512),
        down_block_types=(
            "AttnDownBlock2D",  # ✨ 这里已同步更新为带有全局注意力的模块
            "AttnDownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "AttnUpBlock2D",  # ✨ 这里已同步更新
        ),
        dropout=0.35  # ✨ 保持与训练时一致 (评估模式下自动失效, 不影响结果)
    )
    return unet


# ==========================================
# 2. 自动扫描并提取测试集 (仅保留 17 度数据)
# ==========================================
def get_test_pairs(sim_dir, real_dir):
    test_pairs = []
    print("[*] 正在扫描数据集，自动提取 17° 测试集...")

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
                        # 严格提取 17° 作为测试集
                        if abs(el_deg - 17.0) < 0.5:
                            test_pairs.append({
                                'sim_path': sim_path,
                                'real_path': real_path,
                                'filename': f"{class_name}_{img_name}"
                            })

            if is_root_dir:
                break

    print(f"[*] 测试集提取完成: 共 {len(test_pairs)} 张 17° SAR 图像。")
    return test_pairs


# ==========================================
# 3. 核心生成模块 (✨ Flow Matching 极速欧拉采样)
# ==========================================
@torch.no_grad()
def generate_test_images(test_pairs, device):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 初始化模型并加载最佳权重
    unet = get_pure_unet(img_size=64)
    if not os.path.exists(WEIGHT_PATH):
        raise FileNotFoundError(f"❌ 找不到权重文件 {WEIGHT_PATH}，请检查路径！")

    print(f"\n📦 正在加载最佳泛化模型权重: {WEIGHT_PATH}")
    unet.load_state_dict(torch.load(WEIGHT_PATH, map_location=device))
    unet.to(device)
    unet.eval()

    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Grayscale(1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    print(f"🚀 开始在测试集上批量生成伪真实 SAR 图像 (Batch Size: {BATCH_SIZE}, Euler 步数: {NUM_INFERENCE_STEPS})...")
    print(f"📸 抽查保存的图片将自动拼接为: [仿真骨架(左) | 模型生成(中) | 真实雷达(右)]")

    eval_data = []

    # 设定欧拉积分的时间步长 dt
    dt = 1.0 / NUM_INFERENCE_STEPS

    # 批量循环
    for i in tqdm(range(0, len(test_pairs), BATCH_SIZE), desc="Generating Batches"):
        batch_pairs = test_pairs[i:i + BATCH_SIZE]
        current_bsz = len(batch_pairs)

        # 准备批量条件输入
        cond_list = [transform(Image.open(item['sim_path'])) for item in batch_pairs]
        condition_sim = torch.stack(cond_list).to(device)

        # ✨ Flow Matching 初始状态: 纯噪声 z_0
        z_t = torch.randn((current_bsz, 1, 64, 64), device=device)

        # ✨ Flow Matching 欧拉积分求解 (Euler ODE Solver)
        for step in range(NUM_INFERENCE_STEPS):
            # 当前时间 t ∈ [0, 1)
            t_val = step * dt
            # 扩展 time 为 UNet 支持的维度，映射到 [0, 1000]
            t_tensor = torch.full((current_bsz,), t_val * 1000.0, device=device)

            with torch.amp.autocast('cuda'):
                unet_input = torch.cat([z_t, condition_sim], dim=1)
                # 预测速度场 v
                pred_v = unet(unet_input, t_tensor).sample

            # z_{t + dt} = z_t + v * dt
            z_t = z_t + pred_v * dt

        # 采样结束后，z_t 即为生成的图像 z_1
        image = (z_t / 2 + 0.5).clamp(0, 1)
        image_np = (image.cpu().numpy() * 255.0).astype(np.uint8)

        # 提取生成的图像数据并进行拼接保存
        for j, item in enumerate(batch_pairs):
            img_array = image_np[j, 0]  # 取出当前的单通道 64x64 矩阵

            # 仅随机保存指定比例的图片到硬盘，用于论文可视化检查
            if random.random() < SAVE_RATIO:
                save_path = os.path.join(OUTPUT_DIR, item['filename'])

                # 读取对应的仿真(Sim)和真实(Real)图像，确保尺寸一致
                sim_img_pil = Image.open(item['sim_path']).convert('L').resize((64, 64))
                real_img_pil = Image.open(item['real_path']).convert('L').resize((64, 64))
                fake_img_pil = Image.fromarray(img_array)

                # 创建一块 192x64 的画布 (宽 64*3，高 64)
                # 布局: [左: Sim(输入)] | [中: Fake(生成)] | [右: Real(真实)]
                stitched_img = Image.new('L', (192, 64))
                stitched_img.paste(sim_img_pil, (0, 0))
                stitched_img.paste(fake_img_pil, (64, 0))
                stitched_img.paste(real_img_pil, (128, 0))

                stitched_img.save(save_path)

            # 将生成的 numpy 数组直接存入内存，供后续计算指标使用
            eval_data.append({
                'filename': item['filename'],
                'fake_img_np': img_array,
                'real_path': item['real_path']
            })

    return eval_data


# ==========================================
# 4. 指标评估模块 ✨ 接入官方 pytorch-fid 库
# ==========================================
@torch.no_grad()
def evaluate_metrics(eval_data, device):
    print("\n📈 开始计算图像质量评价指标 (SSIM, PSNR, ✨官方 pytorch-fid)...")

    # 建立临时文件夹用于 pytorch-fid 官方包的文件夹读取机制
    temp_dir = "./fid_temp"
    fid_real_dir = os.path.join(temp_dir, "real")
    fid_fake_dir = os.path.join(temp_dir, "fake")
    os.makedirs(fid_real_dir, exist_ok=True)
    os.makedirs(fid_fake_dir, exist_ok=True)

    results = []

    # 循环内存中的数据，计算 SSIM/PSNR 并将图片保存到临时文件夹
    for item in tqdm(eval_data, desc="Preparing Data & Calculating SSIM/PSNR"):
        img_fake_np = item['fake_img_np']

        img_real_pil = Image.open(item['real_path']).convert('L').resize((64, 64))
        img_real_np = np.array(img_real_pil)

        # SSIM 和 PSNR 依然使用原始的 64x64 图像进行严谨计算
        s_val = ssim(img_real_np, img_fake_np, data_range=255)
        p_val = psnr(img_real_np, img_fake_np, data_range=255)

        results.append({
            "Filename": item['filename'],
            "SSIM": s_val,
            "PSNR": p_val
        })

        # ✨ 将单通道转为 RGB 并保存到临时文件夹中供 pytorch-fid 读取
        img_fake_pil_rgb = Image.fromarray(img_fake_np).convert('RGB')
        img_real_pil_rgb = img_real_pil.convert('RGB')

        img_fake_pil_rgb.save(os.path.join(fid_fake_dir, item['filename']))
        img_real_pil_rgb.save(os.path.join(fid_real_dir, item['filename']))

    print("📊 正在调用 pytorch-fid 官方接口计算最终特征距离 (这需要几秒钟)...")
    # dims=2048 是标准 InceptionV3 输出层
    fd_score = calculate_fid_given_paths([fid_real_dir, fid_fake_dir], batch_size=BATCH_SIZE, device=device, dims=2048)

    # ✨ 清理临时文件夹，保持系统整洁
    try:
        shutil.rmtree(temp_dir)
    except Exception as e:
        print(f"⚠️ 临时文件夹 {temp_dir} 删除失败: {e}")

    # 汇总与输出
    df = pd.DataFrame(results)
    avg_ssim = df["SSIM"].mean()
    avg_psnr = df["PSNR"].mean()

    print("\n" + "★" * 40)
    print(f"📊 17° 测试集最终评估报告 (共 {len(eval_data)} 张):")
    print(f"✅ 平均 SSIM (结构相似度): {avg_ssim:.4f}  (越高越好)")
    print(f"✅ 平均 PSNR (峰值信噪比): {avg_psnr:.4f} dB  (越高越好)")
    print(f"✅ 标准官方 FID (pytorch-fid): {fd_score:.4f}  (越低越好)")
    print("★" * 40)

    csv_path = "final_evaluation_report.csv"
    df.to_csv(csv_path, index=False)
    print(f"📝 每一张图片的详细得分已保存至: {csv_path}")
    print(f"🖼️ 为了防止硬盘冗余，系统仅抽取了 5% 的样本图保存在: {OUTPUT_DIR}")


# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 提取测试对
    test_pairs = get_test_pairs(sim_dir=SIM_DIR, real_dir=REAL_DIR)

    if len(test_pairs) == 0:
        print("❌ 未找到任何 17° 的测试数据，请检查你的数据集路径！")
    else:
        # 2. 批量生成 (将图像阵列保存在内存)
        eval_data = generate_test_images(test_pairs, device)

        # 3. 批量计算指标
        evaluate_metrics(eval_data, device)