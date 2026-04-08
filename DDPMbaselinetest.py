import os
import re
import random
import shutil
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from diffusers import UNet2DModel, DDIMScheduler
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

# 基础输出路径 (与训练脚本保持完全同步)
BASE_OUTPUT_DIR = "./output"

# 生成步数 (DDIM 加速采样，建议 50 步即可达到很好效果)
NUM_INFERENCE_STEPS = 50

# 批量处理大小 (24G显存设为 32 或 64 毫无压力)
BATCH_SIZE = 32

# 图片保存比例 (0.05 即随机保存 5% 的图片作为视觉抽查)
SAVE_RATIO = 0.05


# ==========================================
# 1. 网络结构定义 (✨ 严格对齐训练时的结构)
# ==========================================
def get_model(img_size=64):
    unet = UNet2DModel(
        sample_size=img_size,
        in_channels=2,
        out_channels=1,
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
        dropout=0.5  # ✨ 必须与训练时保持绝对一致，否则加载权重会报错
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
# 3. 核心生成模块 (批量推理极速版 + 智能拼接保存)
# ==========================================
@torch.no_grad()
def generate_test_images(test_pairs, weight_path, output_dir, device):
    os.makedirs(output_dir, exist_ok=True)

    # 初始化模型并加载最佳权重
    unet = get_model(img_size=64)
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"❌ 找不到权重文件 {weight_path}，请检查路径！")

    print(f"\n📦 正在加载模型权重: {weight_path}")
    unet.load_state_dict(torch.load(weight_path, map_location=device))
    unet.to(device)
    unet.eval()

    # ✨ 修复：使用与训练代码一致的 linear 调度表，确保去噪物理过程吻合
    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_schedule="linear",
        prediction_type="epsilon"
    )
    scheduler.set_timesteps(NUM_INFERENCE_STEPS)

    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((64, 64), Image.Resampling.BILINEAR),
        transforms.Grayscale(1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    print(f"🚀 开始批量生成伪真实 SAR 图像 (Batch Size: {BATCH_SIZE}, 步数: {NUM_INFERENCE_STEPS})...")
    print(f"📸 抽查保存的图片将自动拼接为: [仿真骨架(左) | 模型生成(中) | 真实雷达(右)]")

    eval_data = []

    # 批量循环
    for i in tqdm(range(0, len(test_pairs), BATCH_SIZE), desc="Generating Batches"):
        batch_pairs = test_pairs[i:i + BATCH_SIZE]
        current_bsz = len(batch_pairs)

        # 准备批量条件输入
        cond_list = [transform(Image.open(item['sim_path'])) for item in batch_pairs]
        condition_sim = torch.stack(cond_list).to(device)

        # 准备批量初始纯噪声
        z_t = torch.randn((current_bsz, 1, 64, 64), device=device)

        # 扩散去噪过程 (直接在 GPU 上批量运算)
        for t in scheduler.timesteps:
            with torch.amp.autocast('cuda'):
                unet_input = torch.cat([z_t, condition_sim], dim=1)
                noise_pred = unet(unet_input, t).sample
            # 使用 DDIM 进行加速步进
            z_t = scheduler.step(noise_pred, t, z_t).prev_sample

        # 后处理
        image = (z_t / 2 + 0.5).clamp(0, 1)
        image_np = (image.cpu().numpy() * 255.0).astype(np.uint8)

        # 提取生成的图像数据并进行拼接保存
        for j, item in enumerate(batch_pairs):
            img_array = image_np[j, 0]

            # 仅随机保存指定比例的图片到硬盘，用于论文可视化检查
            if random.random() < SAVE_RATIO:
                save_path = os.path.join(output_dir, item['filename'])

                # 读取对应的仿真(Sim)和真实(Real)图像，确保尺寸一致
                sim_img_pil = Image.open(item['sim_path']).convert('L').resize((64, 64))
                real_img_pil = Image.open(item['real_path']).convert('L').resize((64, 64))
                fake_img_pil = Image.fromarray(img_array)

                # 创建一块 192x64 的画布
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
def evaluate_metrics(eval_data, output_dir, device):
    print("\n📈 开始计算图像质量评价指标 (SSIM, PSNR, ✨官方 pytorch-fid)...")

    # 建立临时文件夹用于 pytorch-fid 官方包的文件夹读取机制 (放置在 output 目录下)
    temp_dir = os.path.join(BASE_OUTPUT_DIR, "fid_temp")
    fid_real_dir = os.path.join(temp_dir, "real")
    fid_fake_dir = os.path.join(temp_dir, "fake")

    # 确保清理之前的旧缓存
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(fid_real_dir, exist_ok=True)
    os.makedirs(fid_fake_dir, exist_ok=True)

    results = []

    # 循环内存中的数据，计算 SSIM/PSNR 并将图片保存到临时文件夹
    for item in tqdm(eval_data, desc="Preparing Data & Calculating SSIM/PSNR"):
        img_fake_np = item['fake_img_np']

        img_real_pil = Image.open(item['real_path']).convert('L').resize((64, 64))
        img_real_np = np.array(img_real_pil)

        # SSIM 和 PSNR 使用原始的 64x64 图像进行严谨计算
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
    print(f"📊 测试集最终评估报告 (共 {len(eval_data)} 张):")
    print(f"✅ 平均 SSIM (结构相似度): {avg_ssim:.4f}  (越高越好)")
    print(f"✅ 平均 PSNR (峰值信噪比): {avg_psnr:.4f} dB  (越高越好)")
    print(f"✅ 标准官方 FID (pytorch-fid): {fd_score:.4f}  (越低越好)")
    print("★" * 40)

    # 报告保存到对应的流水线文件夹中
    csv_path = os.path.join(output_dir, "evaluation_report.csv")
    df.to_csv(csv_path, index=False)
    print(f"📝 每一张图片的详细得分已保存至: {csv_path}")
    print(f"🖼️ 为了防止硬盘冗余，系统抽取的样本图保存在: {output_dir}")


# ==========================================
# 5. 流水线任务中心 (✨ 自动寻找各K值下最后一轮的权重)
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 提取测试对
    test_pairs = get_test_pairs(sim_dir=SIM_DIR, real_dir=REAL_DIR)

    if len(test_pairs) == 0:
        print("❌ 未找到任何 17° 的测试数据，请检查你的数据集路径！")
    else:
        # 定义需要进行测试的实验组 (支持多组流水线全自动测试)
        # 例如: [1, 3, 5, 10, 16, None] 表示分别测试 K=1, 3, 5, 10, 16 以及全量数据(None)
        experiments = [1,4,  16, None]

        for k in experiments:
            exp_name = f"K_{k}" if k is not None else "K_All"
            print("\n" + "=" * 60)
            print(f"🚀 开始执行指标评估任务: {exp_name}")
            print("=" * 60)

            model_dir = os.path.join(BASE_OUTPUT_DIR, "saved_models", exp_name)

            if not os.path.exists(model_dir):
                print(f"⏭️ 找不到 {exp_name} 的模型文件夹 ({model_dir})，跳过该实验...")
                continue

            # ✨ 自动扫描该文件夹下所有的 unet_epoch_*.pth，找出轮数最大的文件
            latest_epoch = -1
            target_weight_path = ""
            for filename in os.listdir(model_dir):
                match = re.search(r'unet_epoch_(\d+)\.pth', filename)
                if match:
                    epoch_num = int(match.group(1))
                    if epoch_num > latest_epoch:
                        latest_epoch = epoch_num
                        target_weight_path = os.path.join(model_dir, filename)

            if latest_epoch == -1 or not target_weight_path:
                print(f"⏭️ 在 {model_dir} 下找不到任何 unet_epoch_*.pth 文件，跳过该实验...")
                continue

            print(
                f"🎯 自动定位到 {exp_name} 的最后一轮权重: 第 {latest_epoch} 轮 ({os.path.basename(target_weight_path)})")

            # 将测试结果按照 epoch 分类存放，避免被其他测试覆盖
            target_output_dir = os.path.join(BASE_OUTPUT_DIR, "test_evaluation", exp_name, f"epoch_{latest_epoch}")

            # 2. 批量生成 (将图像阵列保存在内存)
            eval_data = generate_test_images(test_pairs, target_weight_path, target_output_dir, device)

            # 3. 批量计算指标
            evaluate_metrics(eval_data, target_output_dir, device)

        print("\n🎉🎉 所有评估任务已执行完毕！")