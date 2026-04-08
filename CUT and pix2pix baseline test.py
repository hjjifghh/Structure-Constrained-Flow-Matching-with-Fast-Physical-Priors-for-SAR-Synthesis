import os

# 🔥 核心修复 1：防闪退环境变量
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import shutil
import re
import subprocess
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# ✨ 深度学习评估库
try:
    import torch
    from torchvision import transforms
    from skimage.metrics import structural_similarity as calculate_ssim
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    from torchmetrics.image.kid import KernelInceptionDistance
    from pytorch_fid.fid_score import calculate_fid_given_paths
except ImportError as e:
    print(f"\n❌ [严重错误] 缺少量化评估库：{str(e)}")
    print("请先在终端运行以下命令进行安装：")
    print("pip install torch torchvision torchmetrics[image] lpips pytorch-fid scikit-image")
    exit(1)

# ==========================================
# 0. 全局路径与基础配置
# ==========================================
ROOT_DIR = os.path.abspath(".")
REAL_DIR = os.path.join(ROOT_DIR, "data", "real")
SIM_DIR = os.path.join(ROOT_DIR, "data", "sim")
SIM64_DIR = os.path.join(ROOT_DIR, "data", "sim1_64x64")

GAN_DIR = os.path.join(ROOT_DIR, "contrastive-unpaired-translation-master")
INFERENCE_BASE_DIR = os.path.join(ROOT_DIR, "inference_results")
EVAL_RESULTS_DIR = os.path.join(ROOT_DIR, "evaluation_metrics")
OUTPUT_COLUMNS_DIR = os.path.join(ROOT_DIR, "final_visual_comparisons")

INCEPTION_DIM = 2048

os.makedirs(INFERENCE_BASE_DIR, exist_ok=True)
os.makedirs(EVAL_RESULTS_DIR, exist_ok=True)
os.makedirs(OUTPUT_COLUMNS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ 计算设备: {device}")


# ==========================================
# 1. 智能文件名解析器 & 获取 17° 测试对
# ==========================================
def parse_sim_name(name):
    name_no_ext = os.path.splitext(name)[0]
    match_new = re.search(r'El([\d\.]+)_Az([\d\.]+)', name_no_ext, re.IGNORECASE)
    if match_new: return float(match_new.group(1)), float(match_new.group(2))
    match_old = re.search(r'elevDeg_(\d+)_azCenter_(\d+)(?:_(\d+))?', name, re.IGNORECASE)
    if match_old: return float(match_old.group(1)), float(
        f"{match_old.group(2)}.{match_old.group(3) if match_old.group(3) else '0'}")
    return None, None


def parse_real_name(name):
    match = re.search(r'El([\d\.]+)_Az([\d\.]+)', os.path.splitext(name)[0], re.IGNORECASE)
    if match: return float(match.group(1)), float(match.group(2))
    return None, None


def get_test_pairs_only(sim_dir, real_dir):
    val_pairs = []
    if not os.path.exists(sim_dir): return []

    for class_folder in sorted(os.listdir(sim_dir)):
        sim_class_dir = os.path.join(sim_dir, class_folder)
        if not os.path.isdir(sim_class_dir): continue

        real_class_dir = os.path.join(real_dir, class_folder, "crop_original")
        if not os.path.exists(real_class_dir): real_class_dir = os.path.join(real_dir, class_folder)
        if not os.path.exists(real_class_dir): continue

        real_pool = [{'path': os.path.join(real_class_dir, r), 'el': parse_real_name(r)[0], 'az': parse_real_name(r)[1]}
                     for r in os.listdir(real_class_dir) if
                     r.lower().endswith(('.png', '.jpg')) and parse_real_name(r)[0] is not None]

        for s_name in sorted(os.listdir(sim_class_dir)):
            if not s_name.lower().endswith(('.png', '.jpg')): continue
            s_el, s_az = parse_sim_name(s_name)
            if s_el is None or abs(s_el - 17.0) >= 0.5: continue

            matched = next((r['path'] for r in real_pool if abs(r['el'] - s_el) < 0.5 and abs(r['az'] - s_az) < 0.5),
                           None)
            if matched:
                val_pairs.append({
                    'sim': os.path.join(sim_class_dir, s_name),
                    'real': matched, 'class': class_folder, 'name': s_name, 'el': s_el, 'az': s_az
                })
    return val_pairs


# ==========================================
# 2. 准备推理数据集并调用模型 (支持动态 K 值)
# ==========================================
def unaligned_copy_save(item, base_dir):
    file_name = f"{item['class']}_{item['name']}"
    shutil.copy(item['sim'], os.path.join(base_dir, "testA", file_name))
    shutil.copy(item['real'], os.path.join(base_dir, "testB", file_name))
    shutil.copy(item['sim'], os.path.join(base_dir, "trainA", file_name))
    shutil.copy(item['real'], os.path.join(base_dir, "trainB", file_name))


def run_inference_for_dataset_dynamic(val_pairs, dataset_tag, current_k_str):
    print(f"\n⚡ 正在准备 [{dataset_tag}] 数据的推理环境 (K={current_k_str})...")
    unaligned_data_dir = os.path.join(ROOT_DIR, "datasets", f"unaligned_test_{dataset_tag}")

    if os.path.exists(unaligned_data_dir):
        shutil.rmtree(unaligned_data_dir)

    for sub in ['testA', 'testB', 'trainA', 'trainB']:
        os.makedirs(os.path.join(unaligned_data_dir, sub), exist_ok=True)

    for item in tqdm(val_pairs, desc=f"装载 {dataset_tag} 测试数据", leave=False):
        unaligned_copy_save(item, unaligned_data_dir)

    num_copied = len(os.listdir(os.path.join(unaligned_data_dir, 'testA')))
    print(f"   ✅ 数据集装载完毕！[testA] 里有 {num_copied} 张图片。")

    base_cmd = "python test.py --dataset_mode unaligned --phase test --eval --load_size 64 --crop_size 64 --input_nc 1 --output_nc 1 --preprocess none --netG resnet_6blocks --batch_size 1"

    # 根据当前 K 值动态构建权重名
    weight_suffix = f"K_{current_k_str}"

    tasks = [
        {"name": f"CUT (K={current_k_str})", "model": "cut", "weights": f"cut_{weight_suffix}",
         "out_name": f"CUT_{dataset_tag}_{weight_suffix}"},
        {"name": f"CycleGAN (K={current_k_str})", "model": "cycle_gan", "weights": f"cyc_{weight_suffix}",
         "out_name": f"CycleGAN_{dataset_tag}_{weight_suffix}"}
    ]

    for task in tasks:
        print(f"   ▶ 正在运行 {task['name']} 推理 on {dataset_tag}...")

        dest_dir = os.path.join(INFERENCE_BASE_DIR, task['out_name'])
        if os.path.exists(dest_dir): shutil.rmtree(dest_dir)
        os.makedirs(dest_dir, exist_ok=True)

        cmd = f"{base_cmd} --dataroot \"{unaligned_data_dir}\" --name {task['weights']} --model {task['model']} --num_test {len(val_pairs)}"

        res = subprocess.run(cmd, shell=True, cwd=GAN_DIR, capture_output=True, text=True, encoding='utf-8',
                             errors='replace')

        if res.returncode != 0:
            stderr_text = res.stderr if res.stderr is not None else "无错误输出"
            print(f"   ❌ {task['name']} 推理失败！报错:\n{stderr_text[-300:]}")
            continue

        target_results_dir = os.path.join(GAN_DIR, "results", task['weights'])
        extracted = 0
        if os.path.exists(target_results_dir):
            for root, dirs, files in os.walk(target_results_dir):
                is_fake_folder = os.path.basename(root) == "fake_B"

                for f in files:
                    if not f.endswith(('.png', '.jpg')): continue

                    if is_fake_folder or "fake_B" in f or "fake" in f:
                        if "real" in f: continue

                        src_path = os.path.join(root, f)
                        clean_name = f.replace("_fake_B.png", ".png").replace("_fake.png", ".png")
                        dest_path = os.path.join(dest_dir, clean_name)

                        shutil.copy(src_path, dest_path)
                        extracted += 1

        if extracted > 0:
            print(f"   ✅ 成功提取了 {extracted} 张推理结果！")
        else:
            print(f"   ❌ 找不到生成的图片，可能是因为权重文件夹 {target_results_dir} 不存在。")


# ==========================================
# 3. 深度学习量化评估引擎
# ==========================================
@torch.no_grad()
def evaluate_quantitative_for_single_model(fake_images_dir, val_pairs, output_filename):
    print(f"\n📏 正在计算指标 -> {output_filename}...")

    if not os.path.exists(fake_images_dir) or not os.listdir(fake_images_dir):
        print("   ❌ 指标计算失败: 推理文件夹为空，跳过。")
        return None

    temp_real_matched_dir = os.path.join(EVAL_RESULTS_DIR, f"temp_real_{output_filename}")
    paired_fake_dir = os.path.join(EVAL_RESULTS_DIR, f"temp_fake_{output_filename}")

    for d in [temp_real_matched_dir, paired_fake_dir]:
        if os.path.exists(d): shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    valid_pairs_count = 0
    ssim_list, lpips_list = [], []

    kid_subset_size = min(50, len(val_pairs)) if len(val_pairs) > 0 else 50
    kid_metric = KernelInceptionDistance(subset_size=kid_subset_size).to(device)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(device)

    for item in val_pairs:
        base_fake_name = os.path.splitext(f"{item['class']}_{item['name']}")[0]
        fake_path = os.path.join(fake_images_dir, base_fake_name + ".png")
        real_path = item['real']

        if not os.path.exists(fake_path): continue

        file_matched_name = f"{item['class']}_El{item['el']:.1f}_Az{item['az']:.1f}.png"
        shutil.copy(fake_path, os.path.join(paired_fake_dir, file_matched_name))
        shutil.copy(real_path, os.path.join(temp_real_matched_dir, file_matched_name))

        img_fake_pil = Image.open(fake_path).convert('L').resize((64, 64))
        img_real_pil = Image.open(real_path).convert('L').resize((64, 64))

        s_val = calculate_ssim(np.array(img_real_pil), np.array(img_fake_pil), data_range=255)
        ssim_list.append(s_val)

        ts_fake_rgb = transforms.ToTensor()(img_fake_pil.convert('RGB')).unsqueeze(0).to(device)
        ts_real_rgb = transforms.ToTensor()(img_real_pil.convert('RGB')).unsqueeze(0).to(device)

        l_val = lpips_metric(ts_fake_rgb * 2.0 - 1.0, ts_real_rgb * 2.0 - 1.0).item()
        lpips_list.append(l_val)

        kid_metric.update((ts_fake_rgb * 255).byte(), real=False)
        kid_metric.update((ts_real_rgb * 255).byte(), real=True)
        valid_pairs_count += 1

    if valid_pairs_count == 0: return None

    avg_ssim = sum(ssim_list) / len(ssim_list)
    avg_lpips = sum(lpips_list) / len(lpips_list)

    try:
        kid_val = kid_metric.compute()[0].item()
    except:
        kid_val = np.nan

    try:
        fid_val = calculate_fid_given_paths([temp_real_matched_dir, paired_fake_dir], batch_size=50, device=device,
                                            dims=INCEPTION_DIM, num_workers=2)
    except:
        fid_val = np.nan

    shutil.rmtree(temp_real_matched_dir, ignore_errors=True)
    shutil.rmtree(paired_fake_dir, ignore_errors=True)

    return {'ssim': avg_ssim, 'lpips': avg_lpips, 'fid': fid_val, 'kid': kid_val}


# ==========================================
# 4. 生成对比长图 (动态后缀版)
# ==========================================
def stitch_vertical_columns(val_pairs_sim, val_pairs_sim64, output_dir, img_size=64, spacing=8, k_str="4"):
    print(f"\n📸 开始拼接测试集的终极对比长图 (仅针对 K={k_str})...")
    if os.path.exists(output_dir): shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    def get_physical_key(item):
        return f"{item['class']}_El{item['el']:.1f}_Az{item['az']:.1f}"

    sim_dict = {get_physical_key(item): item for item in val_pairs_sim}
    sim64_dict = {get_physical_key(item): item for item in val_pairs_sim64}
    success_count = 0

    weight_suffix = f"K_{k_str}"

    for key, item_sim in tqdm(sim_dict.items(), desc="Stitching Images"):
        if key not in sim64_dict: continue
        item_sim64 = sim64_dict[key]

        name_sim = os.path.splitext(f"{item_sim['class']}_{item_sim['name']}")[0] + ".png"
        name_sim64 = os.path.splitext(f"{item_sim64['class']}_{item_sim64['name']}")[0] + ".png"

        paths = [
            ("1. Real", item_sim['real']),
            ("2. sim64 src", item_sim64['sim']),
            ("3. sim src", item_sim['sim']),
            ("4. Cycle+sim64", os.path.join(INFERENCE_BASE_DIR, f"CycleGAN_sim64_{weight_suffix}", name_sim64)),
            ("5. Cycle+sim", os.path.join(INFERENCE_BASE_DIR, f"CycleGAN_sim_{weight_suffix}", name_sim)),
            ("6. CUT+sim64", os.path.join(INFERENCE_BASE_DIR, f"CUT_sim64_{weight_suffix}", name_sim64)),
            ("7. CUT+sim", os.path.join(INFERENCE_BASE_DIR, f"CUT_sim_{weight_suffix}", name_sim))
        ]

        if any(not os.path.exists(p) for _, p in paths): continue

        canvas = Image.new('L', (img_size, (img_size * 7) + (spacing * 6)), color=255)
        y = 0
        for _, p in paths:
            canvas.paste(Image.open(p).convert('L').resize((img_size, img_size)), (0, y))
            y += (img_size + spacing)

        canvas.save(os.path.join(output_dir, f"{item_sim['class']}_El{item_sim['el']:.2f}_Az{item_sim['az']:.2f}.png"))
        success_count += 1

    print(f"🎉 拼接大功告成！成功生成 {success_count} 张对比长图。")


# ==========================================
# 🚀 主程序大满贯流水线
# ==========================================
if __name__ == "__main__":
    print("=" * 60)
    print("🔥 CUT & CycleGAN 全量流水线启动 (包含 K=1, 4, 16, All)")
    print("=" * 60)

    pairs_sim = get_test_pairs_only(SIM_DIR, REAL_DIR)
    pairs_sim64 = get_test_pairs_only(SIM64_DIR, REAL_DIR)

    if not pairs_sim or not pairs_sim64:
        print("❌ 错误：未找到测试集。")
        exit(1)

    # 准备超级大表格
    master_eval_table = [["Setting", "Method", "SSIM ↑", "LPIPS ↓", "FID ↓", "KID ↓"]]

    # 待测试的 K 值序列
    k_settings = ['1', '4', '16', 'All']

    for k_val in k_settings:
        print(f"\n\n{'=' * 40}")
        print(f"🌟 开始处理实验组: K = {k_val}")
        print(f"{'=' * 40}")

        # 1. 运行推理 (包含 sim 和 sim64)
        run_inference_for_dataset_dynamic(pairs_sim, "sim", k_val)
        run_inference_for_dataset_dynamic(pairs_sim64, "sim64", k_val)

        # 2. 准备当前 K 值的评估配置
        configs = [
            (f"CycleGAN+sim", f"CycleGAN_sim_K_{k_val}", pairs_sim),
            (f"CycleGAN+sim64", f"CycleGAN_sim64_K_{k_val}", pairs_sim64),
            (f"CUT+sim", f"CUT_sim_K_{k_val}", pairs_sim),
            (f"CUT+sim64", f"CUT_sim64_K_{k_val}", pairs_sim64),
        ]

        print(f"\n📊 开始量化评估 (K={k_val})...")
        for name, d_suffix, t_pairs in configs:
            res = evaluate_quantitative_for_single_model(os.path.join(INFERENCE_BASE_DIR, d_suffix), t_pairs, d_suffix)
            if res:
                master_eval_table.append(
                    [f"K={k_val}", name, f"{res['ssim']:.4f}", f"{res['lpips']:.4f}", f"{res['fid']:.3f}",
                     f"{res['kid']:.6f}"]
                )

        # 3. 按照需求：仅仅在 K=4 时保留长图
        if k_val == '4':
            stitch_vertical_columns(pairs_sim, pairs_sim64, OUTPUT_COLUMNS_DIR, spacing=8, k_str=k_val)
        else:
            print(f"\n⏭️ 跳过长图拼接 (当前 K={k_val}，仅 K=4 触发)。")

    # 终极输出汇总报告
    print("\n\n" + "=" * 80)
    print("🏆 全量实验组终极量化评估报告 (适用于消融实验)")
    print("-" * 80)
    if len(master_eval_table) > 1:
        widths = [max(len(row[i]) for row in master_eval_table) + 2 for i in range(len(master_eval_table[0]))]
        print("".join(master_eval_table[0][i].ljust(widths[i]) for i in range(len(master_eval_table[0]))))
        print("-" * 80)

        # 为了美观，当 K 值改变时打印一条分割线
        current_k_group = master_eval_table[1][0]
        for row in master_eval_table[1:]:
            if row[0] != current_k_group:
                print("-" * 80)
                current_k_group = row[0]
            print("".join(row[i].ljust(widths[i]) for i in range(len(row))))
    print("=" * 80 + "\n")