import os
import shutil
import random
import re
import numpy as np
from PIL import Image
import subprocess
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from tqdm import tqdm

# ==========================================
# 0. 全局路径与基础配置
# ==========================================
ROOT_DIR = os.path.abspath(".")
SIM_DIR = os.path.join(ROOT_DIR, "data", "sim")
REAL_DIR = os.path.join(ROOT_DIR, "data", "real")
GAN_DIR = os.path.join(ROOT_DIR, "contrastive-unpaired-translation-master")
INFERENCE_BASE_DIR = os.path.join(ROOT_DIR, "inference_results")
LOG_DIR = os.path.join(ROOT_DIR, "run_logs")

# 🚀 优化配置
MAX_IO_WORKERS = 16
MAX_PARALLEL_JOBS = 2  # 同时启动的任务数
GAN_THREADS = 8

# ✨ 自动化流水线目标：依次跑完 1-shot, 4-shot, 16-shot 和 全量数据(None代表ALL)
EXPERIMENTS = [1, 4, 16, None]

os.makedirs(INFERENCE_BASE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


# ==========================================
# 1. 智能文件名解析器
# ==========================================
def parse_sim_name(name):
    name_no_ext = os.path.splitext(name)[0]
    match_new = re.search(r'El([\d\.]+)_Az([\d\.]+)', name_no_ext, re.IGNORECASE)
    if match_new:
        return float(match_new.group(1)), float(match_new.group(2))

    match_old = re.search(r'elevDeg_(\d+)_azCenter_(\d+)(?:_(\d+))?', name, re.IGNORECASE)
    if match_old:
        el = float(match_old.group(1))
        az_int = match_old.group(2)
        az_dec = match_old.group(3) if match_old.group(3) else "0"
        return el, float(f"{az_int}.{az_dec}")

    return None, None


def parse_real_name(name):
    name_no_ext = os.path.splitext(name)[0]
    match = re.search(r'El([\d\.]+)_Az([\d\.]+)', name_no_ext, re.IGNORECASE)
    if match:
        return float(match.group(1)), float(match.group(2))
    return None, None


# ==========================================
# 2. 数据配对逻辑
# ==========================================
def get_train_val_pairs(sim_dir, real_dir, k_shot_per_class=None):
    val_pairs, train_class_dict = [], {}
    if not os.path.exists(sim_dir): return [], []

    for class_folder in sorted(os.listdir(sim_dir)):
        sim_class_dir = os.path.join(sim_dir, class_folder)
        if not os.path.isdir(sim_class_dir): continue
        real_class_dir = os.path.join(real_dir, class_folder, "crop_original")
        if not os.path.exists(real_class_dir): real_class_dir = os.path.join(real_dir, class_folder)
        if not os.path.exists(real_class_dir): continue

        train_class_dict[class_folder] = []
        real_pool = []
        for r_name in os.listdir(real_class_dir):
            if not r_name.lower().endswith(('.png', '.jpg')): continue
            r_el, r_az = parse_real_name(r_name)
            if r_el is not None: real_pool.append(
                {'path': os.path.join(real_class_dir, r_name), 'el': r_el, 'az': r_az})

        for s_name in sorted(os.listdir(sim_class_dir)):
            if not s_name.lower().endswith(('.png', '.jpg')): continue
            s_el, s_az = parse_sim_name(s_name)
            if s_el is None: continue
            matched = next((r['path'] for r in real_pool if abs(r['el'] - s_el) < 0.5 and abs(r['az'] - s_az) < 0.5),
                           None)
            if matched:
                item = {'sim': os.path.join(sim_class_dir, s_name), 'real': matched, 'class': class_folder,
                        'name': s_name}
                if abs(s_el - 17.0) < 0.5:
                    val_pairs.append(item)
                else:
                    train_class_dict[class_folder].append(item)

    train_pairs = []
    cls_list = list(train_class_dict.keys())
    for idx, cls_name in enumerate(cls_list):
        pairs = train_class_dict[cls_name]
        if not pairs: continue
        if k_shot_per_class is None or len(pairs) <= k_shot_per_class:
            train_pairs.extend(pairs)
        else:
            itvl = len(pairs) / k_shot_per_class
            indices = [int((idx / len(cls_list) * itvl + i * itvl)) % len(pairs) for i in range(k_shot_per_class)]
            train_pairs.extend([pairs[i] for i in indices])
    return train_pairs, val_pairs


# ==========================================
# 3. 数据集快速并行准备
# ==========================================
def unaligned_copy_save(item, base_dir, split):
    file_name = f"{item['class']}_{item['name']}"
    shutil.copy(item['sim'], os.path.join(base_dir, f"{split}A", file_name))
    shutil.copy(item['real'], os.path.join(base_dir, f"{split}B", file_name))


def prepare_unaligned_fast(train_pairs, val_pairs, base_dir):
    if os.path.exists(base_dir): shutil.rmtree(base_dir)
    for sub in ['trainA', 'trainB', 'testA', 'testB']:
        os.makedirs(os.path.join(base_dir, sub), exist_ok=True)
    with ThreadPoolExecutor(MAX_IO_WORKERS) as pool:
        for item in train_pairs: pool.submit(unaligned_copy_save, item, base_dir, 'train')
        for item in val_pairs: pool.submit(unaligned_copy_save, item, base_dir, 'test')


# ==========================================
# 4. 提取结果
# ==========================================
def extract_results_fast(src_results, dest_inference):
    os.makedirs(dest_inference, exist_ok=True)
    img_dir = os.path.join(src_results, "test_latest", "images")
    if not os.path.exists(img_dir): return
    for f in os.listdir(img_dir):
        if f.endswith("_fake_B.png"):
            shutil.copy(os.path.join(img_dir, f), os.path.join(dest_inference, f.replace("_fake_B.png", ".png")))


# ==========================================
# 5. 单个实验工作流
# ==========================================
def run_single_k_experiment(k):
    suffix = f"K_{k}" if k is not None else "K_All"
    log_file = os.path.join(LOG_DIR, f"Pipeline_{suffix}.log")

    train_pairs, val_pairs = get_train_val_pairs(SIM_DIR, REAL_DIR, k)
    if not train_pairs: return f"⚠️ [跳过] {suffix}: 无有效数据"

    # ✨ 动态分配 Batch Size 的核心逻辑
    if k == 1:
        current_batch_size = 1
    elif k == 4:
        current_batch_size = 4
    else:
        current_batch_size = 8

    n_ep = 200 if (k is not None and k <= 16) else 50
    disk_saver_args = f"--save_epoch_freq {n_ep}"

    # 💡 专门为 64x64 SAR 图像锁死的尺寸参数
    size_args = "--load_size 64 --crop_size 64"

    with open(log_file, "w") as log:
        log.write(f"--- 启动实验 {suffix} | 动态 Batch Size 分配: {current_batch_size} ---\n\n")

        # 准备数据
        unaligned_data = os.path.join(ROOT_DIR, "datasets", f"unaligned_{suffix}")
        prepare_unaligned_fast(train_pairs, val_pairs, unaligned_data)

        # ==============================================
        # 阶段 A：执行 CycleGAN
        # ==============================================
        tr_cmd_cyc = f"python train.py --dataroot \"{unaligned_data}\" --name cyc_{suffix} --model cycle_gan " \
                     f"--input_nc 1 --output_nc 1 --preprocess none {size_args} --netG resnet_6blocks " \
                     f"--batch_size {current_batch_size} --num_threads {GAN_THREADS} " \
                     f"--no_html --display_id 0 " \
                     f"--n_epochs {n_ep} --n_epochs_decay {n_ep} {disk_saver_args}"
        subprocess.run(tr_cmd_cyc, shell=True, cwd=GAN_DIR, stdout=log, stderr=log, check=True)

        ts_cmd_cyc = f"python test.py --dataroot \"{unaligned_data}\" --name cyc_{suffix} --model cycle_gan " \
                     f"--input_nc 1 --output_nc 1 --preprocess none {size_args} --netG resnet_6blocks " \
                     f"--batch_size 1 --num_test {len(val_pairs)}"
        subprocess.run(ts_cmd_cyc, shell=True, cwd=GAN_DIR, stdout=log, stderr=log, check=True)

        extract_results_fast(os.path.join(GAN_DIR, "results", f"cyc_{suffix}"),
                             os.path.join(INFERENCE_BASE_DIR, f"CycleGAN_{suffix}"))

        # ==============================================
        # 阶段 B：执行 CUT
        # ==============================================
        # 💡 CUT 特供参数：--num_patches 128 防止 64x64 图像下采样后特征空间不足导致越界崩溃
        tr_cmd_cut = f"python train.py --dataroot \"{unaligned_data}\" --name cut_{suffix} --model cut " \
                     f"--input_nc 1 --output_nc 1 --preprocess none {size_args} --netG resnet_6blocks " \
                     f"--num_patches 128 " \
                     f"--batch_size {current_batch_size} --num_threads {GAN_THREADS} " \
                     f"--no_html --display_id 0 " \
                     f"--n_epochs {n_ep} --n_epochs_decay {n_ep} {disk_saver_args}"
        subprocess.run(tr_cmd_cut, shell=True, cwd=GAN_DIR, stdout=log, stderr=log, check=True)

        ts_cmd_cut = f"python test.py --dataroot \"{unaligned_data}\" --name cut_{suffix} --model cut " \
                     f"--input_nc 1 --output_nc 1 --preprocess none {size_args} --netG resnet_6blocks " \
                     f"--batch_size 1 --num_test {len(val_pairs)}"
        subprocess.run(ts_cmd_cut, shell=True, cwd=GAN_DIR, stdout=log, stderr=log, check=True)

        extract_results_fast(os.path.join(GAN_DIR, "results", f"cut_{suffix}"),
                             os.path.join(INFERENCE_BASE_DIR, f"CUT_{suffix}"))

        # 清理共用数据集
        shutil.rmtree(unaligned_data, ignore_errors=True)

    return f"✅ [CycleGAN & CUT 双杀完成] {suffix} (Batch Size: {current_batch_size})"


# ==========================================
# 6. 主程序
# ==========================================
if __name__ == "__main__":
    print("=" * 60)
    print(f"🔥 双前沿基线全自动联跑模式启动 | 并行数: {MAX_PARALLEL_JOBS}")
    print("📈 实验序列: K=1, 4, 16, ALL")
    print("=" * 60)

    with ProcessPoolExecutor(max_workers=MAX_PARALLEL_JOBS) as executor:
        futures = [executor.submit(run_single_k_experiment, k) for k in EXPERIMENTS]

        for f in tqdm(as_completed(futures), total=len(EXPERIMENTS), desc="总体进度", unit="任务"):
            try:
                tqdm.write(f.result())
            except subprocess.CalledProcessError as e:
                tqdm.write(
                    f"❌ 实验命令崩溃。错误码: {e.returncode}。请在终端执行 `cat {LOG_DIR}/Pipeline_K_*.log` 查看具体原因。")
            except Exception as e:
                tqdm.write(f"❌ 遭遇未知失败: {e}")

    print("\n🎉 所有实验已完全执行完毕！推理图像已保存至 inference_results 目录。")