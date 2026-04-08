import os
import time
import math
import concurrent.futures
import queue
import threading

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["CUDA_MODULE_LOADING"] = "LAZY"

import torch
import numpy as np
from PIL import Image  # 用于保存扩散模型专属的纯净图像

# 【优化】彻底禁用梯度计算
torch.set_grad_enabled(False)

try:
    import warp as wp

    wp.config.mode = "release"
    wp.config.verify_fp = False
    wp.config.verify_bounds = False
    wp.init()
except ImportError:
    print("【严重错误】缺少 warp-lang 库，请在终端运行: pip install warp-lang")
    exit(1)

try:
    from pytorch3d.ops import knn_points

    HAS_PYTORCH3D = True
except ImportError:
    HAS_PYTORCH3D = False


# ================= 辅助工具：精准 GPU 计时器 =================
def sync_time(start_time, stage_name, pt_stream=None, device_str='cuda:0'):
    if pt_stream is not None:
        pt_stream.synchronize()
    elif 'cuda' in device_str:
        torch.cuda.synchronize(device_str)
    return time.time()


# ================= 🚀 Warp 核心算子 =================
@wp.kernel
def _warp_compute_curvatures_kernel(
        centers: wp.array(dtype=wp.vec3),
        normals: wp.array(dtype=wp.vec3),
        part_ids: wp.array(dtype=wp.int64),
        face_neighbors: wp.array(dtype=wp.int32, ndim=2),
        neighbor_counts: wp.array(dtype=wp.int32),
        num_faces: int,
        cos_thresh: float,
        huge_r: float,
        k_eps: float,
        face_r1_out: wp.array(dtype=wp.float32),
        face_r2_out: wp.array(dtype=wp.float32)
):
    tid = wp.tid()
    if tid >= num_faces:
        return

    n0 = normals[tid]
    c0 = centers[tid]
    part0 = part_ids[tid]
    num_nbrs = neighbor_counts[tid]

    if num_nbrs < 6:
        face_r1_out[tid] = float(-1.0)
        face_r2_out[tid] = float(-1.0)
        return

    ref = wp.vec3(1.0, 0.0, 0.0)
    if wp.abs(n0[0]) >= 0.9:
        ref = wp.vec3(0.0, 1.0, 0.0)

    t1 = wp.cross(n0, ref)
    norm_t1 = wp.length(t1)
    if norm_t1 < 1e-14:
        ref = wp.vec3(0.0, 0.0, 1.0)
        t1 = wp.cross(n0, ref)
        norm_t1 = wp.length(t1)
        if norm_t1 < 1e-14:
            face_r1_out[tid] = float(-1.0)
            face_r2_out[tid] = float(-1.0)
            return

    t1 = t1 / norm_t1
    t2 = wp.cross(n0, t1)
    norm_t2 = wp.length(t2)
    if norm_t2 < 1e-14:
        face_r1_out[tid] = float(-1.0)
        face_r2_out[tid] = float(-1.0)
        return
    t2 = t2 / norm_t2

    suu = float(0.0)
    suv = float(0.0)
    svv = float(0.0)
    sup = float(0.0)
    suq = float(0.0)
    svp = float(0.0)
    svq = float(0.0)
    valid_count = int(0)

    sigma_dist = float(0.05)

    for i in range(num_nbrs):
        nbr_idx = face_neighbors[tid, i]
        if part_ids[nbr_idx] != part0:
            continue

        nbr_n = normals[nbr_idx]
        dot_n = wp.dot(nbr_n, n0)

        if dot_n < cos_thresh:
            continue

        dx = centers[nbr_idx] - c0
        dn = nbr_n - n0

        u = wp.dot(dx, t1)
        v = wp.dot(dx, t2)

        uv_norm2 = u * u + v * v
        if uv_norm2 > 1e-16:
            p = wp.dot(dn, t1)
            q = wp.dot(dn, t2)

            dist2 = uv_norm2 + wp.dot(dx, n0) * wp.dot(dx, n0)
            w_dist = wp.exp(-dist2 / (2.0 * sigma_dist * sigma_dist))
            w_norm = wp.exp(-(1.0 - dot_n) * 10.0)
            w = w_dist * w_norm

            suu += w * u * u
            suv += w * u * v
            svv += w * v * v
            sup += w * u * p
            suq += w * u * q
            svp += w * v * p
            svq += w * v * q
            valid_count += 1

    if valid_count < 6:
        face_r1_out[tid] = float(-1.0)
        face_r2_out[tid] = float(-1.0)
        return

    det = suu * svv - suv * suv
    if wp.abs(det) < 1e-16:
        face_r1_out[tid] = float(-1.0)
        face_r2_out[tid] = float(-1.0)
        return

    inv00 = svv / det
    inv01 = -suv / det
    inv11 = suu / det

    m00 = inv00 * sup + inv01 * svp
    m01 = inv00 * suq + inv01 * svq
    m10 = inv01 * sup + inv11 * svp
    m11 = inv01 * suq + inv11 * svq

    b = -0.5 * (m01 + m10)
    a = -m00
    d = -m11

    tr_half = 0.5 * (a + d)
    rad_inner = (0.5 * (a - d)) * (0.5 * (a - d)) + b * b
    if rad_inner < 0.0:
        rad_inner = float(0.0)
    rad = wp.sqrt(rad_inner)

    k1 = tr_half - rad
    k2 = tr_half + rad

    ak1 = wp.abs(k1)
    ak2 = wp.abs(k2)

    k_small = ak1
    k_large = ak2
    if ak1 > ak2:
        k_small = ak2
        k_large = ak1

    r_small = huge_r
    if k_small >= k_eps:
        r_small = wp.min(huge_r, 1.0 / k_small)

    r_large = huge_r
    if k_large >= k_eps:
        r_large = wp.min(huge_r, 1.0 / k_large)

    face_r1_out[tid] = r_small
    face_r2_out[tid] = r_large


@wp.kernel
def _warp_fill_invalid_curvatures_kernel(
        face_r1_in: wp.array(dtype=wp.float32),
        face_r2_in: wp.array(dtype=wp.float32),
        part_ids: wp.array(dtype=wp.int64),
        normals: wp.array(dtype=wp.vec3),
        face_neighbors: wp.array(dtype=wp.int32, ndim=2),
        neighbor_counts: wp.array(dtype=wp.int32),
        num_faces: int,
        cos_thresh: float,
        huge_r: float,
        face_r1_out: wp.array(dtype=wp.float32),
        face_r2_out: wp.array(dtype=wp.float32)
):
    tid = wp.tid()
    if tid >= num_faces:
        return

    r1_val = face_r1_in[tid]
    r2_val = face_r2_in[tid]

    if r1_val >= 0.0:
        face_r1_out[tid] = r1_val
        face_r2_out[tid] = r2_val
        return

    num_nbrs = neighbor_counts[tid]
    part0 = part_ids[tid]
    n0 = normals[tid]

    sum_k1 = float(0.0)
    sum_k2 = float(0.0)
    valid_nbrs = int(0)

    for i in range(num_nbrs):
        nbr_idx = face_neighbors[tid, i]

        if part_ids[nbr_idx] != part0:
            continue

        if wp.dot(normals[nbr_idx], n0) < cos_thresh:
            continue

        nbr_r1 = face_r1_in[nbr_idx]
        nbr_r2 = face_r2_in[nbr_idx]

        if nbr_r1 >= 0.0:
            sum_k1 += 1.0 / nbr_r1
            sum_k2 += 1.0 / nbr_r2
            valid_nbrs += 1

    if valid_nbrs > 0:
        avg_k1 = sum_k1 / float(valid_nbrs)
        avg_k2 = sum_k2 / float(valid_nbrs)

        if avg_k1 < 1e-8:
            face_r1_out[tid] = huge_r
        else:
            face_r1_out[tid] = wp.min(huge_r, 1.0 / avg_k1)

        if avg_k2 < 1e-8:
            face_r2_out[tid] = huge_r
        else:
            face_r2_out[tid] = wp.min(huge_r, 1.0 / avg_k2)
    else:
        face_r1_out[tid] = huge_r
        face_r2_out[tid] = huge_r


@wp.kernel
def _warp_exact_phase_integral_kernel(
        w: wp.array(dtype=wp.vec3), n: wp.array(dtype=wp.vec3),
        p1: wp.array(dtype=wp.vec3), p2: wp.array(dtype=wp.vec3), p3: wp.array(dtype=wp.vec3),
        areas: wp.array(dtype=wp.float32), centers: wp.array(dtype=wp.vec3),
        phase_center: wp.array(dtype=wp.float32),
        out_re: wp.array(dtype=wp.float32), out_im: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    w_vec = w[tid]
    n_vec = n[tid]
    w_dot_n = wp.dot(w_vec, n_vec)
    wt = w_vec - w_dot_n * n_vec
    wt_sq = wp.dot(wt, wt)
    area = areas[tid]
    c_vec = centers[tid]
    pc_val = phase_center[tid]

    if wt_sq < 0.01:
        out_re[tid] = area * wp.cos(pc_val)
        out_im[tid] = area * wp.sin(pc_val)
        return

    sum_re = float(0.0)
    sum_im = float(0.0)
    for i in range(3):
        v1 = wp.vec3()
        v2 = wp.vec3()
        if i == 0:
            v1 = p1[tid]
            v2 = p2[tid]
        elif i == 1:
            v1 = p2[tid]
            v2 = p3[tid]
        else:
            v1 = p3[tid]
            v2 = p1[tid]

        a = v2 - v1
        c_edge = (v1 + v2) / 2.0
        rel_c_edge = c_edge - c_vec

        term1 = wp.dot(w_vec, wp.cross(a, n_vec))
        w_dot_a_half = wp.dot(w_vec, a) / 2.0
        sinc = float(1.0)
        if wp.abs(w_dot_a_half) >= 1e-4:
            sinc = wp.sin(w_dot_a_half) / w_dot_a_half

        w_dot_c_edge = wp.dot(w_vec, rel_c_edge)
        mag = term1 * sinc
        sum_re += mag * wp.cos(w_dot_c_edge)
        sum_im += mag * wp.sin(w_dot_c_edge)

    rel_re = sum_im / wt_sq
    rel_im = -sum_re / wt_sq

    out_re[tid] = rel_re * wp.cos(pc_val) - rel_im * wp.sin(pc_val)
    out_im[tid] = rel_re * wp.sin(pc_val) + rel_im * wp.cos(pc_val)


@wp.kernel
def _warp_cast_rays_kernel(
        mesh: wp.uint64, origins: wp.array(dtype=wp.vec3), dirs: wp.array(dtype=wp.vec3),
        num_faces: wp.int32, t_hit_out: wp.array(dtype=wp.float32), face_id_out: wp.array(dtype=wp.int32)):
    tid = wp.tid()
    orig = origins[tid]
    dir = dirs[tid]
    t = float(0.0)
    u = float(0.0)
    v = float(0.0)
    sign = float(0.0)
    n = wp.vec3()
    face_id = int(-1)

    hit = wp.mesh_query_ray(mesh, orig, dir, float(1e8), t, u, v, sign, n, face_id)
    if hit:
        t_hit_out[tid] = t
        face_id_out[tid] = face_id
    else:
        t_hit_out[tid] = float(1e8)
        face_id_out[tid] = num_faces


@torch.jit.script
def _compute_go_dir_jit(v_in: torch.Tensor, normals: torch.Tensor):
    return v_in - 2.0 * torch.sum(normals * v_in, dim=-1, keepdim=True) * normals


@torch.jit.script
def _compute_go_field_jit(normals: torch.Tensor, E_in: torch.Tensor):
    return -E_in + 2.0 * torch.sum(normals * E_in, dim=-1, keepdim=True) * normals


@wp.kernel
def _warp_sar_echo_kernel(
        points: wp.array(dtype=wp.vec3), amps_re: wp.array(dtype=wp.float32), amps_im: wp.array(dtype=wp.float32),
        alphas: wp.array(dtype=wp.float32), lens: wp.array(dtype=wp.float32),
        k_f_axis: wp.array(dtype=wp.float32), f_over_fc: wp.array(dtype=wp.float32),
        f_sinc_factor: wp.array(dtype=wp.float32), sin_phi_axis: wp.array(dtype=wp.float32),
        cp: wp.array(dtype=wp.float32), sp: wp.array(dtype=wp.float32),
        st: float, ct: float, num_points: int,
        out_re: wp.array(dtype=wp.float32, ndim=2), out_im: wp.array(dtype=wp.float32, ndim=2)):
    phi_idx, f_idx = wp.tid()

    k_f_val = k_f_axis[f_idx]
    f_ratio = f_over_fc[f_idx]
    f_sinc = f_sinc_factor[f_idx]

    cp_val = cp[phi_idx]
    sp_val = sp[phi_idx]
    sin_phi = sin_phi_axis[phi_idx]

    sum_re = float(0.0)
    sum_im = float(0.0)

    for i in range(num_points):
        pt = points[i]
        a_re = amps_re[i]
        a_im = amps_im[i]
        alpha = alphas[i]
        L = lens[i]

        r_proj = pt[0] * cp_val * st + pt[1] * sp_val * st + pt[2] * ct
        phase = k_f_val * r_proj

        if alpha == 1.0:
            x_alpha = f_ratio
        elif alpha == 0.5:
            x_alpha = wp.sqrt(f_ratio)
        else:
            x_alpha = float(1.0)

        sinc_term = float(1.0)
        if L > 1e-5:
            sinc_arg = L * sin_phi * f_sinc
            if wp.abs(sinc_arg) >= 1e-5:
                pi_sinc = wp.pi * sinc_arg
                sinc_term = wp.sin(pi_sinc) / pi_sinc

        phase_re = wp.cos(phase)
        phase_im = wp.sin(phase)

        mag_mod = x_alpha * sinc_term
        t2_re = a_re * mag_mod
        t2_im = a_im * mag_mod

        sum_re += t2_re * phase_re - t2_im * phase_im
        sum_im += t2_re * phase_im + t2_im * phase_re

    out_re[phi_idx, f_idx] = sum_re
    out_im[phi_idx, f_idx] = sum_im


# ================= 预处理模块 (零拷贝极致优化版) =================
class GeometryPreprocessor:
    @staticmethod
    def process_mesh(mesh_path, num_neighbors=25, device='cuda'):
        device_str = "cuda:0" if "cuda" in str(device) else "cpu"
        wp_device = wp.get_device(device_str)
        print(f"\n[*] 正在解析网格: {mesh_path} ...")
        t_start = time.time()

        try:
            import pandas as pd
            df = pd.read_csv(mesh_path, sep=r'\s+', skiprows=1, header=None, engine='c', on_bad_lines='skip')
            verts_df = df[df.columns[:4]].dropna()
            faces_df = df[df.columns[:5]].dropna()

            verts_np = verts_df.iloc[:, 1:4].values.astype(np.float32)
            faces_np = faces_df.iloc[:, 1:4].values.astype(np.int64)
            part_ids_np = faces_df.iloc[:, 4].values.astype(np.int64)
        except Exception:
            with open(mesh_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = [line.strip() for line in f if line.strip()]

            verts, faces, part_ids = [], [], []
            for line in lines[1:]:
                vals = line.split()
                if len(vals) == 4:
                    try:
                        verts.append([float(vals[1]), float(vals[2]), float(vals[3])])
                    except:
                        pass
                elif len(vals) >= 5:
                    try:
                        faces.append([int(float(vals[1])), int(float(vals[2])), int(float(vals[3]))])
                        part_ids.append(int(float(vals[4])))
                    except:
                        pass

            verts_np = np.array(verts, dtype=np.float32)
            faces_np = np.array(faces, dtype=np.int64)
            part_ids_np = np.array(part_ids, dtype=np.int64)

        if faces_np.size > 0 and np.max(faces_np) >= len(verts_np):
            faces_np -= 1

        num_faces = len(faces_np)
        t_parse = time.time()
        print(f"    [Timer] 网格读取与解析耗时: {t_parse - t_start:.4f} 秒 (顶点:{len(verts_np)}, 面片:{num_faces})")

        verts_t = torch.tensor(verts_np, dtype=torch.float32, device=device_str)
        faces_t = torch.tensor(faces_np, dtype=torch.long, device=device_str)
        part_ids_t = torch.tensor(part_ids_np, dtype=torch.long, device=device_str)

        v0 = verts_t[faces_t[:, 0]]
        v1 = verts_t[faces_t[:, 1]]
        v2 = verts_t[faces_t[:, 2]]

        centers_t = (v0 + v1 + v2) / 3.0

        e01 = v1 - v0
        e02 = v2 - v0
        cross = torch.cross(e01, e02, dim=1)
        areas = 0.5 * torch.norm(cross, dim=1)
        face_normals_t = cross / (2.0 * areas.unsqueeze(1) + 1e-12)
        face_normals_t = torch.nn.functional.normalize(face_normals_t, dim=1)

        if 'cuda' in device_str:
            torch.cuda.synchronize(device_str)
        t_geom = time.time()

        num_verts = len(verts_np)
        vert_to_faces = [[] for _ in range(num_verts)]
        for f_idx, tri in enumerate(faces_np):
            vert_to_faces[int(tri[0])].append(f_idx)
            vert_to_faces[int(tri[1])].append(f_idx)
            vert_to_faces[int(tri[2])].append(f_idx)

        max_nbrs = 40
        neighbors_padded = np.zeros((num_faces, max_nbrs), dtype=np.int32)
        neighbors_count = np.zeros(num_faces, dtype=np.int32)

        for f_idx, tri in enumerate(faces_np):
            neigh = set()
            neigh.update(vert_to_faces[int(tri[0])])
            neigh.update(vert_to_faces[int(tri[1])])
            neigh.update(vert_to_faces[int(tri[2])])
            neigh.discard(f_idx)

            n_list = list(neigh)
            count = min(len(n_list), max_nbrs)
            neighbors_count[f_idx] = count
            neighbors_padded[f_idx, :count] = n_list[:count]

        neighbors_t = torch.tensor(neighbors_padded, dtype=torch.int32, device=device_str).contiguous()
        counts_t = torch.tensor(neighbors_count, dtype=torch.int32, device=device_str).contiguous()

        t_main_start = time.time()

        wp_centers = wp.from_torch(centers_t.contiguous(), dtype=wp.vec3)
        wp_normals = wp.from_torch(face_normals_t.contiguous(), dtype=wp.vec3)
        wp_parts = wp.from_torch(part_ids_t.contiguous(), dtype=wp.int64)
        wp_neighbors = wp.from_torch(neighbors_t, dtype=wp.int32)
        wp_counts = wp.from_torch(counts_t, dtype=wp.int32)

        out_r1_raw_t = torch.zeros(num_faces, dtype=torch.float32, device=device_str).contiguous()
        out_r2_raw_t = torch.zeros(num_faces, dtype=torch.float32, device=device_str).contiguous()
        wp_r1_raw = wp.from_torch(out_r1_raw_t, dtype=wp.float32)
        wp_r2_raw = wp.from_torch(out_r2_raw_t, dtype=wp.float32)

        wp.launch(
            kernel=_warp_compute_curvatures_kernel,
            dim=num_faces,
            inputs=[
                wp_centers, wp_normals, wp_parts, wp_neighbors, wp_counts, num_faces,
                math.cos(math.radians(30.0)), 1e6, 1e-8, wp_r1_raw, wp_r2_raw
            ],
            device=wp_device
        )

        face_r1_final_t = torch.zeros(num_faces, dtype=torch.float32, device=device_str).contiguous()
        face_r2_final_t = torch.zeros(num_faces, dtype=torch.float32, device=device_str).contiguous()
        wp_r1_final = wp.from_torch(face_r1_final_t, dtype=wp.float32)
        wp_r2_final = wp.from_torch(face_r2_final_t, dtype=wp.float32)

        wp.launch(
            kernel=_warp_fill_invalid_curvatures_kernel,
            dim=num_faces,
            inputs=[
                wp_r1_raw, wp_r2_raw, wp_parts, wp_normals, wp_neighbors, wp_counts, num_faces,
                math.cos(math.radians(30.0)), 1e6, wp_r1_final, wp_r2_final
            ],
            device=wp_device
        )

        wp.synchronize_device(wp_device)

        t_main_end = time.time()
        print(f"    [Timer] 曲率预处理计算总计耗时: {t_main_end - t_main_start:.4f} 秒")

        return verts_t, faces_t, part_ids_t, face_r1_final_t, face_r2_final_t, face_normals_t


# ================= 🚀 完美回归 PFA 原理的 SAR 仿真器 =================
class PyTorchSARSimulator:
    def __init__(self, fc=9.6e9, resolution=0.202, image_grid_size=64, range_scope=16.0, device='cuda'):
        """
        全量恢复过采样 (oversample)、PFA 极坐标转直角坐标网格计算 (grid_sample)
        以及 Kaiser 窗降旁瓣逻辑。专为生成高物理保真度的数据而设。
        """
        self.device_str = "cuda:0" if "cuda" in str(device) else "cpu"
        self.wp_device = wp.get_device(self.device_str)

        self.c, self.fc, self.res, self.N_pix = 3e8, fc, resolution, image_grid_size
        self.range_scope = range_scope
        self.B = self.c / (2 * self.res)

        # 恢复 6 倍过采样，确保高频相位不走样
        self.oversample = 6.0
        self.delta_f = self.B / (self.range_scope / self.res * self.oversample)
        self.f0 = self.fc - self.B / 2
        self.N_f = int(round(self.B / self.delta_f)) + 1
        self.f_axis = torch.linspace(0, self.N_f - 1, self.N_f, device=self.device_str) * self.delta_f + self.f0

        self.k_f_axis = (-4.0 * math.pi * self.f_axis / self.c).contiguous()
        self.f_over_fc = (self.f_axis / self.fc).contiguous()
        self.f_sinc_factor = (2.0 * self.f_axis / self.c).contiguous()

        self._pfa_cache = {}
        self._phi_cache = {}

    def _get_or_build_pfa_cache(self, el_deg):
        """
        因为你的批处理涉及多个不同的俯仰角 (el_deg)，由于合成孔径原理，
        不同俯仰角所需的方位角跨度 (anglephiwide) 是不同的。动态缓存机制完美解决此问题。
        """
        if el_deg in self._pfa_cache:
            return self._pfa_cache[el_deg]

        # 物理：俯仰角(el)转为天顶角(th_c)
        th_c_rad = math.radians(90.0 - el_deg) if el_deg != 90.0 else 1e-9

        # 计算不同天顶角下的合成孔径范围，并进行过采样
        anglephiwide_deg = 2 * math.degrees(math.asin(self.c / (4 * self.f0 * self.res))) / math.sin(th_c_rad)
        anglephi_step = anglephiwide_deg / (self.range_scope / self.res * self.oversample)
        N_phi = int(round(anglephiwide_deg / anglephi_step)) + 1
        phi_axis_local_deg = torch.linspace(-anglephiwide_deg / 2, anglephiwide_deg / 2, N_phi, device=self.device_str)

        # 为当前仰角分配独立的 Warp 回波接收缓冲区
        out_re_buf = torch.zeros((N_phi, self.N_f), dtype=torch.float32, device=self.device_str)
        out_im_buf = torch.zeros((N_phi, self.N_f), dtype=torch.float32, device=self.device_str)
        sin_phi_axis = torch.sin(torch.deg2rad(phi_axis_local_deg)).contiguous()

        # ================== 核心 PFA 插值网格重构 ==================
        phi_range_rad = math.radians(phi_axis_local_deg[-1].item() - phi_axis_local_deg[0].item())
        f_min, f_max = self.f_axis[0].item(), self.f_axis[-1].item()
        phi_min, phi_max = phi_axis_local_deg[0].item(), phi_axis_local_deg[-1].item()

        dx = self.c / (4 * f_min * math.sin(phi_range_rad / 2))
        Hy = 2 * f_min * math.tan(phi_range_rad / 2)
        Kx_Max = math.sqrt(arg) if (arg := f_max ** 2 - (Hy / 2) ** 2) >= 0 else f_max
        dy = self.c / (2 * abs(Kx_Max - f_min))

        # 插值到 64x64 最终图像网格
        kx = torch.linspace(f_min, Kx_Max, self.N_pix, device=self.device_str)
        ky = torch.linspace(-Hy / 2, Hy / 2, self.N_pix, device=self.device_str)
        KY_grid, KX_grid = torch.meshgrid(ky, kx, indexing='ij')

        complex_grid = KX_grid + 1j * KY_grid
        F_query = torch.abs(complex_grid)
        Phi_query_deg = torch.rad2deg(torch.angle(complex_grid))

        grid_x = 2.0 * (F_query - f_min) / (f_max - f_min) - 1.0
        grid_y = 2.0 * (Phi_query_deg - phi_min) / (phi_max - phi_min) - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).contiguous()

        # 完美还原：采用 Kaiser 窗极大地压制旁瓣！
        win_y = torch.kaiser_window(self.N_pix, periodic=False, beta=8.6, device=self.device_str)
        win_x = torch.kaiser_window(self.N_pix, periodic=False, beta=8.6, device=self.device_str)
        window_2d = torch.outer(win_y, win_x).contiguous()

        cache = {
            "N_phi": N_phi,
            "st": math.sin(th_c_rad),
            "ct": math.cos(th_c_rad),
            "phi_axis_local_deg": phi_axis_local_deg,
            "sin_phi_axis": sin_phi_axis,
            "out_re_buf": out_re_buf,
            "out_im_buf": out_im_buf,
            "grid": grid,
            "window_2d": window_2d
        }
        self._pfa_cache[el_deg] = cache
        return cache

    def generate_image_for_diffusion(self, results, el_deg, center_phi_deg, wp_stream=None):
        if not results:
            return np.zeros((self.N_pix, self.N_pix), dtype=np.float32)

        # 1. 取出当前俯仰角对应的独立物理参数和缓冲池
        cache = self._get_or_build_pfa_cache(el_deg)

        points = torch.tensor(np.stack([r['pos_arr'] for r in results]), dtype=torch.float32, device=self.device_str)
        alphas = torch.tensor([r['alpha'] for r in results], dtype=torch.float32, device=self.device_str)
        lens = torch.tensor([r['length'] for r in results], dtype=torch.float32, device=self.device_str)
        raw_A = torch.tensor([r['rcs_re'] + 1j * r['rcs_im'] for r in results], dtype=torch.complex64,
                             device=self.device_str)

        k_c = 2.0 * math.pi * self.fc / self.c
        ph_rad = math.radians(center_phi_deg)

        # 宏观相位校准中心系
        v_i_c = torch.tensor([-cache["st"] * math.cos(ph_rad), -cache["st"] * math.sin(ph_rad), -cache["ct"]],
                             dtype=torch.float32, device=self.device_str)
        macro_phase = -2.0 * k_c * torch.sum(points * v_i_c, dim=-1)
        amps = raw_A * torch.exp(-1j * macro_phase)

        # 构建局部方位角
        phi_math_rad = torch.deg2rad(center_phi_deg + cache["phi_axis_local_deg"])
        cp = torch.cos(phi_math_rad).contiguous()
        sp = torch.sin(phi_math_rad).contiguous()

        cache["out_re_buf"].zero_()
        cache["out_im_buf"].zero_()

        # 2. Warp 加速计算庞大过采样的回波极坐标阵列
        wp.launch(
            kernel=_warp_sar_echo_kernel, dim=(cache["N_phi"], self.N_f),
            inputs=[wp.from_torch(points.contiguous(), dtype=wp.vec3),
                    wp.from_torch(amps.real.contiguous(), dtype=wp.float32),
                    wp.from_torch(amps.imag.contiguous(), dtype=wp.float32),
                    wp.from_torch(alphas.contiguous(), dtype=wp.float32),
                    wp.from_torch(lens.contiguous(), dtype=wp.float32),
                    wp.from_torch(self.k_f_axis, dtype=wp.float32), wp.from_torch(self.f_over_fc, dtype=wp.float32),
                    wp.from_torch(self.f_sinc_factor, dtype=wp.float32),
                    wp.from_torch(cache["sin_phi_axis"], dtype=wp.float32),
                    wp.from_torch(cp, dtype=wp.float32), wp.from_torch(sp, dtype=wp.float32),
                    float(cache["st"]), float(cache["ct"]), int(points.shape[0]),
                    wp.from_torch(cache["out_re_buf"], dtype=wp.float32),
                    wp.from_torch(cache["out_im_buf"], dtype=wp.float32)],
            device=self.wp_device, stream=wp_stream
        )

        # 3. 经典的 PFA 插值：把高精度极坐标数据重采样为精准的直角坐标 (64x64)
        echo_2ch = torch.stack([cache["out_re_buf"], cache["out_im_buf"]], dim=0).unsqueeze(0)
        interp = torch.nn.functional.grid_sample(echo_2ch, cache["grid"], mode='bicubic', align_corners=True)

        # 4. 加 Kaiser 窗后逆傅里叶变换
        k_space_cart = (interp[0, 0] + 1j * interp[0, 1]) * cache["window_2d"]
        img_matrix = torch.abs(torch.fft.fftshift(torch.fft.ifft2(k_space_cart))).cpu().numpy()

        return img_matrix


# ================= 0. 全局统一极速光追引擎 =================
class GlobalZeroCopyBVH_Engine:
    def __init__(self, verts, faces, num_faces, device):
        self.device_str = "cuda:0" if "cuda" in str(device) else "cpu"
        self.wp_device = wp.get_device(self.device_str)
        self.num_faces = num_faces

        self.wp_verts = wp.from_torch(verts, dtype=wp.vec3)
        self.wp_faces = wp.from_torch(faces.view(-1), dtype=wp.int32)
        self.mesh = wp.Mesh(points=self.wp_verts, indices=self.wp_faces)

        self.max_rays = self.num_faces
        self.t_hit_buf = torch.empty(self.max_rays, dtype=torch.float32, device=self.device_str).contiguous()
        self.prim_id_buf = torch.empty(self.max_rays, dtype=torch.int32, device=self.device_str).contiguous()
        self.wp_t_hit = wp.from_torch(self.t_hit_buf, dtype=wp.float32)
        self.wp_primitive_ids = wp.from_torch(self.prim_id_buf, dtype=wp.int32)

    def cast_rays(self, origins, directions, wp_stream=None):
        num_rays = origins.shape[0]
        if num_rays == 0:
            return {
                't_hit': torch.empty(0, dtype=torch.float32, device=self.device_str),
                'primitive_ids': torch.empty(0, dtype=torch.long, device=self.device_str)
            }

        orig_c = origins.to(torch.float32).contiguous()
        dir_c = directions.to(torch.float32).contiguous()
        wp_origins = wp.from_torch(orig_c, dtype=wp.vec3)
        wp_dirs = wp.from_torch(dir_c, dtype=wp.vec3)

        wp.launch(
            kernel=_warp_cast_rays_kernel,
            dim=num_rays,
            inputs=[self.mesh.id, wp_origins, wp_dirs, wp.int32(self.num_faces), self.wp_t_hit, self.wp_primitive_ids],
            device=self.wp_device,
            stream=wp_stream
        )

        return {'t_hit': self.t_hit_buf[:num_rays], 'primitive_ids': self.prim_id_buf[:num_rays].long()}


# ================= 🚀 并行流水线物理引擎 =================
class StreamlinedRayTracingEngine:
    def __init__(self, verts_t, faces_t, part_ids_t, f_r1_t, f_r2_t, face_normals_t, device='cuda'):
        self.device_str = "cuda:0" if "cuda" in str(device) else "cpu"
        self.wp_device = wp.get_device(self.device_str)

        self.verts = verts_t.to(torch.float32).contiguous()
        self.faces = faces_t.to(torch.int32).contiguous()
        self.num_faces = self.faces.shape[0]
        self.part_ids = part_ids_t
        self.face_r1 = f_r1_t
        self.face_r2 = f_r2_t
        self.normals = face_normals_t

        self.faces_long = self.faces.long()
        self.v0 = self.verts[self.faces_long[:, 0]]
        self.v1 = self.verts[self.faces_long[:, 1]]
        self.v2 = self.verts[self.faces_long[:, 2]]

        cross = torch.cross(self.v1 - self.v0, self.v2 - self.v0, dim=-1)
        self.areas = (0.5 * torch.norm(cross, dim=-1)).contiguous()
        self.centers = ((self.v0 + self.v1 + self.v2) / 3.0).contiguous()

        self.scene = GlobalZeroCopyBVH_Engine(self.verts, self.faces, self.num_faces, self.device_str)

        self.cE_1_buf = torch.zeros(self.num_faces, dtype=torch.complex64, device=self.device_str)
        self.cE_2_buf = torch.zeros(self.num_faces, dtype=torch.complex64, device=self.device_str)
        self.cE_3_buf = torch.zeros(self.num_faces, dtype=torch.complex64, device=self.device_str)

        self.Q_eq2_buf = torch.zeros((self.num_faces, 3), dtype=torch.float32, device=self.device_str)
        self.Q_eq3_buf = torch.zeros((self.num_faces, 3), dtype=torch.float32, device=self.device_str)

        self.E_r1_buf = torch.zeros((self.num_faces, 3), dtype=torch.float32, device=self.device_str)
        self.E_r2_buf = torch.zeros((self.num_faces, 3), dtype=torch.float32, device=self.device_str)
        self.E_in2_buf = torch.zeros((self.num_faces, 3), dtype=torch.float32, device=self.device_str)
        self.E_in3_buf = torch.zeros((self.num_faces, 3), dtype=torch.float32, device=self.device_str)

        self.v_r1_buf = torch.zeros((self.num_faces, 3), dtype=torch.float32, device=self.device_str)
        self.v_r2_buf = torch.zeros((self.num_faces, 3), dtype=torch.float32, device=self.device_str)
        self.v_r3_buf = torch.zeros((self.num_faces, 3), dtype=torch.float32, device=self.device_str)

        self.I_re_buf = torch.empty(self.num_faces, dtype=torch.float32, device=self.device_str).contiguous()
        self.I_im_buf = torch.empty(self.num_faces, dtype=torch.float32, device=self.device_str).contiguous()

        self.s_po1 = torch.cuda.Stream(self.device_str)
        self.s_po2 = torch.cuda.Stream(self.device_str)
        self.s_po3 = torch.cuda.Stream(self.device_str)
        self.wps_po1 = wp.stream_from_torch(self.s_po1)
        self.wps_po2 = wp.stream_from_torch(self.s_po2)
        self.wps_po3 = wp.stream_from_torch(self.s_po3)

    def _compute_po_field(self, v_in, v_s, n, areas, centers, v0, v1, v2, E_in, E_rec, k, phase_center, wp_stream=None):
        J_eta = 2.0 * (
                torch.sum(n * E_in, dim=-1, keepdim=True) * v_in - torch.sum(n * v_in, dim=-1, keepdim=True) * E_in)
        E_rec_scalar = torch.sum(J_eta * E_rec, dim=-1)
        N_len = v_in.shape[0]
        w_in = -k * (v_in - v_s)

        I_re = self.I_re_buf[:N_len]
        I_im = self.I_im_buf[:N_len]

        wp.launch(
            kernel=_warp_exact_phase_integral_kernel, dim=N_len,
            inputs=[
                wp.from_torch(w_in.contiguous(), dtype=wp.vec3),
                wp.from_torch(n, dtype=wp.vec3),
                wp.from_torch(v0, dtype=wp.vec3),
                wp.from_torch(v1, dtype=wp.vec3),
                wp.from_torch(v2, dtype=wp.vec3),
                wp.from_torch(areas, dtype=wp.float32),
                wp.from_torch(centers, dtype=wp.vec3),
                wp.from_torch(phase_center, dtype=wp.float32),
                wp.from_torch(I_re, dtype=wp.float32),
                wp.from_torch(I_im, dtype=wp.float32)
            ],
            device=self.wp_device, stream=wp_stream
        )
        return (-1j * k / math.sqrt(4 * math.pi)) * E_rec_scalar * (I_re + 1j * I_im)

    def _prepare_cluster_payload_gpu(self, valid_mask, bounce_type, Q_equiv, complex_E, f_lists, v_out, v_i,
                                     flat_threshold):
        idx = torch.where(valid_mask)[0]
        if idx.numel() == 0: return None

        parts = [self.part_ids[f[idx]] for f in f_lists]

        if bounce_type == 1:
            keys = parts[0]
        elif bounce_type == 2:
            pmin = torch.minimum(parts[0], parts[1])
            pmax = torch.maximum(parts[0], parts[1])
            keys = pmin * 1000 + pmax
        else:
            sorted_p, _ = torch.sort(torch.stack([parts[0], parts[1], parts[2]], dim=1), dim=1)
            keys = sorted_p[:, 0] * 1000000 + sorted_p[:, 1] * 1000 + sorted_p[:, 2]

        unique_k, inv_idx = torch.unique(keys, return_inverse=True)
        num_c = unique_k.shape[0]
        if num_c == 0: return None

        E_c = complex_E[idx]
        pos_gpu = Q_equiv[idx]
        weights = torch.abs(E_c)

        sum_re = torch.bincount(inv_idx, weights=E_c.real, minlength=num_c)
        sum_im = torch.bincount(inv_idx, weights=E_c.imag, minlength=num_c)
        amp_lin_all = torch.hypot(sum_re, sum_im)

        max_amp_lin = torch.max(amp_lin_all)
        threshold_amp = torch.clamp(max_amp_lin * 0.02, min=1e-15)

        valid_clusters = torch.where(amp_lin_all >= threshold_amp)[0]
        if valid_clusters.numel() == 0: return None

        sum_W = torch.bincount(inv_idx, weights=weights, minlength=num_c) + 1e-12
        Q_c = torch.zeros((num_c, 3), dtype=torch.float32, device=self.device_str)
        Q_c[:, 0] = torch.bincount(inv_idx, weights=pos_gpu[:, 0] * weights, minlength=num_c) / sum_W
        Q_c[:, 1] = torch.bincount(inv_idx, weights=pos_gpu[:, 1] * weights, minlength=num_c) / sum_W
        Q_c[:, 2] = torch.bincount(inv_idx, weights=pos_gpu[:, 2] * weights, minlength=num_c) / sum_W

        align_threshold = math.cos(math.radians(10.0))
        align_dot = torch.sum(v_out[idx] * (-v_i), dim=-1)
        is_aligned = align_dot >= align_threshold

        r1_all_lists = [self.face_r1[f[idx]] for f in f_lists]
        r2_all_lists = [self.face_r2[f[idx]] for f in f_lists]
        normals_all_lists = [self.normals[f[idx]] for f in f_lists]

        survivor_mask = amp_lin_all[inv_idx] >= threshold_amp
        surv_idx = torch.where(survivor_mask)[0]
        if surv_idx.numel() == 0: return None

        surv_inv_idx = inv_idx[surv_idx]
        surv_weights = weights[surv_idx]
        surv_pos = pos_gpu[surv_idx]
        surv_is_aligned = is_aligned[surv_idx]

        surv_parts = [p[surv_idx] for p in parts]
        surv_r1 = [r[surv_idx] for r in r1_all_lists]
        surv_r2 = [r[surv_idx] for r in r2_all_lists]
        surv_normals = [n[surv_idx] for n in normals_all_lists]

        order = torch.argsort(surv_inv_idx)
        surv_inv_sorted = surv_inv_idx[order]

        surv_weights_sorted = surv_weights[order]
        surv_pos_sorted = surv_pos[order]
        surv_is_aligned_sorted = surv_is_aligned[order]

        surv_parts_sorted = [x[order] for x in surv_parts]
        surv_r1_sorted = [x[order] for x in surv_r1]
        surv_r2_sorted = [x[order] for x in surv_r2]
        surv_normals_sorted = [x[order] for x in surv_normals]

        change = torch.ones_like(surv_inv_sorted, dtype=torch.bool)
        if surv_inv_sorted.numel() > 1:
            change[1:] = surv_inv_sorted[1:] != surv_inv_sorted[:-1]

        seg_starts = torch.where(change)[0]
        seg_ends = torch.empty_like(seg_starts)
        if seg_starts.numel() > 1:
            seg_ends[:-1] = seg_starts[1:]
        seg_ends[-1] = surv_inv_sorted.numel()

        seg_cluster_ids = surv_inv_sorted[seg_starts]

        seg_lookup = torch.full((num_c,), -1, dtype=torch.long, device=self.device_str)
        seg_lookup[seg_cluster_ids] = torch.arange(seg_cluster_ids.numel(), device=self.device_str)

        valid_seg_ids = seg_lookup[valid_clusters]
        valid_seg_mask = valid_seg_ids >= 0
        valid_clusters = valid_clusters[valid_seg_mask]
        valid_seg_ids = valid_seg_ids[valid_seg_mask]

        if valid_clusters.numel() == 0: return None

        final_seg_starts = seg_starts[valid_seg_ids]
        final_seg_ends = seg_ends[valid_seg_ids]

        payload = {
            "bounce_type": bounce_type,
            "valid_clusters_cpu": valid_clusters.cpu().numpy(),
            "sum_re_cpu": sum_re[valid_clusters].cpu().numpy(),
            "sum_im_cpu": sum_im[valid_clusters].cpu().numpy(),
            "Q_c_cpu": Q_c[valid_clusters].cpu().numpy(),
            "seg_starts_cpu": final_seg_starts.cpu().numpy(),
            "seg_ends_cpu": final_seg_ends.cpu().numpy(),
            "surv_weights_cpu": surv_weights_sorted.cpu().numpy(),
            "surv_pos_cpu": surv_pos_sorted.cpu().numpy(),
            "surv_is_aligned_cpu": surv_is_aligned_sorted.cpu().numpy(),
            "surv_parts_cpu": [x.cpu().numpy() for x in surv_parts_sorted],
            "surv_r1_cpu": [x.cpu().numpy() for x in surv_r1_sorted],
            "surv_r2_cpu": [x.cpu().numpy() for x in surv_r2_sorted],
            "surv_normals_cpu": [x.cpu().numpy() for x in surv_normals_sorted],
        }
        return payload

    def _finalize_clusters_cpu(self, payload, flat_threshold, k, res_azimuth=0.1):
        if payload is None: return []

        bounce_type = payload["bounce_type"]
        valid_clusters_cpu = payload["valid_clusters_cpu"]
        sum_re_cpu = payload["sum_re_cpu"]
        sum_im_cpu = payload["sum_im_cpu"]
        Q_c_cpu = payload["Q_c_cpu"]

        seg_starts_cpu = payload["seg_starts_cpu"]
        seg_ends_cpu = payload["seg_ends_cpu"]

        surv_weights_cpu = payload["surv_weights_cpu"]
        surv_pos_cpu = payload["surv_pos_cpu"]
        surv_is_aligned_cpu = payload["surv_is_aligned_cpu"]

        surv_parts_cpu = payload["surv_parts_cpu"]
        surv_r1_cpu = payload["surv_r1_cpu"]
        surv_r2_cpu = payload["surv_r2_cpu"]
        surv_normals_cpu = payload["surv_normals_cpu"]

        results = []

        for local_idx, c in enumerate(valid_clusters_cpu):
            s0 = seg_starts_cpu[local_idx]
            s1 = seg_ends_cpu[local_idx]
            if s1 <= s0: continue

            p1 = surv_parts_cpu[0][s0]
            if bounce_type == 1:
                target_parts = [p1];
                p2, p3 = -1, -1
            elif bounce_type == 2:
                p2 = surv_parts_cpu[1][s0];
                p3 = -1
                target_parts = [p1, p2]
            else:
                p2 = surv_parts_cpu[1][s0];
                p3 = surv_parts_cpu[2][s0]
                target_parts = [p1, p2, p3]

            t_list = []
            for pt in target_parts:
                pt_r1_list, pt_r2_list, pt_w_list, pt_normals_list = [], [], [], []
                for b in range(bounce_type):
                    b_parts = surv_parts_cpu[b][s0:s1]
                    mask_pt = (b_parts == pt)
                    if np.any(mask_pt):
                        pt_r1_list.append(surv_r1_cpu[b][s0:s1][mask_pt])
                        pt_r2_list.append(surv_r2_cpu[b][s0:s1][mask_pt])
                        pt_w_list.append(surv_weights_cpu[s0:s1][mask_pt])
                        pt_normals_list.append(surv_normals_cpu[b][s0:s1][mask_pt])

                if pt_w_list:
                    all_r1 = np.concatenate(pt_r1_list, axis=0)
                    all_r2 = np.concatenate(pt_r2_list, axis=0)
                    all_w = np.concatenate(pt_w_list, axis=0)
                    all_n = np.concatenate(pt_normals_list, axis=0)
                    num_rays = len(all_w)

                    if bounce_type in [1, 2]:
                        if num_rays < 3:
                            t_list.append(0)
                        else:
                            n_mean = np.mean(all_n, axis=0)
                            n_centered = all_n - n_mean
                            cov = np.dot(n_centered.T, n_centered) / num_rays
                            evals = np.linalg.eigvalsh(cov)
                            var_mid, var_max = evals[1], evals[2]
                            if var_max < 1e-3:
                                t_list.append(0)
                            elif var_mid / (var_max + 1e-12) < 0.3 or var_mid < 1e-3:
                                t_list.append(1)
                            else:
                                t_list.append(2)
                    else:
                        k_top = min(10, num_rays)
                        top_idx = np.argpartition(all_w, -k_top)[-k_top:] if k_top < num_rays else np.arange(num_rays)
                        top_r1, top_r2 = all_r1[top_idx], all_r2[top_idx]
                        votes = [0 if r1 > flat_threshold else (1 if r2 > flat_threshold else 2) for r1, r2 in
                                 zip(top_r1, top_r2)]
                        t_list.append(int(np.argmax(np.bincount(votes))))
                else:
                    t_list.append(0)

            alpha, mech, is_distributed = 1.0, "未知", False
            if bounce_type == 1:
                if t_list[0] == 0:
                    alpha, mech, is_distributed = 1.0, "单平面", True
                elif t_list[0] == 1:
                    alpha, mech, is_distributed = 0.5, "单圆柱/曲面", True
                else:
                    alpha, mech, is_distributed = 0.0, "单局部点/球", False
            elif bounce_type == 2:
                if t_list == [0, 0]:
                    alpha, mech, is_distributed = 1.0, "二面角", True
                elif 0 in t_list and 1 in t_list:
                    alpha, mech, is_distributed = 0.5, "待定平柱/顶帽", True
                elif t_list == [1, 1]:
                    alpha, mech, is_distributed = 0.0, "圆柱-圆柱", True
                else:
                    alpha, mech, is_distributed = 0.0, "二次局部耦合", False
            elif bounce_type == 3:
                if t_list.count(0) == 3:
                    alpha, mech, is_distributed = 1.0, "三面角", True
                else:
                    alpha, mech, is_distributed = 0.0, "复杂多次耦合", False

            length = 0.0
            aligned_local = surv_is_aligned_cpu[s0:s1]
            local_weights = surv_weights_cpu[s0:s1]
            strong_local = local_weights >= (np.max(local_weights) * 0.1) if len(local_weights) > 0 else np.zeros_like(
                aligned_local, dtype=bool)

            valid_length_rays = aligned_local & strong_local
            num_aligned = np.sum(valid_length_rays)

            if is_distributed and num_aligned >= 4:
                pts_aligned = surv_pos_cpu[s0:s1][valid_length_rays]
                cov_pos = np.dot((pts_aligned - np.mean(pts_aligned, axis=0)).T,
                                 (pts_aligned - np.mean(pts_aligned, axis=0))) / num_aligned
                e_vals_pos, e_vecs_pos = np.linalg.eigh(cov_pos)
                proj = np.dot(pts_aligned, e_vecs_pos[:, 2])
                proj_span = np.max(proj) - np.min(proj)

                if mech == "待定平柱/顶帽":
                    if e_vals_pos[2] > 1e-4 and (
                            e_vals_pos[2] / (e_vals_pos[1] + 1e-12) > 4.0) and proj_span >= 2.0 * res_azimuth:
                        mech = "平柱耦合";
                        length = float(proj_span)
                    else:
                        mech = "顶帽结构";
                        is_distributed = False;
                        length = 0.0
                else:
                    if proj_span >= 2.0 * res_azimuth: length = float(proj_span)
            else:
                if mech == "待定平柱/顶帽": mech = "顶帽结构"; is_distributed = False; length = 0.0

            if num_aligned < 3 and is_distributed: mech += "(非对齐或射线太少)"

            A_intrinsic = sum_re_cpu[local_idx] + 1j * sum_im_cpu[local_idx]
            pos_arr = Q_c_cpu[local_idx]

            results.append({
                'P1': int(p1), 'P2': int(p2), 'P3': int(p3),
                'rcs_dbsm': 10 * math.log10(np.abs(A_intrinsic) ** 2 + 1e-12),
                'rcs_re': float(A_intrinsic.real),
                'rcs_im': float(A_intrinsic.imag),
                'alpha': float(alpha),
                'length': float(length),
                'pos_arr': pos_arr,
                'pos': pos_arr,
                'mech': mech
            })

        return results

    def _fast_aggregate_and_extract(self, valid_mask, bounce_type, Q_equiv, complex_E, f_lists, v_out, v_i,
                                    flat_threshold, k, res_azimuth=0.1):
        payload = self._prepare_cluster_payload_gpu(valid_mask, bounce_type, Q_equiv, complex_E, f_lists, v_out, v_i,
                                                    flat_threshold)
        return self._finalize_clusters_cpu(payload, flat_threshold, k, res_azimuth)

    def run_pipeline(self, el_deg, phi_deg, freq_hz, pt_stream=None, wp_stream=None):
        k = 2.0 * math.pi * freq_hz / 3e8
        wavelength = 3e8 / freq_hz
        flat_threshold = 1000.0 * wavelength

        th, ph = math.radians(90.0 - el_deg), math.radians(phi_deg)
        sp, cp = math.sin(ph), math.cos(ph)

        v_i = torch.tensor([-math.sin(th) * cp, -math.sin(th) * sp, -math.cos(th)], device=self.device_str)
        E_i = torch.tensor([-sp, cp, 0.0], device=self.device_str)

        v_i_exp = v_i.unsqueeze(0).expand(self.num_faces, 3).contiguous()
        v_s_glob = (-v_i).unsqueeze(0).contiguous()
        E_i_exp = E_i.unsqueeze(0).expand(self.num_faces, 3).contiguous()

        self.cE_1_buf.zero_();
        self.cE_2_buf.zero_();
        self.cE_3_buf.zero_()
        self.Q_eq2_buf.zero_();
        self.Q_eq3_buf.zero_()
        self.E_r1_buf.zero_();
        self.E_r2_buf.zero_()
        self.E_in2_buf.zero_();
        self.E_in3_buf.zero_()
        self.v_r1_buf.zero_();
        self.v_r2_buf.zero_();
        self.v_r3_buf.zero_()

        ev_rt1_done = torch.cuda.Event();
        ev_rt2_done = torch.cuda.Event();
        ev_rt3_done = torch.cuda.Event()

        # ---- [Stage 1: RT1] ----
        front = torch.sum(self.normals * v_i, dim=-1) < 0
        idx0 = front.nonzero(as_tuple=True)[0]

        f1_ans = torch.full((self.num_faces,), self.num_faces, dtype=torch.long, device=self.device_str)
        if idx0.shape[0] > 0:
            ans1 = self.scene.cast_rays(self.centers[idx0] + self.normals[idx0] * 1e-4,
                                        -v_i_exp[:idx0.shape[0]].contiguous(), wp_stream)
            f1_ans[idx0] = ans1['primitive_ids']

        f1 = torch.arange(self.num_faces, device=self.device_str)
        valid1 = front & (f1_ans == self.num_faces)
        idx1 = torch.where(valid1)[0]

        if idx1.shape[0] > 0:
            self.v_r1_buf[idx1] = _compute_go_dir_jit(v_i_exp[:idx1.shape[0]], self.normals[idx1])
            self.E_r1_buf[idx1] = _compute_go_field_jit(self.normals[idx1], E_i_exp[:idx1.shape[0]])

        ev_rt1_done.record(pt_stream)
        self.s_po1.wait_event(ev_rt1_done)

        if idx1.shape[0] > 0:
            with torch.cuda.stream(self.s_po1), wp.ScopedStream(self.wps_po1):
                phase_center_1 = -2.0 * k * torch.sum(v_i * self.centers[idx1], dim=-1)
                self.cE_1_buf[idx1] = self._compute_po_field(
                    v_i_exp[:idx1.shape[0]], v_s_glob, self.normals[idx1], self.areas[idx1], self.centers[idx1],
                    self.v0[idx1], self.v1[idx1], self.v2[idx1], E_i_exp[:idx1.shape[0]], E_i.unsqueeze(0), k,
                    phase_center_1, wp_stream=self.wps_po1
                )

        # ---- [Stage 2: RT2] ----
        scale_epsilon = torch.clamp(torch.norm(self.centers, dim=-1, keepdim=True) * 1e-6, min=1e-5)
        ray2_orig = (self.centers + self.v_r1_buf * scale_epsilon + self.normals * (scale_epsilon * 0.1)).contiguous()

        f2 = torch.full((self.num_faces,), self.num_faces, dtype=torch.long, device=self.device_str)
        t2 = torch.full((self.num_faces,), float(1e8), dtype=torch.float32, device=self.device_str)

        if idx1.shape[0] > 0:
            ans2 = self.scene.cast_rays(ray2_orig[idx1], self.v_r1_buf[idx1].contiguous(), wp_stream)
            f2[idx1] = ans2['primitive_ids']
            t2[idx1] = ans2['t_hit']

        valid2 = valid1 & (f2 < self.num_faces) & (t2 > 1e-6) & (f2 != f1)
        f2_s = torch.where(valid2, f2, torch.zeros_like(f2))
        valid2 &= (torch.sum(self.normals[f2_s] * self.v_r1_buf, dim=-1) < 0)

        sh2_f = torch.full((self.num_faces,), self.num_faces, dtype=torch.long, device=self.device_str)
        idx2_sh = valid2.nonzero(as_tuple=True)[0]
        if idx2_sh.shape[0] > 0:
            ans2_sh = self.scene.cast_rays(self.centers[f2_s[idx2_sh]] + self.normals[f2_s[idx2_sh]] * 1e-4,
                                           -v_i_exp[:idx2_sh.shape[0]].contiguous(), wp_stream)
            sh2_f[idx2_sh] = ans2_sh['primitive_ids']
        valid2 &= (sh2_f == self.num_faces)
        idx2 = torch.where(valid2)[0]

        if idx2.shape[0] > 0:
            self.v_r2_buf[idx2] = _compute_go_dir_jit(self.v_r1_buf[idx2], self.normals[f2_s[idx2]])
            d_1to2_v = torch.norm(self.centers[f2_s[idx2]] - self.centers[idx2], dim=-1)
            cos_theta_i1 = torch.clamp(torch.abs(torch.sum(self.normals[idx2] * v_i, dim=-1)), min=1e-3)
            rho1_1 = self.face_r1[idx2] * cos_theta_i1 / 2.0
            rho2_1 = self.face_r2[idx2] / (2.0 * cos_theta_i1)
            t1 = torch.where(self.face_r1[idx2] > flat_threshold, torch.zeros_like(self.face_r1[idx2]),
                             d_1to2_v / (torch.abs(rho1_1) + 1e-12))
            t_r2 = torch.where(self.face_r2[idx2] > flat_threshold, torch.zeros_like(self.face_r2[idx2]),
                               d_1to2_v / (torch.abs(rho2_1) + 1e-12))
            DF2_v = torch.clamp(1.0 / torch.sqrt(torch.abs((1.0 + t1) * (1.0 + t_r2)) + 1e-12), 0.0, 1.0)
            self.E_in2_buf[idx2] = self.E_r1_buf[idx2] * DF2_v.unsqueeze(-1)
            self.E_r2_buf[idx2] = _compute_go_field_jit(self.normals[f2_s[idx2]], self.E_in2_buf[idx2])

        ev_rt2_done.record(pt_stream)
        self.s_po2.wait_event(ev_rt2_done)

        if idx2.shape[0] > 0:
            with torch.cuda.stream(self.s_po2), wp.ScopedStream(self.wps_po2):
                f2_v = f2_s[idx2];
                Q1_v = self.centers[idx2];
                Q2_v = self.centers[f2_v]
                n1_v = self.normals[idx2];
                n2_v = self.normals[f2_v];
                vr1_v = self.v_r1_buf[idx2]

                d1 = torch.sum(Q1_v * n1_v, dim=-1, keepdim=True)
                d2 = torch.sum(Q2_v * n2_v, dim=-1, keepdim=True)
                n3_v = torch.nn.functional.normalize(torch.cross(v_i_exp[idx2], vr1_v, dim=-1), dim=-1)
                d3 = torch.sum(Q1_v * n3_v, dim=-1, keepdim=True)

                c23 = torch.cross(n2_v, n3_v, dim=-1)
                c31 = torch.cross(n3_v, n1_v, dim=-1)
                c12 = torch.cross(n1_v, n2_v, dim=-1)
                det = torch.sum(n1_v * c23, dim=-1, keepdim=True)

                Q_prime_2 = (d1 * c23 + d2 * c31 + d3 * c12) / (det + 1e-12)
                Q_prime_2 = torch.where((torch.abs(det) < 1e-5).squeeze(-1).unsqueeze(-1), (Q1_v + Q2_v) / 2.0,
                                        Q_prime_2)

                d2_v = 0.5 * (torch.sum(v_i * Q1_v, dim=-1) + torch.norm(Q2_v - Q1_v, dim=-1) + torch.sum(v_i * Q2_v,
                                                                                                          dim=-1))
                self.Q_eq2_buf[idx2] = Q_prime_2 - torch.sum(Q_prime_2 * v_i, dim=-1,
                                                             keepdim=True) * v_i + d2_v.unsqueeze(-1) * v_i

                dot_n_vr1_v = torch.clamp(torch.abs(torch.sum(self.normals[f2_v] * self.v_r1_buf[idx2], dim=-1)),
                                          min=1e-2)
                dS2_v = (self.areas[idx2] * torch.abs(torch.sum(self.normals[idx2] * v_i, dim=-1))) / dot_n_vr1_v

                self.cE_2_buf[idx2] = self._compute_po_field(
                    self.v_r1_buf[idx2], v_s_glob, self.normals[f2_v], dS2_v, self.centers[f2_v],
                    self.v0[f2_v], self.v1[f2_v], self.v2[f2_v], self.E_in2_buf[idx2], E_i.unsqueeze(0), k,
                    -2.0 * k * d2_v, wp_stream=self.wps_po2
                )

        # ---- [Stage 3: RT3] ----
        scale_eps3 = torch.clamp(torch.norm(self.centers[f2_s], dim=-1, keepdim=True) * 1e-6, min=1e-5)
        ray3_orig = (self.centers[f2_s] + self.v_r2_buf * scale_eps3 + self.normals[f2_s] * (
                scale_eps3 * 0.1)).contiguous()

        f3 = torch.full((self.num_faces,), self.num_faces, dtype=torch.long, device=self.device_str)
        t3 = torch.full((self.num_faces,), float(1e8), dtype=torch.float32, device=self.device_str)

        if idx2.shape[0] > 0:
            ans3 = self.scene.cast_rays(ray3_orig[idx2], self.v_r2_buf[idx2].contiguous(), wp_stream)
            f3[idx2] = ans3['primitive_ids']
            t3[idx2] = ans3['t_hit']

        valid3 = valid2 & (f3 < self.num_faces) & (t3 > 1e-6) & (f3 != f2_s)
        f3_s = torch.where(valid3, f3, torch.zeros_like(f3))
        valid3 &= (torch.sum(self.normals[f3_s] * self.v_r2_buf, dim=-1) < 0)

        sh3_f = torch.full((self.num_faces,), self.num_faces, dtype=torch.long, device=self.device_str)
        idx3_sh = valid3.nonzero(as_tuple=True)[0]
        if idx3_sh.shape[0] > 0:
            ans3_sh = self.scene.cast_rays(self.centers[f3_s[idx3_sh]] + self.normals[f3_s[idx3_sh]] * 1e-4,
                                           -v_i_exp[:idx3_sh.shape[0]].contiguous(), wp_stream)
            sh3_f[idx3_sh] = ans3_sh['primitive_ids']
        valid3 &= (sh3_f == self.num_faces)
        idx3 = torch.where(valid3)[0]

        if idx3.shape[0] > 0:
            self.v_r3_buf[idx3] = _compute_go_dir_jit(self.v_r2_buf[idx3], self.normals[f3_s[idx3]])
            d_2to3_v = torch.norm(self.centers[f3_s[idx3]] - self.centers[f2_s[idx3]], dim=-1)
            cos_theta_i2 = torch.clamp(torch.abs(torch.sum(self.normals[f2_s[idx3]] * self.v_r1_buf[idx3], dim=-1)),
                                       min=1e-3)
            rho1_2 = self.face_r1[f2_s[idx3]] * cos_theta_i2 / 2.0
            rho2_2 = self.face_r2[f2_s[idx3]] / (2.0 * cos_theta_i2)
            t1_3 = torch.where(self.face_r1[f2_s[idx3]] > flat_threshold, torch.zeros_like(self.face_r1[f2_s[idx3]]),
                               d_2to3_v / (torch.abs(rho1_2) + 1e-12))
            t_r2_3 = torch.where(self.face_r2[f2_s[idx3]] > flat_threshold, torch.zeros_like(self.face_r2[f2_s[idx3]]),
                                 d_2to3_v / (torch.abs(rho2_2) + 1e-12))
            DF3_v = torch.clamp(1.0 / torch.sqrt(torch.abs((1.0 + t1_3) * (1.0 + t_r2_3)) + 1e-12), 0.0, 1.0)
            self.E_in3_buf[idx3] = self.E_r2_buf[idx3] * DF3_v.unsqueeze(-1)

        ev_rt3_done.record(pt_stream)
        self.s_po3.wait_event(ev_rt3_done)

        if idx3.shape[0] > 0:
            with torch.cuda.stream(self.s_po3), wp.ScopedStream(self.wps_po3):
                f2_v = f2_s[idx3];
                f3_v = f3_s[idx3];
                Q1_v = self.centers[idx3];
                Q2_v = self.centers[f2_v];
                Q3_v = self.centers[f3_v]
                n_eq2_v = torch.nn.functional.normalize(self.v_r2_buf[idx3] - v_i_exp[idx3], dim=-1)

                d1_p = torch.sum(self.Q_eq2_buf[idx3] * n_eq2_v, dim=-1, keepdim=True)
                d2_p = torch.sum(Q3_v * self.normals[f3_v], dim=-1, keepdim=True)
                n3_p = torch.nn.functional.normalize(torch.cross(v_i_exp[idx3], self.v_r2_buf[idx3], dim=-1), dim=-1)
                d3_p = torch.sum(self.Q_eq2_buf[idx3] * n3_p, dim=-1, keepdim=True)

                c23_p = torch.cross(self.normals[f3_v], n3_p, dim=-1)
                c31_p = torch.cross(n3_p, n_eq2_v, dim=-1)
                c12_p = torch.cross(n_eq2_v, self.normals[f3_v], dim=-1)
                det_p = torch.sum(n_eq2_v * c23_p, dim=-1, keepdim=True)

                Q_prime_3 = (d1_p * c23_p + d2_p * c31_p + d3_p * c12_p) / (det_p + 1e-12)
                Q_prime_3 = torch.where((torch.abs(det_p) < 1e-5).squeeze(-1).unsqueeze(-1), (Q1_v + Q2_v + Q3_v) / 3.0,
                                        Q_prime_3)

                d3_v = 0.5 * (torch.sum(v_i * Q1_v, dim=-1) + torch.norm(Q2_v - Q1_v, dim=-1) + torch.norm(Q3_v - Q2_v,
                                                                                                           dim=-1) + torch.sum(
                    v_i * Q3_v, dim=-1))
                self.Q_eq3_buf[idx3] = Q_prime_3 - torch.sum(Q_prime_3 * v_i, dim=-1,
                                                             keepdim=True) * v_i + d3_v.unsqueeze(-1) * v_i

                dot_n_vr2_v = torch.clamp(torch.abs(torch.sum(self.normals[f3_v] * self.v_r2_buf[idx3], dim=-1)),
                                          min=1e-2)
                dS3_v = (self.areas[idx3] * torch.abs(torch.sum(self.normals[idx3] * v_i, dim=-1))) / dot_n_vr2_v

                self.cE_3_buf[idx3] = self._compute_po_field(
                    self.v_r2_buf[idx3], v_s_glob, self.normals[f3_v], dS3_v, self.centers[f3_v],
                    self.v0[f3_v], self.v1[f3_v], self.v2[f3_v], self.E_in3_buf[idx3], E_i.unsqueeze(0), k,
                    -2.0 * k * d3_v, wp_stream=self.wps_po3
                )

        pt_stream.wait_stream(self.s_po1)
        pt_stream.wait_stream(self.s_po2)
        pt_stream.wait_stream(self.s_po3)

        res1 = self._fast_aggregate_and_extract(valid1, 1, self.centers, self.cE_1_buf, [f1], self.v_r1_buf, v_i,
                                                flat_threshold, k, res_azimuth=0.1)
        res2 = self._fast_aggregate_and_extract(valid2, 2, self.Q_eq2_buf, self.cE_2_buf, [f1, f2_s], self.v_r2_buf,
                                                v_i, flat_threshold, k, res_azimuth=0.1)
        res3 = self._fast_aggregate_and_extract(valid3, 3, self.Q_eq3_buf, self.cE_3_buf, [f1, f2_s, f3_s],
                                                self.v_r3_buf, v_i, flat_threshold, k, res_azimuth=0.1)

        all_res = res1 + res2 + res3
        if not all_res: return []

        global_max_amp = max(math.hypot(r['rcs_re'], r['rcs_im']) for r in all_res)
        global_threshold = global_max_amp * 0.003
        return [r for r in all_res if math.hypot(r['rcs_re'], r['rcs_im']) >= global_threshold]


# ================= 🚀 专门为扩散模型构建的纯净 SAR 图像无损保存线程 =================
def saving_daemon(save_queue):
    """
    利用 PIL 库极速保存符合扩散模型输入规范的纯净 64x64 单通道灰度图像。
    """
    while True:
        task = save_queue.get()
        if task is None:
            break

        img_raw, file_path = task

        # 扩散模型需要的物理转换：使用相对 dB 值后线性映射
        img_db = 20 * np.log10(img_raw + 1e-15)
        max_db = np.max(img_db)
        min_db = max_db - 40.0
        img_db_clipped = np.clip(img_db, min_db, max_db)

        # 归一化到 0.0 ~ 1.0 区间
        img_norm = (img_db_clipped - min_db) / 40.0

        # 转换至 8 位
        img_uint8 = (img_norm * 255).astype(np.uint8)

        # 【重点修改】取消之前的翻转/转置机制，直接将原矩阵顺时针旋转90度
        # np.rot90 的 k=-1 表示顺时针旋转 90 度，.copy() 防止内存步长(stride)非连续导致 PIL 报错
      #  img_uint8 = np.rot90(img_uint8, k=-1).copy()

        # 保存为无通道冗余的纯正灰度图 ('L' mode)
        img_pil = Image.fromarray(img_uint8, mode='L')
        img_pil.save(file_path)

        save_queue.task_done()


# ================= 测试运行模块 =================
def worker_func(task_idx, task, sim, sar_sim, device, save_queue, output_dir, model_name):
    el_deg, p_deg, freq = task['el'], task['phi'], task['freq']
    device_str = "cuda:0" if "cuda" in str(device) else "cpu"
    print(f"  >>>> 任务 {task_idx + 1} 开始计算 (El={el_deg:.2f}°, Az={p_deg:.2f}°) ...")
    start_all = time.time()

    pt_stream = torch.cuda.Stream(device=device_str)
    wp_stream = wp.stream_from_torch(pt_stream)

    with torch.cuda.stream(pt_stream), wp.ScopedStream(wp_stream):
        # 1. 运行核心光追引擎
        results = sim.run_pipeline(el_deg, p_deg, freq, pt_stream=pt_stream, wp_stream=wp_stream)

        # 2. 生成纯净 64x64 图像矩阵
        fig_name = None
        if results:
            img_matrix = sar_sim.generate_image_for_diffusion(results, el_deg, p_deg, wp_stream=wp_stream)

            # 文件命名：浮点数保留两位小数
            fig_name = os.path.join(output_dir, f'{model_name}_El{el_deg:.2f}_Az{p_deg:.2f}.png')
            save_queue.put((img_matrix.copy(), fig_name))

    return task_idx, el_deg, p_deg, freq, results, fig_name, time.time() - start_all


# 【极致优化】并行任务封装器，打通资源池
def parallel_task_wrapper(task_bundle):
    task_idx, task, resource_queue, save_queue, device, output_dir, model_name = task_bundle
    sim, sar_sim = resource_queue.get()
    try:
        return worker_func(task_idx, task, sim, sar_sim, device, save_queue, output_dir, model_name)
    finally:
        resource_queue.put((sim, sar_sim))


# ================= 主程序 =================
if __name__ == "__main__":
    mesh_path = "mesh/2s1.mesh"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if os.path.exists(mesh_path):
        # 自动提取出网格模型的名称
        model_name = os.path.splitext(os.path.basename(mesh_path))[0]

        # 初始化解析器
        v_t, f_t, p_ids_t, f_r1_t, f_r2_t, fn_t = GeometryPreprocessor.process_mesh(mesh_path, device=device)

        output_dir = "data/sim/2s1"
        os.makedirs(output_dir, exist_ok=True)

        num_workers = 3
        print(f"\n[*] 正在为 GPU 初始化 {num_workers} 个完全隔离的并发流水线资源池...")

        resource_queue = queue.Queue()
        for _ in range(num_workers):
            sim = StreamlinedRayTracingEngine(v_t, f_t, p_ids_t, f_r1_t, f_r2_t, fn_t, device=device)
            sar_sim = PyTorchSARSimulator(fc=9.6e9, resolution=0.202, image_grid_size=64, range_scope=16.0,
                                          device=device)
            resource_queue.put((sim, sar_sim))

        save_queue = queue.Queue()
        save_thread = threading.Thread(target=saving_daemon, args=(save_queue,), daemon=True)
        save_thread.start()

        # ================= 动态姿态提取 =================
        print("\n[*] 正在从数据文件中自动提取姿态角组合...")
        FREQ_HZ = 96e8
        npy_path = "dataset_real_test1/scattering_centers_2s1_2d.npy"
        poses = []

        if os.path.exists(npy_path):
            data = np.load(npy_path, allow_pickle=True)
            if data.ndim == 0:
                data = data.item()
            pose_set = set()
            for sc_obj in data:
                if isinstance(sc_obj, dict):
                    arr = np.array(sc_obj.get('points', []), dtype=np.float32)
                else:
                    arr = np.array(sc_obj, dtype=np.float32)
                if arr.shape[0] > 0 and arr.shape[1] >= 5:
                    # 获取姿态角并去重
                    az = float(arr[0, 3])
                    el = float(arr[0, 4])
                    pose_set.add((el, az))
            poses = list(pose_set)
            poses.sort()
            print(f"[*] 成功从 {npy_path} 提取出 {len(poses)} 个唯一的 (俯仰角, 方位角) 姿态组合。")
        else:
            print(f"[*] ⚠️ 未找到 {npy_path}，执行默认演示。")
            poses = [(15.0, 0.0), (45.0, 90.0)]

        tasks = [{'el': el, 'phi': az, 'freq': FREQ_HZ} for el, az in poses]

        print(f"[*] 当前处理模型: {model_name.upper()}")
        print(f"[*] 共生成 {len(tasks)} 个并发任务。")
        print("[*] 启动全异步无阻塞资源池调度...\n")

        task_bundles = [
            (i, task, resource_queue, save_queue, device, output_dir, model_name)
            for i, task in enumerate(tasks)
        ]

        total_start_time = time.time()
        completed_tasks = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            results_iter = executor.map(parallel_task_wrapper, task_bundles)

            for result in results_iter:
                idx, el, ph, freq, results, img_path, elapsed = result
                completed_tasks += 1

        save_queue.put(None)
        save_thread.join()
        total_end_time = time.time()

        print(f"\n================ 运行总结 ================")
        print(f"[*] 目标模型 {model_name.upper()} 的专属纯净图像生成完毕！")
        print(f"[*] 共处理了 {completed_tasks} 个姿态组合。")
        print(f"[*] 总计耗时: {total_end_time - total_start_time:.4f} 秒。")
        print(f"[*] 扩散网络专用的纯净 64x64 图像保存在: {os.path.abspath(output_dir)}")
        print(f"==========================================\n")
    else:
        print(f"未找到网格模型：{mesh_path}")