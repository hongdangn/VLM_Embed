#!/usr/bin/env python3
"""Visualize common gradients between Student and Teacher models.

Usage example:
  python visualize_common_grads.py \
    --student-dir teacher_gradients/gvendi_fastvlm_phase1 \
    --teacher-dir teacher_gradients/qwen2b_vqa_grad \
    --method TSNE \
    --out-dir .

    python visualize_common_grads.py \
    --student-dir teacher_gradients/sft_fastvlm_vqa_phase1 \
    --teacher-dir teacher_gradients/qwen2b_vqa_grad \
    --method TSNE \
    --out-dir .
"""

import os
import argparse
import glob
from collections import defaultdict

try:
    import torch
except Exception:
    torch = None

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def load_grad_file(path):
    if path.endswith((".npy",)):
        return np.load(path, allow_pickle=True).item() if path.endswith(".npy") else np.load(path)

    if path.endswith((".pt", ".pth")):
        if torch is None:
            raise RuntimeError("PyTorch not available to load .pt/.pth files")
        obj = torch.load(path, map_location="cpu")
        if hasattr(obj, "numpy"):
            return obj.numpy()
        return obj

    try:
        return np.load(path, allow_pickle=True).item()
    except Exception:
        raise ValueError(f"Unsupported gradient file format: {path}")


def flatten_to_1d(x):
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x.reshape(-1).astype(np.float64)
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().reshape(-1).astype(np.float64)
    arr = np.asarray(x)
    return arr.reshape(-1).astype(np.float64)


def visualize_gradients_2d(G_s, G_t, param_name, method="PCA", out_dir="plots"):
    """
    Giảm chiều không gian và vẽ biểu đồ phân tán (Scatter plot) cho Student và Teacher.
    """
    n_student = G_s.shape[0]
    n_teacher = G_t.shape[0]
    total_samples = n_student + n_teacher
    
    if total_samples < 3:
        print(f"Skipping visualization for {param_name}: Not enough samples ({total_samples}).")
        return

    # Gom chung dữ liệu để fit vào cùng một không gian
    G_combined = np.vstack([G_s, G_t])
    
    if method == "PCA":
        reducer = PCA(n_components=2)
    elif method == "TSNE":
        perplexity = min(30, total_samples - 1) 
        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    else:
        raise ValueError("Method must be 'PCA' or 'TSNE'")

    try:
        G_2d = reducer.fit_transform(G_combined)
    except Exception as e:
        print(f"Failed to reduce dimension for {param_name}: {e}")
        return

    # Tách lại dữ liệu sau khi đã giảm xuống 2D
    s_2d = G_2d[:n_student, :]
    t_2d = G_2d[n_student:, :]

    # Vẽ biểu đồ
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(10, 8))
    
    plt.scatter(s_2d[:, 0], s_2d[:, 1], c='#1f77b4', alpha=0.7, label='Student', edgecolors='w', s=60)
    plt.scatter(t_2d[:, 0], t_2d[:, 1], c='#d62728', alpha=0.7, label='Teacher', edgecolors='w', s=60)
    
    plt.title(f"Gradient Projection ({method})\nParameter: {param_name} (Dim: {G_s.shape[1]})", fontsize=14)
    plt.xlabel(f"{method} Component 1")
    plt.ylabel(f"{method} Component 2")
    plt.legend(loc="best", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Tạo tên file an toàn (bỏ các ký tự đặc biệt trong tên param)
    safe_name = param_name.replace("/", "_").replace(".", "_")
    save_path = os.path.join(out_dir, f"{safe_name}_{method}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {save_path}")


def main(student_dir, teacher_dir, out_dir="plots", method="PCA", min_samples=2):
    student_files = sorted(glob.glob(os.path.join(student_dir, "*")))
    if not student_files:
        raise SystemExit(f"No files found in student_dir={student_dir}")

    # 1. Tìm các file chung giữa 2 thư mục
    pairs = []
    for s in student_files:
        name = os.path.basename(s)
        t = os.path.join(teacher_dir, name)
        if os.path.exists(t):
            pairs.append((s, t))
        else:
            print(f"Warning: no matching teacher file for {name}")

    if not pairs:
        raise SystemExit("No matching student/teacher filename pairs found")

    student_grads_by_key = defaultdict(list)
    teacher_grads_by_key = defaultdict(list)
    
    print("Loading and flattening gradients...")
    # 2. Đọc file và gom vector gradient theo từng param key
    for s_path, t_path in pairs:
        s_obj = load_grad_file(s_path)
        t_obj = load_grad_file(t_path)
        
        # Xử lý Student
        if isinstance(s_obj, dict):
            for k, v in s_obj.items():
                vec = flatten_to_1d(v)
                if vec is not None:
                    student_grads_by_key[k].append(vec)
        else:
            vec = flatten_to_1d(s_obj)
            if vec is not None:
                student_grads_by_key[os.path.basename(s_path)].append(vec)

        # Xử lý Teacher
        if isinstance(t_obj, dict):
            for k, v in t_obj.items():
                vec = flatten_to_1d(v)
                if vec is not None:
                    teacher_grads_by_key[k].append(vec)
        else:
            vec = flatten_to_1d(t_obj)
            if vec is not None:
                teacher_grads_by_key[os.path.basename(t_path)].append(vec)

    # 3. Lấy ra các gradient chung (Common Keys)
    common_keys = set(student_grads_by_key.keys()).intersection(teacher_grads_by_key.keys())
    print(f"Found {len(common_keys)} common gradient parameters. Starting visualization...\n")

    # 4. Trực quan hóa từng common key
    for key in sorted(common_keys):
        s_vecs = student_grads_by_key[key]
        t_vecs = teacher_grads_by_key[key]
        
        if len(s_vecs) < min_samples or len(t_vecs) < min_samples:
            print(f"Skipping {key}: Not enough samples (Student: {len(s_vecs)}, Teacher: {len(t_vecs)})")
            continue
            
        s_shapes = [v.shape[0] for v in s_vecs]
        t_shapes = [v.shape[0] for v in t_vecs]
        
        if len(set(s_shapes)) != 1 or len(set(t_shapes)) != 1 or s_shapes[0] != t_shapes[0]:
            print(f"Skipping {key}: Inconsistent vector dimensions.")
            continue
            
        G_s = np.stack(s_vecs, axis=0)  # (n_samples, dim)
        G_t = np.stack(t_vecs, axis=0)  # (n_samples, dim)
        
        # Vẽ biểu đồ
        visualize_gradients_2d(G_s, G_t, key, method=method, out_dir=out_dir)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Visualize common gradients between Student and Teacher")
    p.add_argument("--student-dir", required=True, help="Directory with student gradient files")
    p.add_argument("--teacher-dir", required=True, help="Directory with teacher gradient files")
    p.add_argument("--out-dir", default="grad_visualizations", help="Directory to save plots")
    p.add_argument("--method", choices=["PCA", "TSNE"], default="PCA", help="Dimensionality reduction method")
    p.add_argument("--min-samples", type=int, default=2, help="Minimum samples required to plot")
    args = p.parse_args()

    main(args.student_dir, args.teacher_dir, out_dir=args.out_dir, method=args.method, min_samples=args.min_samples)