import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import random

# --- 1. KHAI BÁO ĐƯỜNG DẪN ---
# Cập nhật lại đường dẫn tới các subfolder chứa file gradient của bạn
folder_sft_grads = "training/sft_VOC2007_fastvlm_contrative_signal/step_grads"
folder_gvendi_grads = "training/gvendi_VOC2007_fastvlm_contrative_signal/step_grads"
folder_teacher_grads = "training/gvendi_VOC2007_fastvlm_contrative_signal/teacher_step_grads"

# --- 2. CHỈ ĐỊNH STEP MUỐN VẼ ---
target_step = random.randint(800, 950)

def get_accumulated_spectrum(folder, step, filename_pattern="step_{}_grads.pt"):
    """Load tensor, tính Gram matrix, và trả về Accumulated Spectrum cho 1 step."""
    path = os.path.join(folder, filename_pattern.format(step))
    if not os.path.exists(path):
        print(f"⚠️ Cảnh báo: Không tìm thấy file tại {path}")
        return None
    
    # Load tensor shape (b, d)
    X = torch.load(path, map_location='cpu') 
    
    if X.dtype != torch.float32 and X.dtype != torch.float64:
        X = X.to(torch.float32)

    # Tính Gram matrix: G = X @ X^T -> shape (b, b)
    G = torch.matmul(X, X.T)
    
    # Tính Eigenvalues
    eigenvalues, _ = torch.linalg.eigh(G)
    
    # Lật ngược để sắp xếp giảm dần và bỏ số âm do sai số
    eigenvalues = torch.flip(eigenvalues, dims=[0])
    eigenvalues = torch.relu(eigenvalues) 
    
    # Tính Cumulative sum và chuẩn hóa về [0, 1]
    cum_spectrum = torch.cumsum(eigenvalues, dim=0)
    if cum_spectrum[-1] > 0:
        cum_spectrum = cum_spectrum / cum_spectrum[-1]
        
    return cum_spectrum.numpy()

# --- 3. TÍNH TOÁN CHO TARGET STEP ---
# Lưu ý kiểm tra xem file của bạn tên là "step_1000_grads.pt" hay "step_1000_grad.pt"
filename_pattern = "step_{}_grads.pt" 

print(f"Đang xử lý dữ liệu cho step {target_step}...")

spec_sft = get_accumulated_spectrum(folder_sft_grads, target_step, filename_pattern)
spec_gvendi = get_accumulated_spectrum(folder_gvendi_grads, target_step, filename_pattern)
spec_teacher = get_accumulated_spectrum(folder_teacher_grads, target_step, filename_pattern)

# --- 4. VISUALIZE ---
# Kiểm tra xem có lấy được dữ liệu nào không
all_specs = [s for s in [spec_sft, spec_gvendi, spec_teacher] if s is not None]

if len(all_specs) == 0:
    print(f"❌ Không tìm thấy dữ liệu ở step {target_step} trong cả 3 folder.")
else:
    plt.figure(figsize=(7, 6))

    # Tìm trục x lớn nhất (b lớn nhất) đề phòng trường hợp batch size các model khác nhau
    max_b = max([len(s) for s in all_specs])

    # Vẽ từng đường nếu có dữ liệu
    if spec_sft is not None:
        plt.plot(np.arange(1, len(spec_sft) + 1), spec_sft, label='SFT', color='#1f77b4', linewidth=2.5)
    if spec_gvendi is not None:
        plt.plot(np.arange(1, len(spec_gvendi) + 1), spec_gvendi, label='ReGKD', color='#ff7f0e', linewidth=2.5)
    if spec_teacher is not None:
        plt.plot(np.arange(1, len(spec_teacher) + 1), spec_teacher, label='Teacher', color='#2ca02c', linewidth=2.5, linestyle='--')

    # Trang trí biểu đồ
    plt.xlabel("Eigenvalue Index", fontsize=15)
    plt.ylabel("Cumulative Variance Ratio", fontsize=15)
    # plt.title(f"Accumulated Spectrum of Gram Matrix at Step {target_step}", fontsize=14)
    plt.legend(fontsize=15)
    plt.grid(True, linestyle='--', alpha=0.6)

    # Giới hạn trục hiển thị
    plt.xlim(1, max_b)
    plt.ylim(0, 1.05)

    # Lưu biểu đồ
    out_file = f"spectrum_step_{target_step}.png"
    plt.savefig(out_file, dpi=600, bbox_inches="tight")
    plt.close()

    print(f"✅ Đã lưu biểu đồ tại: {out_file}")