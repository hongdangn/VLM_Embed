import torch
import os
import re
import random
import numpy as np
import matplotlib.pyplot as plt

random.seed(40)  # Đặt seed để kết quả có thể tái lập được

# 1. Khai báo 3 folder (bạn điều chỉnh lại path cho chính xác nhé)
folder_sft = "training/sft_SUN397_fastvlm/step_eranks"
folder_gvendi = "training/gvendi_SUN397_fastvlm/step_align_eranks"
folder_teacher = "training/gvendi_SUN397_fastvlm/teacher_step_eranks"

def get_step(filename):
    match = re.search(r"step_(\d+)_erank\.pt", filename)
    return int(match.group(1)) if match else -1

def get_available_steps(folder):
    """Lấy danh sách các step có sẵn trong 1 folder"""
    if not os.path.exists(folder):
        return set()
    files = [f for f in os.listdir(folder) if f.endswith(".pt")]
    steps = [get_step(f) for f in files]
    return set(s for s in steps if s != -1)

def load_erank(folder, step):
    """Load giá trị eRank của một step cụ thể"""
    path = os.path.join(folder, f"step_{step}_erank.pt")
    if not os.path.exists(path):
        return None
    val = torch.load(path)
    return val.item() if isinstance(val, torch.Tensor) else val

# 2. Tìm các step chung giữa cả 3 folder để đảm bảo so sánh công bằng
steps_sft = get_available_steps(folder_sft)
steps_gvendi = get_available_steps(folder_gvendi)
steps_teacher = get_available_steps(folder_teacher)

common_steps = sorted(list(steps_sft & steps_gvendi & steps_teacher))

if not common_steps:
    raise ValueError("Không tìm thấy step nào chung giữa 3 folder!")

# 3. Lọc lấy các step ở giữa và cuối (ví dụ: bỏ qua 50% step đầu tiên)
mid_point = len(common_steps) // 2
# candidate_steps = common_steps[mid_point:]
candidate_steps = common_steps

# Random chọn ra N steps (ví dụ: 8 steps để lên biểu đồ không bị rối)
num_samples = min(50, len(candidate_steps))
# sampled_steps = sorted(random.sample(candidate_steps, num_samples))
sampled_steps = candidate_steps

print(f"Các steps được chọn để visualize: {sampled_steps}")

# 4. Load dữ liệu cho các steps đã chọn
vals_sft = [load_erank(folder_sft, s) for s in sampled_steps]
vals_gvendi = [load_erank(folder_gvendi, s) for s in sampled_steps]
vals_teacher = [load_erank(folder_teacher, s) for s in sampled_steps]

# 5. Visualize bằng Scatter Plot
x = np.arange(len(sampled_steps))  # Vị trí trên trục X
offset = 0.15                      # Dịch chuyển nhẹ để các điểm không bị đè lên nhau
marker_size = 100                  # Kích thước của điểm scatter (thay vì width=0.25)

plt.figure(figsize=(14, 7))

# Dùng scatter, truyền s=marker_size. 
# Có thể thêm alpha (độ trong suốt) để dễ nhìn nếu chúng hơi đè lên nhau.
plt.scatter(x - offset, vals_sft, s=marker_size, label='SFT', color='#1f77b4', alpha=0.8)
plt.scatter(x, vals_gvendi, s=marker_size, label='GVendi', color='#ff7f0e', alpha=0.8)
plt.scatter(x + offset, vals_teacher, s=marker_size, label='Teacher', color='#2ca02c', alpha=0.8)

# Trang trí biểu đồ
plt.xlabel("Training Step", fontsize=12)
plt.ylabel("eRank Value", fontsize=12)
plt.title("Effective Rank (eRank) Comparison at Middle/End Training Steps", fontsize=14)
plt.xticks(x, sampled_steps) # Gắn label cho trục x là số step thật
plt.legend(fontsize=12)

# Thêm grid cả trục y và x để dễ dóng thẳng hàng từ step lên
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.grid(axis='x', linestyle=':', alpha=0.4) 

# Lưu file
out_file = "erank_comparison_scatter.png"
plt.savefig(out_file, dpi=200, bbox_inches="tight")
plt.close()

print(f"Đã lưu biểu đồ tại: {out_file}")