import os
import torch
import numpy as np
from tqdm import tqdm

# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN
# ==========================================
BASE_GRAD_DIR = 'teacher_gradients'
GRAD_DIRS = {
    'gvendi': os.path.join(BASE_GRAD_DIR, 'gvendi_fastvlm_SUN397_phase1'),
    'sft': os.path.join(BASE_GRAD_DIR, 'sft_fastvlm_SUN397_phase1'),
    'teacher': os.path.join(BASE_GRAD_DIR, 'qwen2b_cls_grad'),
}

# Tự động nhận diện GPU để tăng tốc tính SVD (nếu có)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Đang sử dụng thiết bị: {device}")

def compute_erank_from_gram(batch_tensors, device=None):
    """
    Compute effective rank (stable rank) from Gram matrix.

    eRank(K) = (tr(K))^2 / ||K||_F^2

    Args:
        batch_tensors: Tensor [batch_size, ...]
        device: torch device (optional)

    Returns:
        float: effective rank
    """
    X = batch_tensors.view(batch_tensors.size(0), -1).float()
    
    if device is not None:
        X = X.to(device)

    # Gram matrix
    K = X @ X.T  # [N, N]

    # trace(K) = sum of eigenvalues
    trace = torch.trace(K)

    # Frobenius norm squared = sum of squared entries
    frob_sq = torch.sum(K * K)

    if frob_sq == 0:
        return 0.0

    # stable rank
    erank = (trace ** 2) / frob_sq

    return erank.item()

def evaluate_erank_batches(folder_path, file_names, batch_size=16):
    """
    Tải file, gom thành batch và tính chuỗi eRank cho một model
    """
    erank_results = []
    current_batch = []
    
    for file_name in tqdm(file_names, desc=f"Đang xử lý {os.path.basename(folder_path)}"):
        file_path = os.path.join(folder_path, file_name)
        
        # Tải tensor lên CPU trước để tránh đầy RAM GPU, chỉ chuyển lên GPU khi tính toán
        tensor_data = torch.load(file_path, map_location='cpu', weights_only=True)
        current_batch.append(tensor_data['grad_teacher'])  # Giả sử file chứa dict với key 'grad_teacher'
        
        # Đủ 1 batch thì tính eRank
        if len(current_batch) == batch_size:
            if batch_size==1:
                batch_tensor=current_batch
            batch_tensor = torch.stack(current_batch)
            erank = compute_erank_from_gram(batch_tensor)
            erank_results.append(erank)
            current_batch = [] # Reset lại batch
            
    # Xử lý phần dư (nếu tổng số file không chia hết cho batch_size)
    if len(current_batch) > 1: # SVD cần ít nhất ma trận 2x2 để có ý nghĩa so sánh
        batch_tensor = torch.stack(current_batch)
        erank = compute_erank_from_gram(batch_tensor)
        erank_results.append(erank)
        
    return erank_results

def main():
    # 1. Tìm sự giao nhau (intersection) của các file trong cả 3 thư mục
    # Điều này đảm bảo chúng ta đang so sánh táo với táo (cùng một gradient update)
    print("Đang tìm kiếm các file chung giữa 3 mô hình...")
    try:
        gvendi_files = set(os.listdir(GRAD_DIRS['gvendi']))
        sft_files = set(os.listdir(GRAD_DIRS['sft']))
        teacher_files = set(os.listdir(GRAD_DIRS['teacher']))
    except FileNotFoundError as e:
        print(f"Lỗi: Không tìm thấy thư mục. Chi tiết: {e}")
        return

    common_files = list(gvendi_files.intersection(sft_files).intersection(teacher_files))
    common_files.sort() # Sắp xếp để đảm bảo thứ tự luôn giống nhau
    
    num_files = len(common_files)
    print(f"Tìm thấy {num_files} file gradient tồn tại ở cả 3 mô hình.")
    
    if num_files == 0:
        print("Không có file nào chung để so sánh. Vui lòng kiểm tra lại dữ liệu.")
        return

    # Bạn có thể giới hạn số file chạy thử để test code nhanh: common_files = common_files[:320]
    batch_size = 64
    print(f"Bắt đầu tính toán với batch_size = {batch_size}...\n")

    # 2. Tính eRank cho từng model
    gvendi_eranks = evaluate_erank_batches(GRAD_DIRS['gvendi'], common_files, batch_size)
    sft_eranks = evaluate_erank_batches(GRAD_DIRS['sft'], common_files, batch_size)
    teacher_eranks = evaluate_erank_batches(GRAD_DIRS['teacher'], common_files, batch_size)

    # 3. In kết quả tổng quan
    print("\n" + "="*50)
    print(" KẾT QUẢ SO SÁNH EFFECTIVE RANK (eRank)")
    print("="*50)
    print(f"{'Mô hình':<15} | {'Trung bình':<10} | {'Độ lệch chuẩn (Std)':<20}")
    print("-" * 50)
    print(f"{'Teacher':<15} | {np.mean(teacher_eranks):<10.4f} | {np.std(teacher_eranks):<20.4f}")
    print(f"{'SFT':<15} | {np.mean(sft_eranks):<10.4f} | {np.std(sft_eranks):<20.4f}")
    print(f"{'Gvendi':<15} | {np.mean(gvendi_eranks):<10.4f} | {np.std(gvendi_eranks):<20.4f}")
    print("="*50)

    # (Tùy chọn) In 10 batch đầu tiên để xem chuỗi
    print("\n[Chi tiết 10 batch đầu tiên]")
    print(f"Teacher : {[round(x, 4) for x in teacher_eranks[:10]]}")
    print(f"SFT     : {[round(x, 4) for x in sft_eranks[:10]]}")
    print(f"Gvendi  : {[round(x, 4) for x in gvendi_eranks[:10]]}")

if __name__ == "__main__":
    main()