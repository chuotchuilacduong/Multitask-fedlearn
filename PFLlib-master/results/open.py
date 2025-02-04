import os
import h5py
import matplotlib.pyplot as plt

# Đường dẫn đến file .h5
fedavg_file = r"G:\BK\Quantum\test_fed\FedLearn-with-quantum\PFLlib-master\results\MNIST_FedAvg_test_0.h5"
fedper_file = r"G:\BK\Quantum\test_fed\FedLearn-with-quantum\PFLlib-master\results\MNIST_FedPer_test_0(1).h5"

# Hàm đọc dữ liệu từ file .h5
def read_h5_data(file_path):
    # Kiểm tra file có tồn tại hay không
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Mở file và đọc dữ liệu
    with h5py.File(file_path, "r") as h5_file:
        # Kiểm tra sự tồn tại của các dataset
        if "rs_test_acc" not in h5_file or "rs_train_loss" not in h5_file:
            raise KeyError(f"Missing required datasets in {file_path}")
        
        # Đọc dữ liệu
        rs_test_acc = h5_file["rs_test_acc"][:]
        rs_train_loss = h5_file["rs_train_loss"][:]
    
    return rs_test_acc, rs_train_loss

# Đọc dữ liệu từ hai file
try:
    fedavg_acc, fedavg_loss = read_h5_data(fedavg_file)
    fedper_acc, fedper_loss = read_h5_data(fedper_file)
except (FileNotFoundError, KeyError) as e:
    print(f"Error: {e}")
    exit()

# Số vòng (rounds)
rounds_fedavg = range(1, len(fedavg_acc) + 1)
rounds_fedper = range(1, len(fedper_acc) + 1)

# Vẽ biểu đồ so sánh
plt.figure(figsize=(15, 5))  # Điều chỉnh kích thước để phù hợp với bố cục ngang

# Biểu đồ 1: So sánh Test Accuracy
plt.subplot(1, 2, 1)
plt.plot(rounds_fedavg, fedavg_acc, marker='o', label='FedAvg Accuracy', color='blue')
plt.plot(rounds_fedper, fedper_acc, marker='s', label='FedPer Accuracy', color='green')
plt.xlabel("Rounds")
plt.ylabel("Accuracy")
plt.title("Comparison of Test Accuracy (FedAvg vs. FedPer)")
plt.grid(True)
plt.legend()

# Biểu đồ 2: So sánh Training Loss
plt.subplot(1, 2, 2)
plt.plot(rounds_fedavg, fedavg_loss, marker='o', label='FedAvg Loss', color='red')
plt.plot(rounds_fedper, fedper_loss, marker='s', label='FedPer Loss', color='orange')
plt.xlabel("Rounds")
plt.ylabel("Loss")
plt.title("Comparison of Training Loss (FedAvg vs. FedPer)")
plt.grid(True)
plt.legend()

# Hiển thị biểu đồ
plt.tight_layout()
plt.show()
