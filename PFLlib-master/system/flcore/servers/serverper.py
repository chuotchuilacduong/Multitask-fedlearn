import random
import time
from flcore.clients.clientper import clientPer
from flcore.servers.serverbase import Server
from threading import Thread
import numpy as np

class FedPer(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.all_client_quantum_times = []  # Danh sách lưu thời gian quantum của tất cả các client
        self.set_slow_clients()
        self.set_clients(clientPer)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate personalized models")
                self.evaluate()

            client_round_times = []  # Lưu thời gian của từng client trong round này

            for client in self.selected_clients:
                start_time_client = time.time()  # Bắt đầu thời gian cho mỗi client
                client.train()  # Huấn luyện client
                client_time = time.time() - start_time_client  # Thời gian huấn luyện của client
                client_round_times.append(client_time)  # Lưu thời gian của client

                # In ra thời gian huấn luyện của từng client
                print(f"Client {client.id} - Training time: {client_time:.4f} seconds")

            # Tính trung bình thời gian của tất cả client trong round này
            avg_round_time = np.mean(client_round_times) if client_round_times else 0
            print(f"Round {i} - Average Training Time: {avg_round_time:.4f} seconds")

            # Tính trung bình cộng thời gian quantum của các client trong vòng này
            avg_round_quantum_time = np.mean([client.model.quantum_time for client in self.selected_clients]) \
                if self.selected_clients else 0
            print(f"Round {i} - Average Quantum Layer Time: {avg_round_quantum_time:.4f} seconds")

            self.receive_models()

            if self.dlg_eval and i % self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()

            # Thêm thời gian của vòng vào log
            self.Budget.append(time.time() - s_t)
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])

            # Lưu tổng hợp thời gian quantum của tất cả các client
            self.all_client_quantum_times.extend([client.client_quantum_times[-1] for client in self.selected_clients])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        # Tính trung bình cộng thời gian quantum của tất cả các client sau tất cả vòng
        avg_total_quantum_time = np.mean(self.all_client_quantum_times) if self.all_client_quantum_times else 0
        print(f"\nAverage Quantum Layer Time across all rounds: {avg_total_quantum_time:.4f} seconds")

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        self.save_results()
        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientPer)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1 - self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in active_clients:
            client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                               client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.model.base)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples
