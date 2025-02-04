# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import time
from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
from threading import Thread
import numpy as np

class FedAvg(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.all_client_quantum_times = []  # Danh sách lưu thời gian quantum của tất cả các client

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientAVG)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []


    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
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

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])
            # Lưu tổng hợp thời gian quantum của tất cả các client
            self.all_client_quantum_times.extend([client.client_quantum_times[-1] for client in self.selected_clients])



            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        # Tính trung bình cộng thời gian quantum của tất cả các client sau tất cả vòng
        avg_total_quantum_time = np.mean(self.all_client_quantum_times) if self.all_client_quantum_times else 0
        print(f"\nAverage Quantum Layer Time across all rounds: {avg_total_quantum_time:.4f} seconds")

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientAVG)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()
