import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score, f1_score
import torch.nn.functional as F

class PSGDataset(Dataset):
    def __init__(self, pkl_file, device):
        self.device = device

        with open(pkl_file, 'rb') as f:
            data_dict = pickle.load(f)

        self.data = data_dict['data']
        self.labels = data_dict['label']

        # self.data = self.data[:, [0,1,3,4,5,6,7,8,12,13], :]
        self.data = self.data[:, [0, 1, 4, 5, 6, 7, 8, 9, 13, 14], :]
        # self.labels = self.labels[:, [0,1,3,4,5,6,7,8,12,13], :]
    #
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        sample = torch.tensor(sample, dtype=torch.float).to(self.device)
        label = torch.tensor(label, dtype=torch.float).to(self.device)
        return sample, label


class Trainer(object):
    def __init__(self, model, lr, weight_decay):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=lr, weight_decay=weight_decay)
        self.criterion = nn.CrossEntropyLoss()

    def train(self, dataloader):
        self.model.train()
        loss_total = 0
        self.optimizer.zero_grad()
        for batch_data, batch_labels in dataloader:
            # 将数据移动到指定的设备上
            batch_data, batch_labels = batch_data, batch_labels

            # 前向传播
            outputs = self.model(batch_data)
            # 计算损失
            loss = self.criterion(outputs, batch_labels)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_total += loss.item() * batch_data.size(0)  # 累积总损失，乘以批量大小以获得正确的损失总和

        return loss_total / len(dataloader.dataset)  # 返回平均损失


class Tester(object):
    def __init__(self, model):
        self.model = model

    def test(self, dataloader):
        self.model.eval()
        all_labels = []
        all_predictions = []
        all_scores = []
        with torch.no_grad():
            for batch_data in dataloader:
                inputs, labels = batch_data
                predicted_scores = self.model(inputs)  # Assuming the model returns scores directly
                predicted_labels = np.argmax(predicted_scores.cpu().numpy(), axis=1)

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted_labels)
                all_scores.extend(F.softmax(predicted_scores, dim=1).cpu().numpy())

        # Convert lists to numpy arrays
        all_labels = np.array(all_labels)
        all_labels = np.argmax(all_labels, axis=1)
        all_predictions = np.array(all_predictions)
        all_scores = np.array(all_scores)

        # Calculate metrics
        acc = accuracy_score(all_labels, all_predictions)
        auc = roc_auc_score(all_labels, all_scores, multi_class="ovr")
        precision = precision_score(all_labels, all_predictions, average='macro')
        recall = recall_score(all_labels, all_predictions, average='macro')
        f1 = f1_score(all_labels, all_predictions, average='macro')

        return acc, auc, precision, recall, f1

    def save_AUCs(self, AUCs, filename):
        with open(filename, 'a') as f:
            f.write('\t'.join(map(str, AUCs)) + '\n')

    def save_model(self, model, filename):
        torch.save(model.state_dict(), filename)


def training(pkl_dir, device, batch_size, model, lr, weight_decay, epochs, save_path):
    best_auc = 0

    dataset = PSGDataset(pkl_dir, device)
    train_size = int(0.8 * len(dataset))  # 80% 用于训练
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    trainer = Trainer(model, lr, weight_decay)
    tester = Tester(model)

    print('Epoch\tLoss_train\tACC_test\tAUC_test\tPrecision_test\tRecall_test\tF1')
    for i in range(epochs):
        loss = trainer.train(train_loader)
        acc, auc, precision, recall, f1 = tester.test(test_loader)
        results = [i, loss, acc, auc, precision, recall, f1]
        print('\t'.join(map(str, results)))
        # if best_auc < auc:
        #     model_path = './results/checkpoints/'+save_path+'.pth'
        #     auc_path = './results/auc/'+save_path+'.txt'
        #     tester.save_model(model, model_path)
        #     tester.save_AUCs(results, auc_path)
