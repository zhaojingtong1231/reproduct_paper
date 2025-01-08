import torch.nn.functional as F
import random
from sklearn.metrics import average_precision_score
from pretrain import Pretrain
import aug
from torch_geometric.loader import LinkNeighborLoader, HGTLoader, LinkLoader
from PrimeKG import FullGraphNegSampler
import torch
from PrimeKG import PreData
from utils import process
import argparse
from datetime import datetime
import os
import sys
from sklearn.metrics import precision_recall_curve, auc
def get_time_str():
    # 获取当前日期和时间
    now = datetime.now()
    # 提取月、日、小时和分钟
    month = now.month
    day = now.day
    hour = now.hour
    minute = now.minute
    # 将提取的信息格式化为用下划线隔开的字符串
    formatted_str = f"{month:02d}_{day:02d}_{hour:02d}_{minute:02d}"
    return formatted_str
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TxGNN_prompt')
    parser.add_argument("--gpu", type=int, default=2,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--num-hidden", type=int, default=512,
                        help="number of hidden units")
    parser.add_argument("--epochs", type=int, default=10,
                        help="number of pretrain.sh epochs")
    parser.add_argument("--seed", type=int, default=12,
                        help="random seed")
    parser.add_argument("--batch-size", type=int, default=1024,
                        help="loader small_data")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument("--edge", type=int, default=0,
                       help="0 con 1indication")
    parser.add_argument(
        "--split",
        type=str,
        default="cell_proliferation",
        choices=['random', 'complex_disease', 'disease_eval', 'cell_proliferation', 'mental_health', 'cardiovascular', 'anemia', 'adrenal_gland','autoimmune', 'metabolic_disorder', 'diabetes', 'neurodigenerative', 'full_graph', 'downstream_pred', 'few_edeges_to_kg', 'few_edeges_to_indications'],  # 指定合法的候选项
        help="Choose the data split type"
    )

    args = parser.parse_args()
    gpu_id = args.gpu
    hidden_dim = args.num_hidden
    seed = args.seed
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    edge = args.edge
    time_str = get_time_str()
    split = args.split

    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    dd_etypes = [('drug', 'contraindication', 'disease'),
                 ('drug', 'indication', 'disease')]
    edge_type = dd_etypes[edge]

    preData = PreData(data_folder_path='/data/zhaojingtong/PrimeKG/data_all/')
    model_save_path = '/data/zhaojingtong/PrimeKG/HAN/'+split+'/'+str(edge)
    model_save_path = os.path.join(model_save_path,
                                   f"lr{lr}_edge_type_{edge_type}_split{split}_time{time_str}_seed{seed}")

    os.makedirs(model_save_path, exist_ok=True)

    g, df, df_train, df_valid, df_test, disease_eval_idx, no_kg, g_valid_pos, g_valid_neg = preData.prepare_split(
        split=split, seed=seed, no_kg=False,hidden_dim = hidden_dim)

    g = process.initialize_node_embedding(g, hidden_dim)

    data = g
    l2_coef = 0.0001

    num_layers_num = 1
    dropout = 0.2
    data = data.to(device)
    model = Pretrain(data=data, hidden_dim=hidden_dim, batch_size=batch_size, num_layers_num=num_layers_num,
                     dropout=dropout, device=device)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)
    print('begin train ......')
    edge_types = data.edge_types
    g_valid_pos = g_valid_pos.to(device)
    g_valid_neg = g_valid_neg.to(device)

    pos_loader = LinkNeighborLoader(
        data,
        num_neighbors=[60,20],  # 为每个关系采样邻居个数
        batch_size=batch_size,
        edge_label_index=(edge_type, data[edge_type].edge_index)
    )

    best_auc = 0  # Initialize the best AUPRC value
    best_epoch = -1  # Track the epoch with the best result
    best_model_path = None  # Track the path of the best model

    for epoch in range(epochs):
        print('epoch:' + str(epoch))
        epoch_results = [epoch]  # Current epoch's results
        loss_all = 0
        auc_all = 0
        all_loss = 0
        all_auc = 0
        test_auc = 0

        # Training loop
        for pos_g in pos_loader:
            ng = FullGraphNegSampler(pos_g, k=1, method='fix_dst')
            neg_g = ng(pos_g)
            pred_score_pos, pred_score_neg = model.forword_minibatch(pos_g, neg_g, edge_type, pretrain_model=True)

            # Concatenate scores and labels
            scores = torch.cat((pred_score_pos, pred_score_neg), dim=0)
            labels = [1] * len(pred_score_pos) + [0] * len(pred_score_neg)

            # Compute binary cross-entropy loss
            loss = F.binary_cross_entropy(torch.sigmoid(scores), torch.Tensor(labels).float().to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate Precision-Recall and AUPRC
            precision, recall, _ = precision_recall_curve(
                torch.Tensor(labels).cpu().detach().numpy(),
                torch.sigmoid(scores).cpu().detach().numpy()
            )
            auprc = auc(recall, precision)

            all_loss += loss.item()
            all_auc += auprc

        print(f"Train AUPRC: {all_auc / len(pos_loader):.4f}")

        # Testing loop
        with torch.no_grad():
            for node_type in g.x_dict.keys():
                if node_type in g.x_dict:
                    g_valid_pos.x_dict[node_type] = g.x_dict[node_type].clone()

            test_loader = LinkNeighborLoader(
                g_valid_pos,
                num_neighbors=[60, 20],  # Number of neighbors for each relation
                batch_size=batch_size,
                edge_label_index=(edge_type, g_valid_pos[edge_type].edge_index)
            )

            for test_g in test_loader:
                ng = FullGraphNegSampler(test_g, k=1, method='fix_dst')
                neg_g = ng(test_g)
                pred_score_pos, pred_score_neg = model.forword_minibatch(test_g, neg_g, edge_type, pretrain_model=True)

                # Concatenate scores and labels
                scores = torch.cat((pred_score_pos, pred_score_neg), dim=0)
                labels = [1] * len(pred_score_pos) + [0] * len(pred_score_neg)

                # Calculate Precision-Recall and AUPRC
                precision, recall, _ = precision_recall_curve(
                    torch.Tensor(labels).cpu().detach().numpy(),
                    torch.sigmoid(scores).cpu().detach().numpy()
                )
                auprc = auc(recall, precision)
                test_auc += auprc

            avg_test_auc = test_auc / len(test_loader)
            print(f"{edge_type} AUPRC: {avg_test_auc:.4f}")

            # Save the best model and results
            if avg_test_auc > best_auc:
                best_auc = avg_test_auc
                best_epoch = epoch
                # Delete the previous best model if exists
                if best_model_path and os.path.exists(best_model_path):
                    os.remove(best_model_path)

                best_model_path = os.path.join(model_save_path, 'best_model.pth')
                torch.save(model.state_dict(), best_model_path)
                with open(os.path.join(model_save_path, "result.txt"), "w") as f:
                    f.write(f"Best Epoch: {best_epoch}\n")
                    f.write(f"{edge_type} Best AUPRC: {best_auc:.4f}\n")
