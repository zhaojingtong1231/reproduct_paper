import torch
import torch.nn.functional as F
import argparse
from utils import *

try:
    from torch_lr_finder import LRFinder, TrainDataLoaderIter, ValDataLoaderIter
except ImportError:
    # Run from source
    import sys
    sys.path.insert(0, '../../../Desktop')
    from torch_lr_finder import LRFinder, TrainDataLoaderIter, ValDataLoaderIter

class Net(nn.Module):
    def __init__(self, in_channels):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=5, stride=3)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, stride=1)
        self.pool1 = nn.MaxPool1d(2, 2)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=13, stride=1)
        self.conv4 = nn.Conv1d(128, 256, kernel_size=7, stride=1)
        self.pool2 = nn.MaxPool1d(2, 2)
        self.conv5 = nn.Conv1d(256, 256, kernel_size=5, stride=1)
        self.conv6 = nn.Conv1d(256, 64, kernel_size=5, stride=1)
        self.pool3 = nn.MaxPool1d(2, 2)
        self.conv7 = nn.Conv1d(64, 32, kernel_size=3, stride=1)
        self.conv8 = nn.Conv1d(32, 64, kernel_size=6, stride=1)
        self.pool4 = nn.MaxPool1d(2, 2)
        self.conv9 = nn.Conv1d(64, 8, kernel_size=5, stride=1)
        self.conv10 = nn.Conv1d(8, 8, kernel_size=2, stride=1)
        self.pool5 = nn.MaxPool1d(2, 2)
        self.fc1 = nn.Linear(440, 64)
        self.fc2 = nn.Linear(64, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool3(x)
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = self.pool4(x)
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        x = self.pool5(x)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        # x = F.softmax(self.fc2(x), dim=1)
        x = self.fc2(x)
        return x



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--epochs", type=int, default=500,
                        help="number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--lr", type=float, default=0.0001,
                        help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0001,
                        help="weight decay")
    parser.add_argument("--seed", type=int, default=17,
                        help="random seed")
    parser.add_argument("--pkl_dir", type=str, default="/data/zhaojingtong/PSG/5_classes_standardized_standardscaler.pkl", help="the path of the pickle file")

    parser.add_argument("--in_channels", type=int, default=24, help="input channels")
    parser.add_argument("--lstm_model_type", type=str, default="lstm", choices=["lstm", "mlstm", 'slstm'])
    parser.add_argument("--input_size", type=int, default=512, help="lstm input size")
    parser.add_argument("--hidden_size", type=int, default=512, help="lstm hidden size")
    parser.add_argument("--num_layers", type=int, default=1, help="lstm layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="lstm dropout")
    parser.add_argument("--graph_model_type", type=str, default="gcn", choices=["gcn", "gat"])
    parser.add_argument("--num_features", type=int, default=3000, help="graph feature size")
    parser.add_argument("--num_classes", type=int, default=5, help="number of classes")

    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    weight_decay = args.weight_decay
    seed = args.seed
    pkl_dir = args.pkl_dir

    in_channels = args.in_channels
    lstm_model_type = args.lstm_model_type
    input_size = args.input_size
    hidden_size = args.hidden_size
    num_layers = args.num_layers
    dropout = args.dropout
    graph_model_type = args.graph_model_type
    num_features = args.num_features
    num_classes = args.num_classes

    save_path = f"lr{lr}_in_channels{in_channels}_lstm_model_type{lstm_model_type}_num_layers{num_layers}_graph_model_type{graph_model_type}_num_features{num_features}_num_classes{num_classes}"
    # model = Hybrid_CNN(in_channels, lstm_model_type, input_size, hidden_size, num_layers, dropout, graph_model_type, num_features, num_classes, device)
    model = Net(10)
    model.to(device)
    training(pkl_dir, device, batch_size, model, lr, weight_decay, epochs, save_path)


