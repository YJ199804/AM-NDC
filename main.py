import argparse
from train import *
from model import *
from utils import *
from datasets import *
import warnings

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--random_splits', type=bool, default=False)
parser.add_argument('--runs', type=int, default=10)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=0.005)
parser.add_argument('--early_stopping', type=int, default=100)
parser.add_argument('--hidden_dim', type=int, default=64)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--k', type=int, default=2)
parser.add_argument('--alpha', type=float, default=1)
parser.add_argument('--tau', type=float, default=0.5)

args = parser.parse_args()

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.dataset == "Cora" or args.dataset == "CiteSeer" or args.dataset == "PubMed":
    dataset = get_planetoid_dataset(args.dataset, args.normalize_features)
    permute_masks = random_planetoid_splits if args.random_splits else None
    print("Data:", dataset[0])
    if args.dataset == "Cora":
        A_hat = get_A_hat(dataset[0])
        A_hat = torch.FloatTensor(A_hat).float().to(device)
        adj_label_1 = get_A_r(A_hat, 1)
        adj_label_2 = get_A_r(A_hat, 2)
        model = AM_NDC_two_layer(dataset.num_features, args.hidden_dim, dataset.num_classes, args.dropout).to(device)
        run(dataset, model, args.runs, args.epochs, args.lr, args.weight_decay, args.early_stopping, args.alpha,
            args.tau,
            args.k, permute_masks, False, adj_label_1, adj_label_2)
    else:
        A_hat = get_A_hat(dataset[0])
        A_hat = torch.FloatTensor(A_hat).float().to(device)
        adj_label_1 = get_A_r(A_hat, 1)
        adj_label_2 = get_A_r(A_hat, 2)
        adj_label_3 = get_A_r(A_hat, 3)
        model = AM_NDC_three_layer(dataset.num_features, args.hidden_dim, dataset.num_classes, args.dropout).to(device)
        run(dataset, model, args.runs, args.epochs, args.lr, args.weight_decay, args.early_stopping, args.alpha,
            args.tau,
            args.k, permute_masks, False, adj_label_1, adj_label_2, adj_label_3)

elif args.dataset == "computers" or args.dataset == "photo":
    dataset = get_amazon_dataset(args.dataset, False)
    permute_masks = random_coauthor_amazon_splits
    print("Data:", dataset[0])
    if args.dataset == "photo":
        A_hat = get_A_hat(dataset[0])
        A_hat = torch.FloatTensor(A_hat).float().to(device)
        adj_label_1 = get_A_r(A_hat, 1)
        adj_label_2 = get_A_r(A_hat, 2)
        adj_label_3 = get_A_r(A_hat, 3)
        adj_label_4 = get_A_r(A_hat, 4)
        model = AM_NDC_four_layer(dataset.num_features, args.hidden_dim, dataset.num_classes, args.dropout).to(device)
        run(dataset, model, args.runs, args.epochs, args.lr, args.weight_decay, args.early_stopping, args.alpha,
            args.tau,
            args.k, permute_masks, True, adj_label_1, adj_label_2, adj_label_3, adj_label_4)
    else:
        A_hat = get_A_hat(dataset[0])
        A_hat = torch.FloatTensor(A_hat).float().to(device)
        adj_label_1 = get_A_r(A_hat, 1)
        adj_label_2 = get_A_r(A_hat, 2)
        model = AM_NDC_two_layer(dataset.num_features, args.hidden_dim, dataset.num_classes, args.dropout).to(device)
        run(dataset, model, args.runs, args.epochs, args.lr, args.weight_decay, args.early_stopping, args.alpha,
            args.tau,
            args.k, permute_masks, True, adj_label_1, adj_label_2)

