from utils import *
from torch.optim import Adam
from tqdm import tqdm
import time
from torch import tensor
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_two_layer_model(model, optimizer, data, adj_label_1, adj_label_2, alpha, tau):
    model.train()
    optimizer.zero_grad()
    output, x_1, x_2 = model(data.x)
    loss_train_class = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
    loss_Ncontrast_1 = Ncontrast(x_1, adj_label_1, tau)
    loss_Ncontrast_2 = Ncontrast(x_2, adj_label_2, tau)
    loss_train = loss_train_class + alpha * (loss_Ncontrast_1 + loss_Ncontrast_2)
    loss_train.backward()
    optimizer.step()
    return


def train_three_layer_model(model, optimizer, data, adj_label_1, adj_label_2, adj_label_3, alpha, tau):
    model.train()
    optimizer.zero_grad()
    output, x_1, x_2, x_3 = model(data.x)
    loss_train_class = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
    loss_Ncontrast_1 = Ncontrast(x_1, adj_label_1, tau)
    loss_Ncontrast_2 = Ncontrast(x_2, adj_label_2, tau)
    loss_Ncontrast_3 = Ncontrast(x_3, adj_label_3, tau)
    loss_train = loss_train_class + alpha * (loss_Ncontrast_1 + loss_Ncontrast_2 + loss_Ncontrast_3)
    loss_train.backward()
    optimizer.step()
    return


def train_four_layer_model(model, optimizer, data, adj_label_1, adj_label_2, adj_label_3, adj_label_4, alpha, tau):
    model.train()
    optimizer.zero_grad()
    output, x_1, x_2, x_3, x_4 = model(data.x)
    loss_train_class = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
    loss_Ncontrast_1 = Ncontrast(x_1, adj_label_1, tau)
    loss_Ncontrast_2 = Ncontrast(x_2, adj_label_2, tau)
    loss_Ncontrast_3 = Ncontrast(x_3, adj_label_3, tau)
    loss_Ncontrast_4 = Ncontrast(x_4, adj_label_4, tau)
    loss_train = loss_train_class + alpha * (loss_Ncontrast_1 + loss_Ncontrast_2 + loss_Ncontrast_3 + loss_Ncontrast_4)
    loss_train.backward()
    optimizer.step()
    return


def evaluate(model, data):
    model.eval()

    with torch.no_grad():
        logits = model(data.x)

    outs = {}
    for key in ['train', 'val', 'test']:
        mask = data['{}_mask'.format(key)]
        loss = F.nll_loss(logits[mask], data.y[mask]).item()
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()

        outs['{}_loss'.format(key)] = loss
        outs['{}_acc'.format(key)] = acc

    return outs


def run(dataset, model, runs, epochs, lr, weight_decay, early_stopping, alpha, tau, k, permute_masks=None, lcc=False, adj_label_1=None, adj_label_2=None,
        adj_label_3=None, adj_label_4=None):
    val_losses, accs, durations = [], [], []

    lcc_mask = None
    if lcc:  # select largest connected component
        data_ori = dataset[0]
        data_nx = to_networkx(data_ori)
        data_nx = data_nx.to_undirected()
        print("Original #nodes:", data_nx.number_of_nodes())
        data_nx = data_nx.subgraph(max(nx.connected_components(data_nx), key=len))
        print("#Nodes after lcc:", data_nx.number_of_nodes())
        lcc_mask = list(data_nx.nodes)

    data = dataset[0]

    pbar = tqdm(range(runs), unit='run')

    for _ in pbar:

        if permute_masks is not None:
            data = permute_masks(data, dataset.num_classes, lcc_mask)
        data = data.to(device)

        model.reset_parameters()
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_start = time.perf_counter()

        best_val_loss = float('inf')
        test_acc = 0
        val_loss_history = []

        for epoch in range(1, epochs + 1):
            if k == 2:
                out = train_two_layer_model(model, optimizer, data, adj_label_1, adj_label_2, alpha, tau)
            elif k == 3:
                out = train_three_layer_model(model, optimizer, data, adj_label_1, adj_label_2, adj_label_3, alpha, tau)
            else:
                out = train_four_layer_model(model, optimizer, data, adj_label_1, adj_label_2, adj_label_3, adj_label_4,
                                             alpha, tau)
            eval_info = evaluate(model, data)
            eval_info['epoch'] = epoch

            if eval_info['val_loss'] < best_val_loss:
                best_val_loss = eval_info['val_loss']
                test_acc = eval_info['test_acc']

            val_loss_history.append(eval_info['val_loss'])
            if early_stopping > 0 and epoch > epochs // 2:
                tmp = tensor(val_loss_history[-(early_stopping + 1):-1])
                if eval_info['val_loss'] > tmp.mean().item():
                    break

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_end = time.perf_counter()

        val_losses.append(best_val_loss)
        accs.append(test_acc)
        durations.append(t_end - t_start)

    loss, acc, duration = tensor(val_losses), tensor(accs), tensor(durations)

    print('Val Loss: {:.4f}, Test Accuracy: {:.3f} ± {:.3f}, Duration: {:.3f}'.
          format(loss.mean().item(),
                 acc.mean().item(),
                 acc.std().item(),
                 duration.mean().item()))
