import torch.nn.functional as F
from utils import *
from torch.nn import Dropout, Linear, LayerNorm
import torch.nn as nn
# two layer model
class Mlp_two_layer(nn.Module):
    def __init__(self, input_dim, hid_dim, dropout):
        super(Mlp_two_layer, self).__init__()
        self.fc1 = Linear(input_dim, hid_dim)
        self.fc2 = Linear(hid_dim, hid_dim)
        #         self.fc3 = Linear(hid_dim, hid_dim)
        #         self.act_fn = torch.nn.functional.gelu
        self._init_weights()

        self.dropout = Dropout(dropout)

    #         self.layernorm = LayerNorm(hid_dim, eps=1e-6)

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        #         nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    #         nn.init.normal_(self.fc3.bias, std=1e-6)

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()

    #         self.fc3.reset_parameters()

    def forward(self, x):
        x_1 = self.fc1(x)
        x = F.relu(x_1)
        x = self.dropout(x)
        x_2 = self.fc2(x)
        return x_1, x_2


class AM_NDC_two_layer(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(AM_NDC_two_layer, self).__init__()
        self.nhid = nhid
        self.mlp = Mlp_two_layer(nfeat, self.nhid, dropout)
        self.classifier = Linear(self.nhid, nclass)
        self.proj = Linear(self.nhid, 1)

    def reset_parameters(self):
        self.mlp.reset_parameters()
        self.classifier.reset_parameters()
        self.proj.reset_parameters()

    def forward(self, x):
        x_1, x_2 = self.mlp(x)
        #         x_1, x_2, x_3 = self.mlp(x)

        preds = []
        preds.append(x_1)
        preds.append(x_2)
        #         preds.append(x_3)
        pps = torch.stack(preds, dim=1)
        retain_score = self.proj(pps)
        retain_score = retain_score.squeeze()
        #         retain_score = self.nonlinearity_x(retain_score)
        retain_score = torch.sigmoid(retain_score)
        retain_score = retain_score.unsqueeze(1)
        feature_cls = torch.matmul(retain_score, pps).squeeze()

        if self.training:
            x_dis_1 = get_feature_dis(x_1)
            x_dis_2 = get_feature_dis(x_2)
        #             x_dis_3 = get_feature_dis(x_3)

        class_feature = self.classifier(feature_cls)
        class_logits = F.log_softmax(class_feature, dim=1)

        #         if self.training:
        #             return class_logits, x_dis_1, x_dis_2, x_dis_3
        if self.training:
            return class_logits, x_dis_1, x_dis_2
        else:
            return class_logits


# three layer model
class Mlp_three_layer(nn.Module):
    def __init__(self, input_dim, hid_dim, dropout):
        super(Mlp_three_layer, self).__init__()
        self.fc1 = Linear(input_dim, hid_dim)
        self.fc2 = Linear(hid_dim, hid_dim)
        self.fc3 = Linear(hid_dim, hid_dim)
        self._init_weights()

        self.dropout = Dropout(dropout)


    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)
        nn.init.normal_(self.fc3.bias, std=1e-6)

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()

    def forward(self, x):
        x_1 = self.fc1(x)
        x = F.relu(x_1)
        x = self.dropout(x)
        x_2 = self.fc2(x)
        x = F.relu(x_2)
        x = self.dropout(x)
        x_3 = self.fc3(x)
        return x_1, x_2, x_3

class AM_NDC_three_layer(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(AM_NDC_three_layer, self).__init__()
        self.nhid = nhid
        self.mlp = Mlp_three_layer(nfeat, self.nhid, dropout)
        self.classifier = Linear(self.nhid, nclass)
        self.proj = Linear(self.nhid, 1)

    def reset_parameters(self):
        self.mlp.reset_parameters()
        self.classifier.reset_parameters()
        self.proj.reset_parameters()

    def forward(self, x):
        x_1, x_2, x_3 = self.mlp(x)

        preds = []
        preds.append(x_1)
        preds.append(x_2)
        preds.append(x_3)
        pps = torch.stack(preds, dim=1)
        retain_score = self.proj(pps)
        retain_score = retain_score.squeeze()
        retain_score = torch.sigmoid(retain_score)
        retain_score = retain_score.unsqueeze(1)
        feature_cls = torch.matmul(retain_score, pps).squeeze()

        if self.training:
            x_dis_1 = get_feature_dis(x_1)
            x_dis_2 = get_feature_dis(x_2)
            x_dis_3 = get_feature_dis(x_3)

        class_feature = self.classifier(feature_cls)
        class_logits = F.log_softmax(class_feature, dim=1)

        if self.training:
            return class_logits, x_dis_1, x_dis_2, x_dis_3
        else:
            return class_logits

# four layer model
class Mlp_four_layer(nn.Module):
    def __init__(self, input_dim, hid_dim, dropout):
        super(Mlp_four_layer, self).__init__()
        self.fc1 = Linear(input_dim, hid_dim)
        self.fc2 = Linear(hid_dim, hid_dim)
        self.fc3 = Linear(hid_dim, hid_dim)
        self.fc4 = Linear(hid_dim, hid_dim)
        self.act_fn = torch.nn.functional.gelu
        self._init_weights()

        self.dropout = Dropout(dropout)
        self.layernorm = LayerNorm(hid_dim, eps=1e-6)

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)
        nn.init.normal_(self.fc3.bias, std=1e-6)
        nn.init.normal_(self.fc4.bias, std=1e-6)

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()
        self.fc4.reset_parameters()

    def forward(self, x):
        x_1 = self.fc1(x)
        x = F.relu(x_1)
        x = self.dropout(x)
        x_2 = self.fc2(x)
        x = F.relu(x_2)
        x = self.dropout(x)
        x_3 = self.fc3(x)
        x = F.relu(x_3)
        x = self.dropout(x)
        x_4 = self.fc4(x)
        return x_1, x_2, x_3, x_4

class AM_NDC_four_layer(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(AM_NDC_four_layer, self).__init__()
        self.nhid = nhid
        self.mlp = Mlp_four_layer(nfeat, self.nhid, dropout)
        self.classifier = Linear(self.nhid, nclass)
        self.proj = Linear(self.nhid, 1)

    def reset_parameters(self):
        self.mlp.reset_parameters()
        self.classifier.reset_parameters()
        self.proj.reset_parameters()

    def forward(self, x):
        x_1, x_2, x_3, x_4 = self.mlp(x)

        preds = []
        preds.append(x_1)
        preds.append(x_2)
        preds.append(x_3)
        preds.append(x_4)
        pps = torch.stack(preds, dim=1)
        retain_score = self.proj(pps)
        retain_score = retain_score.squeeze()
        retain_score = torch.sigmoid(retain_score)
        retain_score = retain_score.unsqueeze(1)
        feature_cls = torch.matmul(retain_score, pps).squeeze()

        if self.training:
            x_dis_1 = get_feature_dis(x_1)
            x_dis_2 = get_feature_dis(x_2)
            x_dis_3 = get_feature_dis(x_3)
            x_dis_4 = get_feature_dis(x_4)

        class_feature = self.classifier(feature_cls)
        class_logits = F.log_softmax(class_feature, dim=1)

        if self.training:
            return class_logits, x_dis_1, x_dis_2, x_dis_3, x_dis_4
        else:
            return class_logits

