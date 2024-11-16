import torch
import torch.nn as nn
import torch.nn.functional as F
from models import DGI, GraphCL
from layers import GCN, AvgReadout


class downprompt(nn.Module):
    def __init__(self, prompt1, prompt2, prompt3, a4, ft_in, nb_classes, feature, labels):
        super(downprompt, self).__init__()

        self.downprompt = downstreamprompt(ft_in)

        self.nb_classes = nb_classes
        self.labels = labels

        self.a4 = a4
        self.leakyrelu = nn.ELU()
        self.prompt = torch.cat((prompt1, prompt2, prompt3), 0)

        self.nodelabelprompt = weighted_prompt(3)

        self.dffprompt = weighted_feature(2)

        feature = feature.squeeze().cuda()

        self.aveemb0 = torch.FloatTensor(ft_in, ).cuda()
        self.aveemb1 = torch.FloatTensor(ft_in, ).cuda()
        self.aveemb2 = torch.FloatTensor(ft_in, ).cuda()
        self.aveemb3 = torch.FloatTensor(ft_in, ).cuda()
        self.aveemb4 = torch.FloatTensor(ft_in, ).cuda()
        self.aveemb5 = torch.FloatTensor(ft_in, ).cuda()
        self.aveemb6 = torch.FloatTensor(ft_in, ).cuda()

        self.one = torch.ones(1, ft_in).cuda()

        self.ave = torch.FloatTensor(nb_classes, ft_in).cuda()

    def forward(self, seq, seq1, train=0):
        weight = self.leakyrelu(self.nodelabelprompt(self.prompt))
        weight = self.one + weight

        rawret1 = weight * seq
        rawret2 = self.downprompt(seq)
        rawret4 = seq1

        rawret3 = self.dffprompt(rawret1, rawret2)
        # # print("a4",self.a4,"a5",self.a5)

        rawret = rawret3 + self.a4 * rawret4

        # rawret = seq
        rawret = rawret.cuda()
        # rawret = torch.stack((rawret,rawret,rawret,rawret,rawret,rawret))
        if train == 1:
            self.ave = averageemb(labels=self.labels, rawret=rawret, nb_class=self.nb_classes)
            # if self.labels[x].item() == 6:
            #     self.aveemb6 = rawret[x]
        # self.ave = weight * self.ave
        # print("rawretsize",rawret.size())

        ret = torch.FloatTensor(seq.shape[0], self.nb_classes).cuda()

        for x in range(0, seq.shape[0]):
            ret[x][0] = torch.cosine_similarity(rawret[x], self.ave[0], dim=0)
            ret[x][1] = torch.cosine_similarity(rawret[x], self.ave[1], dim=0)
            ret[x][2] = torch.cosine_similarity(rawret[x], self.ave[2], dim=0)
            ret[x][3] = torch.cosine_similarity(rawret[x], self.ave[3], dim=0)
            ret[x][4] = torch.cosine_similarity(rawret[x], self.ave[4], dim=0)
            ret[x][5] = torch.cosine_similarity(rawret[x], self.ave[5], dim=0)
            if self.nb_classes == 7:
                ret[x][6] = torch.cosine_similarity(rawret[x], self.ave[6], dim=0)

        ret = F.softmax(ret, dim=1)
        return ret

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)


def averageemb(labels, rawret, nb_class):
    retlabel = torch.FloatTensor(nb_class, int(rawret.shape[0] / nb_class), int(rawret.shape[1])).cuda()
    cnt1 = 0
    cnt2 = 0
    cnt3 = 0
    cnt4 = 0
    cnt5 = 0
    cnt6 = 0
    cnt7 = 0
    # print("labels",labels)
    for x in range(0, rawret.shape[0]):
        if labels[x].item() == 0:
            retlabel[0][cnt1] = rawret[x]
            cnt1 = cnt1 + 1
        if labels[x].item() == 1:
            retlabel[1][cnt2] = rawret[x]
            cnt2 = cnt2 + 1
        if labels[x].item() == 2:
            retlabel[2][cnt3] = rawret[x]
            cnt3 = cnt3 + 1
        if labels[x].item() == 3:
            retlabel[3][cnt4] = rawret[x]
            cnt4 = cnt4 + 1
        if labels[x].item() == 4:
            retlabel[4][cnt5] = rawret[x]
            cnt5 = cnt5 + 1
        if labels[x].item() == 5:
            retlabel[5][cnt6] = rawret[x]
            cnt6 = cnt6 + 1
        if labels[x].item() == 6:
            retlabel[6][cnt7] = rawret[x]
            cnt7 = cnt7 + 1
    retlabel = torch.mean(retlabel, dim=1)
    return retlabel


class weighted_prompt(nn.Module):
    def __init__(self, weightednum):
        super(weighted_prompt, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(1, weightednum), requires_grad=True)
        self.act = nn.ELU()
        self.reset_parameters()

    def reset_parameters(self):
        # torch.nn.init.xavier_uniform_(self.weight)

        self.weight[0][0].data.fill_(0.5)
        self.weight[0][1].data.fill_(0.4)
        self.weight[0][2].data.fill_(0.3)

    def forward(self, graph_embedding):
        # print("weight",self.weight)
        graph_embedding = torch.mm(self.weight, graph_embedding)
        return graph_embedding


class weighted_feature(nn.Module):
    def __init__(self, weightednum):
        super(weighted_feature, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(1, weightednum), requires_grad=True)
        self.act = nn.ELU()
        self.reset_parameters()

    def reset_parameters(self):
        # torch.nn.init.xavier_uniform_(self.weight)

        self.weight[0][0].data.fill_(0)
        self.weight[0][1].data.fill_(1)

    def forward(self, graph_embedding1, graph_embedding2):
        # print("weight",self.weight)
        graph_embedding = self.weight[0][0] * graph_embedding1 + self.weight[0][1] * graph_embedding2
        return self.act(graph_embedding)


class downstreamprompt(nn.Module):
    def __init__(self, hid_units):
        super(downstreamprompt, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(1, hid_units), requires_grad=True)
        self.act = nn.ELU()
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

        # self.weight[0][0].data.fill_(0.3)
        # self.weight[0][1].data.fill_(0.3)
        # self.weight[0][2].data.fill_(0.3)

    def forward(self, graph_embedding):
        # print("weight",self.weight)
        graph_embedding = self.weight * graph_embedding
        return graph_embedding


class featureprompt(nn.Module):
    def __init__(self, prompt1, prompt2, prompt3):
        super(featureprompt, self).__init__()
        self.prompt = torch.cat((prompt1, prompt2, prompt3), 0)
        self.weightprompt = weighted_prompt(3)

    def forward(self, feature):
        # print("prompt",self.weightprompt.weight)
        weight = self.weightprompt(self.prompt)
        feature = weight * feature
        return feature
