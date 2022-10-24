import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import time
from transformers import BartModel

class projection_MLP(nn.Module):
    def __init__(self, in_dim, hidden_size, out_dim=4000):
        super().__init__()
        # hidden_dim1 = 1024
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_size, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class SimCLR(nn.Module):
    def __init__(self):
        super().__init__()

        self.projection = projection_MLP(1024, 3)
        self.temperature = 2

        self.projt = nn.Linear(1024, 2)

    def nt_xentloss(self, z1, z2):
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        N, Z = z1.shape
        device = z1.device
        #print('1',z1.shape)
        #print('2',z2.shape)
        representations = torch.cat([z1, z2], dim=1)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=-1)
        l_pos = torch.diag(similarity_matrix, N)
        r_pos = torch.diag(similarity_matrix, -N)
        positives = torch.cat([similarity_matrix, similarity_matrix])[:,-1]

        diag = torch.eye(2 * N, dtype=torch.bool, device=device)
        diag[N:, :N] = diag[:N, N:] = diag[:N, :N]

        negatives = similarity_matrix[~diag[:int((diag.shape[0])/2), : int((diag.shape[0])/2)]][:positives.shape[0]]

        logits = torch.cat([positives.unsqueeze(1), negatives.unsqueeze(1)], dim=1)
        #logits = self.projt(logits)
        logits /= self.temperature

        labels = torch.zeros(2 * N, device=device, dtype=torch.int64)

        loss = F.cross_entropy(logits, labels, reduction='sum')
        return loss / (2 * N)   #公式3

    def forward(self, x1, x2):
        x1 = self.projection(x1)
        x2 = self.projection(x2)
        loss = self.nt_xentloss(x1, x2)
        return loss



class KGEModel(nn.Module):
    def __init__(self,
                 model_name,
                 nentity, nrelation,
                 hidden_dim, gamma,
                 double_entity_embedding=False,
                 double_relation_embedding=False
                 ):
        super(KGEModel, self).__init__()

        #self.ln1 = nn.Linear(1024, 1)
        #self.ln2 = nn.Linear(768, 1024)

        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )

        self.entity_dim = hidden_dim*2 if double_entity_embedding else hidden_dim
        self.relation_dim = hidden_dim*2 if double_relation_embedding else hidden_dim

        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )
        self.mem = nn.GRU(input_size=1024, hidden_size=1024, num_layers=1, bias=True)
        if model_name == 'pRotatE':
            self.modulus = nn.Parameter(torch.Tensor([[0.5 * self.embedding_range.item()]]))
        self.pi = 3.14159262358979323846


    def forward(self, sample, mode='single'):
        if mode == 'single':
            batch_size, negative_sample_size = sample.size(0), 1

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:,0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=sample[:,1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:,2]
            ).unsqueeze(1)

        elif mode == 'head-batch':
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part.view(-1).long()
            ).view(batch_size, negative_sample_size, -1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part[:, 2]
            ).unsqueeze(1)

        elif mode == 'tail-batch':
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part.view(-1).long()
            ).view(batch_size, negative_sample_size, -1)

        else:
            raise ValueError('mode %s not supported' % mode)

        model_func = {
            'TransE': self.TransE,
            'RotatE': self.RotatE,
            'pRotatE': self.pRotatE,
            'ComplEx': self.ComplEx
        }
        #bart_model = BartModel.from_pretrained('./ptm/').to('cuda')

        if self.model_name in model_func:
            score = model_func[self.model_name](head, relation, tail, mode)
        else:
            raise ValueError('model %s not supported' % self.model_name)

        return score

    def TransE(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail
        #score = self.ln2(bart_model(self.ln1((head + (relation - tail))).squeeze(-1).long())['last_hidden_state'])
        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def RotatE(self, head, relation, tail, mode):
        pi = 3.14159265358979323846
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)

        #Make phases of relations uniformly distributed in [-pi, pi]

        re_phase_relation = re_relation/(self.embedding_range.item()/pi)
        im_phase_relation = im_relation/(self.embedding_range.item()/pi)

        re_relation = torch.cos(re_phase_relation)
        im_relation = torch.sin(im_phase_relation)

        re_score = re_relation * re_tail + im_relation * im_tail
        im_score = re_relation * im_tail - im_relation * re_tail
        re_score = re_score - re_head
        im_score = im_score - im_head


        score = torch.stack([re_score, im_score], dim = 0)
        score = score.norm(dim=0)

        score = self.gamma.item() - score.sum(dim = 2)
        return score


    def pRotatE(self, head, relation, tail, mode):
        pi = 3.14159262358979323846
        #Make phases of entities and relations uniformly distributed in [-pi, pi]
        phase_head = head/(self.embedding_range.item()/pi)
        phase_relation = relation/(self.embedding_range.item()/pi)
        phase_tail = tail/(self.embedding_range.item()/pi)

        if mode == 'head-batch':
            score = phase_head + (phase_relation - phase_tail)
        else:
            score = (phase_head + phase_relation) - phase_tail

        score = torch.sin(score)
        score = torch.abs(score)

        score = self.gamma.item() - score.sum(dim = 2) * self.modulus
        return score


    def ComplEx(self, head, relation, tail, mode):
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            score = re_head * re_score + im_head * im_score
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            score = re_score * re_tail + im_score * im_tail

        score = score.sum(dim = 2)
        return score