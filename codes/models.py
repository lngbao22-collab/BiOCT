import torch
from torch import nn
from tqdm import tqdm
from abc import ABC, abstractmethod
from typing import Tuple, List, Dict
from qutils import *
import os
import numpy as np

class KBCModel(nn.Module, ABC):
    def get_ranking(
            self, queries: torch.Tensor,
            filters: Dict[Tuple[int, int], List[int]],
            batch_size: int = 1000, chunk_size: int = -1
    ):
        ranks = torch.ones(len(queries))
        with tqdm(total=queries.shape[0], unit='ex') as bar:
            bar.set_description(f'Evaluation')
            with torch.no_grad():
                b_begin = 0
                while b_begin < len(queries):
                    these_queries = queries[b_begin:b_begin + batch_size]
                    target_idxs = these_queries[:, 2].cpu().tolist()
                    scores, _ = self.forward(these_queries)
                    targets = torch.stack([scores[row, col] for row, col in enumerate(target_idxs)]).unsqueeze(-1)

                    for i, query in enumerate(these_queries):
                        filter_out = filters[(query[0].item(), query[1].item())]
                        filter_out += [queries[b_begin + i, 2].item()]   
                        scores[i, torch.LongTensor(filter_out)] = -1e6
                    ranks[b_begin:b_begin + batch_size] += torch.sum(
                        (scores >= targets).float(), dim=1
                    ).cpu()
                    b_begin += batch_size
                    bar.update(batch_size)
        return ranks

class BiQUE(KBCModel):
    def __init__(self, sizes: Tuple[int, int, int], rank: int, init_size: float = 1e-3):
        super(BiQUE, self).__init__()
        self.sizes = sizes
        self.rank = rank
        self.h1 = nn.Embedding(sizes[0], 8 * rank, sparse=True)
        self.h2 = nn.Embedding(sizes[0], 8 * rank, sparse=True)
        self.r1 = nn.Embedding(sizes[1], 16 * rank, sparse=True) 
        self.r2 = nn.Embedding(sizes[1], 16 * rank, sparse=True) 
        self.h1.weight.data *= init_size
        self.h2.weight.data *= init_size
        self.r1.weight.data *= init_size
        self.r2.weight.data *= init_size

    def forward(self, x):
        head1 = self.h1(x[:, 0])
        head2 = self.h2(x[:, 0])
        rel1 = self.r1(x[:, 1])
        rel2 = self.r2(x[:, 1])
        tail1 = self.h1(x[:, 2])
        tail2 = self.h2(x[:, 2])

        head1 += rel1[:, self.rank * 8:]
        head2 += rel2[:, self.rank * 8:]
        w_a1, x_a1, y_a1, z_a1 = torch.split(head1, self.rank * 2, dim=-1)
        w_a2, x_a2, y_a2, z_a2 = torch.split(head2, self.rank * 2, dim=-1)
        w_b1, x_b1, y_b1, z_b1 = torch.split(rel1[:, :self.rank*8], self.rank * 2, dim=-1) 
        w_b2, x_b2, y_b2, z_b2 = torch.split(rel2[:, :self.rank*8], self.rank * 2, dim=-1)

        A = biquaternion_mul(w_a1, x_a1, y_a1, z_a1, w_b1, x_b1, y_b1, z_b1)
        B = biquaternion_mul(complex_conjugate(w_b2), -complex_conjugate(x_b2), -complex_conjugate(y_b2), -complex_conjugate(z_b2), w_a2, x_a2, y_a2, z_a2)
        C = biquaternion_mul(w_b2, x_b2, y_b2, z_b2, w_a1, x_a1, y_a1, z_a1)
        D = biquaternion_mul(w_a2, x_a2, y_a2, z_a2, complex_conjugate(w_b1), -complex_conjugate(x_b1), -complex_conjugate(y_b1), -complex_conjugate(z_b1))
        res = torch.cat([A - B, C + D], -1)
        return  res @ torch.cat([self.h1.weight.transpose(0, 1), self.h2.weight.transpose(0, 1)], 0), [(get_norm(head1, head2, 8), get_norm(rel1[:, :self.rank*8], rel2[:, :self.rank*8], 8), get_norm(tail1, tail2, 8))]