import os
import pickle
import numpy as np
from typing import Dict, Tuple, List
from models import KBCModel
from metrics import evaluate_link_prediction

class Dataset(object):
    def __init__(self, data_path: str, name: str):
        self.root = os.path.join(data_path, name)

        self.data = {}
        for f in ['train', 'test', 'valid']:
            in_file = open(os.path.join(self.root, f + '.pickle'), 'rb')
            self.data[f] = pickle.load(in_file)

        print(self.data['train'].shape)

        maxis = np.max(self.data['train'], axis=0)
        self.n_entities = int(max(maxis[0], maxis[2]) + 1)
        self.n_predicates = int(maxis[1] + 1)
        self.n_predicates *= 2

        inp_f = open(os.path.join(self.root, 'to_skip.pickle'), 'rb')
        self.to_skip: Dict[str, Dict[Tuple[int, int], List[int]]] = pickle.load(inp_f)
        inp_f.close()

    def get_weight(self):
        appear_list = np.zeros(self.n_entities)
        copy = np.copy(self.data['train'])
        for triple in copy:
            h, r, t = triple
            appear_list[h] += 1
            appear_list[t] += 1

        w = appear_list / np.max(appear_list) * 0.9 + 0.1
        return w

    def get_examples(self, split):
        return self.data[split]

    def get_train(self):
        copy = np.copy(self.data['train'])
        tmp = np.copy(copy[:, 0])
        copy[:, 0] = copy[:, 2]
        copy[:, 2] = tmp
        copy[:, 1] += self.n_predicates // 2  
        return np.vstack((self.data['train'], copy))

    def eval(
            self, model: KBCModel, split: str, n_queries: int = -1, missing_eval: str = 'both',
            at: Tuple[int] = (1, 3, 10), log_result=False, save_path=None
    ):
        side_metrics, _ = evaluate_link_prediction(
            self,
            model,
            split,
            n_queries=n_queries,
            missing_eval=missing_eval,
            at=at,
        )

        mean_reciprocal_rank = {k: v['MRR'] for k, v in side_metrics.items()}
        hits_at = {k: v['hits'] for k, v in side_metrics.items()}
        return mean_reciprocal_rank, hits_at

    def get_shape(self):
        return self.n_entities, self.n_predicates, self.n_entities
####################################################################
# import os
# import torch
# import pickle
# import numpy as np
# from typing import Dict, Tuple, List
# from models import KBCModel

# class Dataset(object):
#     def __init__(self, data_path: str, name: str):
#         self.root = os.path.join(data_path, name)

#         self.data = {}
#         for f in ['train', 'test', 'valid']:
#             in_file = open(os.path.join(self.root, f + '.pickle'), 'rb')
#             self.data[f] = pickle.load(in_file)

#         print(self.data['train'].shape)

#         maxis = np.max(self.data['train'], axis=0)
#         self.n_entities = int(max(maxis[0], maxis[2]) + 1)
#         self.n_predicates = int(maxis[1] + 1)
#         self.n_predicates *= 2

#         inp_f = open(os.path.join(self.root, 'to_skip.pickle'), 'rb')
#         self.to_skip: Dict[str, Dict[Tuple[int, int], List[int]]] = pickle.load(inp_f)
#         inp_f.close()

#     def get_weight(self):
#         appear_list = np.zeros(self.n_entities)
#         copy = np.copy(self.data['train'])
#         for triple in copy:
#             h, r, t = triple
#             appear_list[h] += 1
#             appear_list[t] += 1

#         w = appear_list / np.max(appear_list) * 0.9 + 0.1
#         return w

#     def get_examples(self, split):
#         return self.data[split]

#     def get_train(self):
#         copy = np.copy(self.data['train'])
#         tmp = np.copy(copy[:, 0])
#         copy[:, 0] = copy[:, 2]
#         copy[:, 2] = tmp
#         copy[:, 1] += self.n_predicates // 2  
#         return np.vstack((self.data['train'], copy))

#     def eval(
#             self, model: KBCModel, split: str, n_queries: int = -1, missing_eval: str = 'both',
#             at: Tuple[int] = (1, 3, 10), log_result=False, save_path=None
#     ):
#         model.eval()
#         test = self.get_examples(split)
#         examples = torch.from_numpy(test.astype('int64')).cuda()
#         missing = [missing_eval]
#         if missing_eval == 'both':
#             missing = ['rhs', 'lhs']

#         mean_reciprocal_rank = {}
#         hits_at = {}
#         aaa, bbb = [], []

#         #eval lhs
#         q = examples.clone()
#         if n_queries > 0:
#             permutation = torch.randperm(len(examples))[:n_queries]
#             q = examples[permutation]
#         tmp = torch.clone(q[:, 0])
#         q[:, 0] = q[:, 2]
#         q[:, 2] = tmp
#         q[:, 1] += self.n_predicates // 2
#         ranks_lhs = model.get_ranking(q, self.to_skip['lhs'], batch_size=500)
#         for i in range(len(q.cpu().detach().numpy())):
#             if q[:, 1].cpu().detach().numpy()[i] == 21:
#                 aaa.append(ranks_lhs.cpu().detach().numpy()[i])
#         mean_reciprocal_rank['lhs'] = torch.mean(1./torch.as_tensor(aaa)).item()
#         hits_at['lhs'] = torch.FloatTensor((list(map(
#             lambda x: torch.mean((torch.as_tensor(aaa) <= x).float()).item(), at))))
#         #eval rhs
#         q = examples.clone()
#         if n_queries > 0:
#             permutation = torch.randperm(len(examples))[:n_queries]
#             q = examples[permutation]
#         ranks_rhs = model.get_ranking(q, self.to_skip['rhs'], batch_size=500)
#         for i in range(len(q.cpu().detach().numpy())):
#             if q[:, 1].cpu().detach().numpy()[i] == 10:
#                 bbb.append(ranks_rhs.cpu().detach().numpy()[i])
#         mean_reciprocal_rank['rhs'] = torch.mean(1./torch.as_tensor(bbb)).item()
#         hits_at['rhs'] = torch.FloatTensor((list(map(
#             lambda x: torch.mean((torch.as_tensor(bbb) <= x).float()).item(), at))))
#         return mean_reciprocal_rank, hits_at

#     def get_shape(self):
#         return self.n_entities, self.n_predicates, self.n_entities