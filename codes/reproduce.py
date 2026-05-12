import os
import sys
from pathlib import Path
from typing import Any, Dict

import torch

from datasets import Dataset
from metrics import evaluate_link_prediction
from models import *
from regularizers import *


class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def load_model(model, save_path):
    state = torch.load(os.path.join(save_path, 'checkpoint'), map_location=next(model.parameters()).device)
    model.load_state_dict(state)

    return model


def format_link_metrics(metrics: Dict[str, Any]):
    rounded_hits = [round(float(x), 3) for x in metrics['hits@[1,3,10]'].numpy().tolist()]
    return {
        'MR': round(float(metrics['MR']), 3),
        'MRR': round(float(metrics['MRR']), 3),
        'hits@[1,3,10]': rounded_hits,
    }


PROJECT_ROOT = Path(__file__).resolve().parent.parent
data_path = str(PROJECT_ROOT / "data")
checkpoint = sys.argv[1]
config = {
    "model": "BiQUE",
    "regularizer": "wN3",
    "optimizer": "Adagrad",
    "rank": 128,
    "batch_size": 5000,
    "init": 0.001,
}


if checkpoint == "WN18RR":
    save_path = str(PROJECT_ROOT / 'ckpt' / 'BiQUE_WN18RR')
    config["dataset"] = "WN18RR"
elif checkpoint == "FB237":
    save_path = str(PROJECT_ROOT / 'ckpt' / 'BiQUE_FB237')
    config["dataset"] = "FB237"
elif checkpoint == "YAGO3":
    save_path = str(PROJECT_ROOT / 'ckpt' / 'BiQUE_YAGO3-10')
    config["dataset"] = "YAGO3-10"
elif checkpoint == "CN100K":
    save_path = str(PROJECT_ROOT / 'ckpt' / 'BiQUE_Concept100k')
    config["dataset"] = "conceptnet-100k"
elif checkpoint == "ATOMIC":
    save_path = str(PROJECT_ROOT / 'ckpt' / 'BiQUE_Atomic')
    config["dataset"] = "Atomic"


args = dotdict(config)

dataset = Dataset(data_path, args.dataset)
examples = torch.from_numpy(dataset.get_train().astype('int64'))
exec('model = '+args.model+'(dataset.get_shape(), args.rank, args.init)')
exec('regularizer = '+args.regularizer+'(args.reg)')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
regularizer.to(device)
model = load_model(model, save_path)
_, test_metrics = evaluate_link_prediction(dataset, model, 'test', -1)
test = format_link_metrics(test_metrics)
print(test)