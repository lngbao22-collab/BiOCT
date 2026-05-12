import os
import json
import torch
import argparse
import warnings
import numpy as np
from torch import optim
from datasets import Dataset
from metrics import evaluate_link_prediction, evaluate_triple_classification_labeled

from models import *
from regularizers import *
from optimizers import KBCOptimizer

# Disable sparse tensor invariant checks for performance (BiQUE uses sparse embeddings)
torch.sparse.check_sparse_tensor_invariants(False)

# Suppress the sparse invariant checks warning
warnings.filterwarnings("ignore", message=".*Sparse invariant checks are implicitly disabled.*")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DEFAULT_LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

datasets = ['WN18RR', 'FB237', 'YAGO3-10', 'Atomic', 'Concept100k']
optimizers = ['Adagrad']

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', choices=datasets, help="Dataset in {}".format(datasets))
parser.add_argument('--model', type=str, default='BiQUE')
parser.add_argument('--regularizer', type=str, default='wN3')
parser.add_argument('--optimizer', choices=optimizers, default='Adagrad', help="Optimizer in {}".format(optimizers))
parser.add_argument('--max_epochs', default=200, type=int, help="Number of epochs.")
parser.add_argument('--valid', default=5, type=float, help="Number of epochs before valid.")
parser.add_argument('--rank', default=128, type=int, help="component size.")
parser.add_argument('--batch_size', default=500, type=int, help="batch_size")
parser.add_argument('--reg', default=0, type=float, help="Regularization factor")
parser.add_argument('--init', default=1e-3, type=float, help="Initial scale")
parser.add_argument('--learning_rate', default=1e-1, type=float, help="Learning rate")
parser.add_argument('-train', '--do_train', action='store_true')
parser.add_argument('-test', '--do_test', action='store_true')
parser.add_argument('-save', '--do_save', action='store_true')
parser.add_argument('-weight', '--do_ce_weight', action='store_true')
parser.add_argument('-path', '--save_path', type=str, default=DEFAULT_LOG_DIR)
parser.add_argument('-id', '--model_id', type=str, default='0')
parser.add_argument('-ckpt', '--checkpoint', type=str, default='')
parser.add_argument('--task', choices=['link_prediction', 'triple_classification', 'both'], default='link_prediction')
parser.add_argument('--eval_batch_size', default=1024, type=int, help='Batch size for evaluation scoring')

args = parser.parse_args()


def save_model(model, save_path):
    torch.save(model.state_dict(), os.path.join(save_path, 'checkpoint'))
    h1 = model.h1
    h2 = model.h2
    r1 = model.r1
    r2 = model.r2
    np.save(os.path.join(save_path, 'entity_embedding1.npy'), h1.weight.detach().cpu().numpy())
    np.save(os.path.join(save_path, 'entity_embedding2.npy'), h2.weight.detach().cpu().numpy())
    np.save(os.path.join(save_path, 'relation_embedding1.npy'), r1.weight.detach().cpu().numpy())
    np.save(os.path.join(save_path, 'relation_embedding2.npy'), r2.weight.detach().cpu().numpy())

def load_model(model, save_path):
    state = torch.load(os.path.join(save_path, 'checkpoint'))
    model.load_state_dict(state)
    return model

save_path = args.save_path
if args.do_save:
    assert args.save_path
    save_suffix = f"{args.model}_{args.dataset}_{args.regularizer}_{args.batch_size}_{args.rank}_{args.reg}_{args.learning_rate}_{args.model_id}"

    os.makedirs(args.save_path, exist_ok=True)

    save_path = os.path.join(args.save_path, save_suffix)
    os.makedirs(save_path, exist_ok=True)

    with open(os.path.join(save_path, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)


data_path = DATA_DIR
dataset = Dataset(data_path, args.dataset)
examples = torch.from_numpy(dataset.get_train().astype('int64'))

if args.do_ce_weight:
    ce_weight = torch.Tensor(dataset.get_weight()).to(DEVICE)
else:
    ce_weight = None

print(dataset.get_shape())

model = None
regularizer = None
exec('model = '+args.model+'(dataset.get_shape(), args.rank, args.init)')
exec('regularizer = '+args.regularizer+'(args.reg)')

model.to(DEVICE)
regularizer.to(DEVICE)

optim_method = {
    'Adagrad': lambda: optim.Adagrad(model.parameters(), lr=args.learning_rate),
}[args.optimizer]()

optimizer = KBCOptimizer(model, regularizer, optim_method, args.batch_size)


def run_link_prediction_eval(split: str):
    _, metrics = evaluate_link_prediction(dataset, model, split, -1)
    return metrics


def run_triple_classification_eval(split: str = 'test'):
    return evaluate_triple_classification_labeled(
        dataset,
        model,
        project_root=PROJECT_ROOT,
        dataset_name=args.dataset,
        eval_batch_size=args.eval_batch_size,
    )


cur_loss = 0

if args.checkpoint != '':
    model.load_state_dict(torch.load(os.path.join(args.checkpoint, 'checkpoint'), map_location=DEVICE))

if args.do_test:
    if args.task in ['link_prediction', 'both']:
        test = run_link_prediction_eval('test')
        print(f"\t LINK PREDICTION TEST : {test} ")
    if args.task in ['triple_classification', 'both']:
        tc_test = run_triple_classification_eval('test')
        if tc_test is None:
            print(f"\t TRIPLE CLASSIFICATION TEST : skipped (missing or incompatible labeled data in src_data/{args.dataset}_w_labels)")
        else:
            print(f"\t TRIPLE CLASSIFICATION TEST : {tc_test} ")
    print("\t ============================================")

best_valid_mrr = 0.0
best_epc = 0
if args.do_train:
    with open(os.path.join(save_path, 'train.log'), 'w') as log_file:
        for e in range(args.max_epochs):
            print("Epoch: {}".format(e+1))

            cur_loss = optimizer.epoch(examples, e=e, weight=ce_weight)

            if (e + 1) % args.valid == 0:
                valid = run_link_prediction_eval('valid')
                print("\t VALID (LINK PREDICTION): ", valid)

                log_file.write("Epoch: {}\n".format(e+1))
                log_file.write("\t VALID: {}\n".format(valid))

                log_file.flush()

                if valid["MRR"] > best_valid_mrr:
                    save_model(model, save_path)
                    best_valid_mrr = valid['MRR']
                    best_epc = e


        checkpoint_path = os.path.join(save_path, 'checkpoint')
        if os.path.exists(checkpoint_path):
            model = load_model(model, save_path)
        test = run_link_prediction_eval('test')
        print(f"\t BEST VALID MRR : {best_valid_mrr}, IN EPOCH : {best_epc}")
        print(f"\t LINK PREDICTION TEST : {test} ")
        if args.task in ['triple_classification', 'both']:
            tc_test = run_triple_classification_eval('test')
            if tc_test is None:
                print(f"\t TRIPLE CLASSIFICATION TEST : skipped (missing or incompatible labeled data in src_data/{args.dataset}_w_labels)")
            else:
                print(f"\t TRIPLE CLASSIFICATION TEST : {tc_test} ")
        print("\t ============================================")

        log_file.write(f"\t BEST VALID MRR : {best_valid_mrr}, IN EPOCH : {best_epc}\n")
        log_file.write(f"\t LINK PREDICTION TEST : {test} \n")
        if args.task in ['triple_classification', 'both']:
            if tc_test is None:
                log_file.write(f"\t TRIPLE CLASSIFICATION TEST : skipped (missing or incompatible labeled data in src_data/{args.dataset}_w_labels)\n")
            else:
                log_file.write(f"\t TRIPLE CLASSIFICATION TEST : {tc_test} \n")
        log_file.write("\t ============================================\n")
