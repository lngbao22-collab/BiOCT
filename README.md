# Training Commands

This repository contains the training scripts and data helpers for the BiQUE experiments.

## Create a virtual environment

From the repository root on Windows, create and activate a `venv` with:

```bat
python -m venv venv
venv\Scripts\activate
```

Then install the libraries:

```bat
pip install -r requirements.txt
```

## Where to run

Run the commands from the repository root. The scripts now resolve `data/` and `logs/` from the repo root, so log files stay inside `logs/`.

If you want to force a specific GPU, set:

```bat
set CUDA_VISIBLE_DEVICES=0
```

## Common flags

`learn.py` supports these main options:

- `--dataset`: one of `WN18RR`, `FB237`, `YAGO3-10`, `Atomic`, `Concept100k`
- `--model`: model name, default `BiQUE`
- `--regularizer`: default `wN3`
- `--optimizer`: default `Adagrad`
- `--rank`: embedding size
- `--batch_size`: training batch size
- `--reg`: regularization strength
- `--learning_rate`: optimizer learning rate
- `-train`: enable training
- `-test`: run evaluation
- `-save`: save checkpoint and config under `logs/`
- `-weight`: use dataset cross-entropy weights
- `-id`: model run id used in the save folder name

## Existing run commands

### FB237

```bat
python codes\learn.py --dataset FB237 --model BiQUE --rank 500 ^
        --optimizer Adagrad --learning_rate 1e-1 --batch_size 500 ^
        --regularizer N3 --reg 0.0 --max_epochs 200 --valid 5 -train -id 0 -save
```

### WN18RR

```bat
python codes\learn.py --dataset WN18RR --model BiQUE --rank 500 ^
        --optimizer Adagrad --learning_rate 1e-1 --batch_size 300 ^
        --regularizer N3 --reg 1.5e-1 --max_epochs 100 --valid 5 -train -id 0 -save -weight
```

### YAGO3-10

```bat
python codes\learn.py --dataset YAGO3-10 --model BiQUE --rank 128 ^
        --optimizer Adagrad --learning_rate 1e-1 --batch_size 1000 ^
        --regularizer N3 --reg 5e-3 --max_epochs 200 --valid 5 -train -id 0 -save
```

### Concept100k

```bat
python codes\learn.py --dataset Concept100k --model BiQUE --rank 500 ^
        --optimizer Adagrad --learning_rate 1e-1 --batch_size 5000 ^
        --regularizer N3 --reg 1e-1 --max_epochs 200 --valid 5 -train -id 0 -save
```

### Atomic

```bat
python codes\learn.py --dataset Atomic --model BiQUE --rank 128 ^
        --optimizer Adagrad --learning_rate 1e-1 --batch_size 2000 ^
        --regularizer N3 --reg 5e-3 --max_epochs 200 --valid 5 -train -id 0 -save
```

## Notes

- The save directory defaults to `logs/`.
- The save folder name is built from the model, dataset, regularizer, batch size, rank, regularization value, learning rate, and run id.
- The WN18RR example above uses `-weight` and writes `train.log` plus the checkpoint files into `logs/`.