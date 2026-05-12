from typing import Any, Dict, Tuple

import numpy as np
import torch
import os


def ranking_metrics_from_ranks(
    ranks: torch.Tensor,
    at: Tuple[int, ...] = (1, 3, 10),
) -> Dict[str, Any]:
    ranks = ranks.float()
    return {
        "MR": torch.mean(ranks).item(),
        "MRR": torch.mean(1.0 / ranks).item(),
        "hits": torch.FloatTensor([
            torch.mean((ranks <= k).float()).item() for k in at
        ]),
    }


def average_link_prediction_metrics(side_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    keys = [k for k in ["lhs", "rhs"] if k in side_metrics]
    if not keys:
        return {"MR": 0.0, "MRR": 0.0, "hits@[1,3,10]": torch.zeros(3)}

    mr = float(np.mean([side_metrics[k]["MR"] for k in keys]))
    mrr = float(np.mean([side_metrics[k]["MRR"] for k in keys]))
    hits = torch.stack([side_metrics[k]["hits"] for k in keys], dim=0).mean(dim=0)
    return {"MR": mr, "MRR": mrr, "hits@[1,3,10]": hits}


def evaluate_link_prediction(
    dataset: Any,
    model: Any,
    split: str,
    n_queries: int = -1,
    missing_eval: str = "both",
    at: Tuple[int, ...] = (1, 3, 10),
):
    model.eval()
    device = next(model.parameters()).device
    test = dataset.get_examples(split)
    examples = torch.from_numpy(test.astype("int64")).to(device)

    missing = [missing_eval]
    if missing_eval == "both":
        missing = ["rhs", "lhs"]

    side_metrics: Dict[str, Dict[str, Any]] = {}
    for m in missing:
        q = examples.clone()
        if n_queries > 0:
            permutation = torch.randperm(len(examples), device=examples.device)[:n_queries]
            q = examples[permutation]

        if m == "lhs":
            tmp = torch.clone(q[:, 0])
            q[:, 0] = q[:, 2]
            q[:, 2] = tmp
            q[:, 1] += dataset.n_predicates // 2

        ranks = model.get_ranking(q, dataset.to_skip[m], batch_size=500)
        side_metrics[m] = ranking_metrics_from_ranks(ranks, at=at)

    return side_metrics, average_link_prediction_metrics(side_metrics)


def _trapz_area(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def roc_auc_score_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.int64)
    pos = float(np.sum(y_true == 1))
    neg = float(np.sum(y_true == 0))
    if pos == 0 or neg == 0:
        return float("nan")

    order = np.argsort(-y_score)
    y_sorted = y_true[order]

    tps = np.cumsum(y_sorted == 1)
    fps = np.cumsum(y_sorted == 0)

    tpr = np.concatenate(([0.0], tps / pos, [1.0]))
    fpr = np.concatenate(([0.0], fps / neg, [1.0]))
    return _trapz_area(fpr, tpr)


def pr_auc_score_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.int64)
    pos = float(np.sum(y_true == 1))
    if pos == 0:
        return float("nan")

    order = np.argsort(-y_score)
    y_sorted = y_true[order]

    tps = np.cumsum(y_sorted == 1)
    fps = np.cumsum(y_sorted == 0)

    precision = tps / np.maximum(tps + fps, 1)
    recall = tps / pos

    precision = np.concatenate(([1.0], precision))
    recall = np.concatenate(([0.0], recall))
    return _trapz_area(recall, precision)


def binary_classification_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    y_true = y_true.astype(np.int64)
    y_pred = (y_score >= threshold).astype(np.int64)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    total = max(len(y_true), 1)
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "Accuracy": float(accuracy),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1),
        "PR-AUC": pr_auc_score_binary(y_true, y_score),
        "ROC-AUC": roc_auc_score_binary(y_true, y_score),
    }


def _sample_negative_triples(triples: np.ndarray, n_entities: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    negatives = triples.copy()
    replace_head = rng.random(len(triples)) < 0.5

    random_entities = rng.integers(0, n_entities, size=len(triples))
    negatives[replace_head, 0] = random_entities[replace_head]
    negatives[~replace_head, 2] = random_entities[~replace_head]
    return negatives


def _score_triples(model: Any, triples: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    device = next(model.parameters()).device
    scores = []
    with torch.no_grad():
        for i in range(0, len(triples), batch_size):
            batch_np = triples[i:i + batch_size].astype("int64")
            batch = torch.from_numpy(batch_np).to(device)
            predictions, _ = model.forward(batch)
            tail_idx = batch[:, 2].unsqueeze(1)
            triple_scores = predictions.gather(1, tail_idx).squeeze(1)
            scores.append(triple_scores.detach().cpu().numpy())

    if not scores:
        return np.array([], dtype=np.float32)
    return np.concatenate(scores, axis=0)


def _best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray, num_candidates: int = 200) -> float:
    if len(y_score) == 0:
        return 0.0

    lo = float(np.min(y_score))
    hi = float(np.max(y_score))
    if lo == hi:
        return lo

    candidates = np.linspace(lo, hi, num=num_candidates)
    best_threshold = candidates[0]
    best_f1 = -1.0

    def f1_at_threshold(threshold: float) -> float:
        y_pred = (y_score >= threshold).astype(np.int64)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall == 0:
            return 0.0
        return float(2 * precision * recall / (precision + recall))

    for threshold in candidates:
        f1 = f1_at_threshold(float(threshold))
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)
    return best_threshold


def evaluate_triple_classification(
    dataset: Any,
    model: Any,
    split: str = "test",
    threshold_split: str = "valid",
    eval_batch_size: int = 1024,
) -> Dict[str, float]:
    pos_test = dataset.get_examples(split)
    neg_test = _sample_negative_triples(pos_test, dataset.n_entities, seed=123)

    y_true_test = np.concatenate(
        [np.ones(len(pos_test), dtype=np.int64), np.zeros(len(neg_test), dtype=np.int64)]
    )
    y_score_test = np.concatenate(
        [
            _score_triples(model, pos_test, batch_size=eval_batch_size),
            _score_triples(model, neg_test, batch_size=eval_batch_size),
        ]
    )

    pos_valid = dataset.get_examples(threshold_split)
    neg_valid = _sample_negative_triples(pos_valid, dataset.n_entities, seed=321)
    y_true_valid = np.concatenate(
        [np.ones(len(pos_valid), dtype=np.int64), np.zeros(len(neg_valid), dtype=np.int64)]
    )
    y_score_valid = np.concatenate(
        [
            _score_triples(model, pos_valid, batch_size=eval_batch_size),
            _score_triples(model, neg_valid, batch_size=eval_batch_size),
        ]
    )

    threshold = _best_f1_threshold(y_true_valid, y_score_valid)
    result = binary_classification_metrics(y_true_test, y_score_test, threshold=threshold)
    result["Threshold"] = float(threshold)
    return result


def has_labeled_triple_data(project_root: str, dataset_name: str) -> bool:
    label_dir = os.path.join(project_root, "src_data", f"{dataset_name}_w_labels")
    return os.path.exists(os.path.join(label_dir, "valid.txt")) and os.path.exists(
        os.path.join(label_dir, "test.txt")
    )


def _load_entity_map(project_root: str, dataset_name: str) -> Dict[str, int]:
    ent_file = os.path.join(project_root, "data", dataset_name, "ent_id")
    ent_map: Dict[str, int] = {}
    with open(ent_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                continue
            ent_map[parts[0]] = int(parts[1])
    return ent_map


def _read_labeled_split(project_root: str, dataset_name: str, split: str):
    label_file = os.path.join(project_root, "src_data", f"{dataset_name}_w_labels", f"{split}.txt")
    rows = []
    with open(label_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 4:
                continue
            h_raw, rel_name, t_raw, label = parts
            rows.append((h_raw, rel_name, t_raw, int(label)))
    return rows


def _infer_relation_name_to_id(dataset: Any, labeled_valid, labeled_test, ent_map: Dict[str, int]) -> Dict[str, int]:
    rel_map: Dict[str, int] = {}

    def build_pair_to_rels(split_np: np.ndarray) -> Dict[Tuple[int, int], set]:
        pair_to_rels: Dict[Tuple[int, int], set] = {}
        for h, r, t in split_np:
            key = (int(h), int(t))
            if key not in pair_to_rels:
                pair_to_rels[key] = set()
            pair_to_rels[key].add(int(r))
        return pair_to_rels

    valid_pair_to_rels = build_pair_to_rels(dataset.get_examples("valid"))
    test_pair_to_rels = build_pair_to_rels(dataset.get_examples("test"))

    for rows, pair_to_rels in [(labeled_valid, valid_pair_to_rels), (labeled_test, test_pair_to_rels)]:
        for h_raw, rel_name, t_raw, label in rows:
            if label != 1:
                continue
            if h_raw not in ent_map or t_raw not in ent_map:
                continue
            key = (ent_map[h_raw], ent_map[t_raw])
            rel_candidates = pair_to_rels.get(key, set())
            if len(rel_candidates) != 1:
                continue
            rel_id = next(iter(rel_candidates))
            if rel_name in rel_map and rel_map[rel_name] != rel_id:
                continue
            rel_map[rel_name] = rel_id

    # Fallback for WN18RR raw relation names used in labeled files.
    if not rel_map and hasattr(dataset, "n_predicates") and dataset.n_predicates // 2 >= 11:
        wn18rr_map = {
            "_also_see": 0,
            "_derivationally_related_form": 1,
            "_has_part": 2,
            "_hypernym": 3,
            "_instance_hypernym": 4,
            "_member_meronym": 5,
            "_member_of_domain_region": 6,
            "_member_of_domain_usage": 7,
            "_similar_to": 8,
            "_synset_domain_topic_of": 9,
            "_verb_group": 10,
        }
        rel_map.update(wn18rr_map)

    return rel_map


def _convert_labeled_rows(rows, ent_map: Dict[str, int], rel_map: Dict[str, int]):
    def resolve_entity(raw_id: str):
        if raw_id in ent_map:
            return ent_map[raw_id]
        normalized = raw_id.lstrip("0")
        if normalized == "":
            normalized = "0"
        return ent_map.get(normalized)

    triples = []
    labels = []
    for h_raw, rel_name, t_raw, label in rows:
        h_id = resolve_entity(h_raw)
        t_id = resolve_entity(t_raw)
        if h_id is None or t_id is None:
            continue
        if rel_name not in rel_map:
            continue
        triples.append([h_id, rel_map[rel_name], t_id])
        labels.append(label)

    if not triples:
        return np.zeros((0, 3), dtype=np.int64), np.zeros((0,), dtype=np.int64)
    return np.asarray(triples, dtype=np.int64), np.asarray(labels, dtype=np.int64)


def evaluate_triple_classification_labeled(
    dataset: Any,
    model: Any,
    project_root: str,
    dataset_name: str,
    eval_batch_size: int = 1024,
) -> Dict[str, float] | None:
    if not has_labeled_triple_data(project_root, dataset_name):
        return None

    ent_map = _load_entity_map(project_root, dataset_name)
    labeled_valid = _read_labeled_split(project_root, dataset_name, "valid")
    labeled_test = _read_labeled_split(project_root, dataset_name, "test")
    rel_map = _infer_relation_name_to_id(dataset, labeled_valid, labeled_test, ent_map)

    valid_triples, valid_labels = _convert_labeled_rows(labeled_valid, ent_map, rel_map)
    test_triples, test_labels = _convert_labeled_rows(labeled_test, ent_map, rel_map)
    if len(valid_triples) == 0 or len(test_triples) == 0:
        return None

    valid_scores = _score_triples(model, valid_triples, batch_size=eval_batch_size)
    test_scores = _score_triples(model, test_triples, batch_size=eval_batch_size)

    threshold = _best_f1_threshold(valid_labels, valid_scores)
    result = binary_classification_metrics(test_labels, test_scores, threshold=threshold)
    result["Threshold"] = float(threshold)
    result["NumValid"] = int(len(valid_labels))
    result["NumTest"] = int(len(test_labels))
    return result