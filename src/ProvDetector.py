import os
import random
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.metrics import ndcg_score
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 0) Utilities
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


def safe_auc(y_true, y_score):
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return roc_auc_score(y_true, y_score)


def safe_ap(y_true, y_score):
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return average_precision_score(y_true, y_score)


def best_f1_from_scores(y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    if len(np.unique(y_true)) < 2:
        return float("nan"), float("nan")

    thresholds = np.unique(y_score)
    best_f1 = -1.0
    best_thr = thresholds[0] if len(thresholds) > 0 else 0.5

    for thr in thresholds:
        y_pred = (y_score >= thr).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr

    return best_f1, best_thr


def compute_ndcg_all(y_true_binary, y_scores, k=None):
    y_true = np.asarray(y_true_binary, dtype=float).reshape(1, -1)
    y_pred = np.asarray(y_scores, dtype=float).reshape(1, -1)
    return ndcg_score(y_true, y_pred, k=k)


def minmax(x):
    x = np.asarray(x, dtype=float)
    if np.max(x) - np.min(x) < 1e-12:
        return np.zeros_like(x)
    return (x - np.min(x)) / (np.max(x) - np.min(x))


# ============================================================
# 1) Load data
# ============================================================

def load_binary_matrix_csv(input_csv, index_col=None):
    df = pd.read_csv(input_csv)

    if index_col is not None:
        df = pd.read_csv(input_csv, index_col=index_col)
    else:
        for cand in ["process_id", "uuid", "Object_ID", "subject_uuid", "pid"]:
            if cand in df.columns:
                df = df.set_index(cand)
                break
        else:
            df = df.set_index(df.columns[0])

    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    df = (df > 0).astype(np.float32)

    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    return df


def load_parent_matrix_csv(parent_csv, index_col=None):
    df = pd.read_csv(parent_csv)

    if index_col is not None:
        df = pd.read_csv(parent_csv, index_col=index_col)
    else:
        for cand in ["process_id", "uuid", "Object_ID", "subject_uuid", "pid"]:
            if cand in df.columns:
                df = df.set_index(cand)
                break
        else:
            df = df.set_index(df.columns[0])

    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    df = (df > 0).astype(np.float32)

    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    return df


def load_apt_ids(gt_csv):
    gt_df = pd.read_csv(gt_csv)

    if "label" in gt_df.columns and "uuid" in gt_df.columns:
        apt_ids = gt_df[
            gt_df["label"].astype(str).str.contains("AdmSubject::Node", na=False)
        ]["uuid"].astype(str).tolist()
    else:
        apt_ids = gt_df.iloc[:, 0].astype(str).tolist()

    return set(apt_ids)


# ============================================================
# 2) Parent / child structure
# ============================================================

def build_parent_child_dicts(df_parent, process_ids):
    process_set = set(process_ids)
    parent_of = {}
    children_of = {pid: [] for pid in process_ids}

    if df_parent is None:
        return parent_of, children_of

    common_children = [pid for pid in df_parent.index if pid in process_set]

    for child in common_children:
        row = df_parent.loc[child]
        active_parents = row[row == 1].index.tolist()
        active_parents = [p for p in active_parents if p in process_set]
        if len(active_parents) > 0:
            parent = active_parents[0]
            parent_of[child] = parent
            children_of[parent].append(child)

    return parent_of, children_of


# ============================================================
# 3) ProvDetector-style components
# ============================================================

def compute_action_rarity(df_binary, benign_mask=None):
    X = df_binary.values.astype(np.float32)

    if benign_mask is None or benign_mask.sum() == 0:
        X_ref = X
    else:
        X_ref = X[benign_mask]

    action_freq = X_ref.mean(axis=0)
    rarity = -np.log(action_freq + 1e-8)
    return rarity


def compute_rare_behavior_score(df_binary, rarity_weights):
    X = df_binary.values.astype(np.float32)
    return (X * rarity_weights.reshape(1, -1)).sum(axis=1)


def compute_transition_score(df_binary, parent_of):
    """
    Score high when process differs strongly from its parent.
    """
    process_ids = df_binary.index.tolist()
    pid_to_idx = {pid: i for i, pid in enumerate(process_ids)}
    X = df_binary.values.astype(np.float32)

    transition_score = np.zeros(len(process_ids), dtype=np.float32)

    for child, parent in parent_of.items():
        i = pid_to_idx[child]
        j = pid_to_idx[parent]

        child_vec = X[i].reshape(1, -1)
        parent_vec = X[j].reshape(1, -1)

        sim = cosine_similarity(child_vec, parent_vec)[0, 0]
        transition_score[i] = 1.0 - sim

    return transition_score


def propagate_suspicion(base_score, children_of, process_ids, num_iters=3, decay=0.5):
    """
    Diffuse suspiciousness along parent-child tree.
    """
    pid_to_idx = {pid: i for i, pid in enumerate(process_ids)}
    score = base_score.astype(np.float32).copy()

    for _ in range(num_iters):
        new_score = score.copy()
        for parent, childs in children_of.items():
            p_idx = pid_to_idx[parent]
            for child in childs:
                c_idx = pid_to_idx[child]
                # parent and child reinforce one another
                new_score[c_idx] += decay * score[p_idx]
                new_score[p_idx] += decay * score[c_idx]
        score = new_score

    return score


# ============================================================
# 4) Full ProvDetector-style scoring
# ============================================================

def provdetector_style_scores(
    df_binary,
    df_parent=None,
    apt_ids=None,
    alpha=0.5,   # rare behavior
    beta=0.3,    # transition anomaly
    gamma=0.2,   # propagation
    num_prop_iters=3,
    prop_decay=0.5
):
    process_ids = df_binary.index.tolist()
    n = len(process_ids)

    if apt_ids is not None:
        benign_mask = np.array([pid not in apt_ids for pid in process_ids], dtype=bool)
        if benign_mask.sum() == 0:
            benign_mask = np.ones(n, dtype=bool)
    else:
        benign_mask = np.ones(n, dtype=bool)

    parent_of, children_of = build_parent_child_dicts(df_parent, process_ids)

    rarity_weights = compute_action_rarity(df_binary, benign_mask=benign_mask)
    rare_behavior_score = compute_rare_behavior_score(df_binary, rarity_weights)
    transition_score = compute_transition_score(df_binary, parent_of)

    rare_behavior_score_n = minmax(rare_behavior_score)
    transition_score_n = minmax(transition_score)

    base_score = alpha * rare_behavior_score_n + beta * transition_score_n
    propagated_score = propagate_suspicion(
        base_score=base_score,
        children_of=children_of,
        process_ids=process_ids,
        num_iters=num_prop_iters,
        decay=prop_decay
    )
    propagated_score_n = minmax(propagated_score)

    final_score = (
        alpha * rare_behavior_score_n +
        beta * transition_score_n +
        gamma * propagated_score_n
    )

    parts = {
        "rare_behavior_score": rare_behavior_score,
        "transition_score": transition_score,
        "propagated_score": propagated_score,
        "rare_behavior_score_n": rare_behavior_score_n,
        "transition_score_n": transition_score_n,
        "propagated_score_n": propagated_score_n,
        "rarity_weights": rarity_weights,
        "parent_of": parent_of,
        "children_of": children_of
    }

    return final_score, parts


# ============================================================
# 5) End-to-end runner
# ============================================================

def run_provdetector_pipeline(
    input_csv,
    gt_csv,
    parent_csv=None,
    index_col=None,
    alpha=0.5,
    beta=0.3,
    gamma=0.2,
    num_prop_iters=3,
    prop_decay=0.5,
    ndcg_ks=(100, 500, 1000),
    seed=42
):
    set_seed(seed)

    df_binary = load_binary_matrix_csv(input_csv, index_col=index_col)
    df_parent = None
    if parent_csv is not None and os.path.exists(parent_csv):
        df_parent = load_parent_matrix_csv(parent_csv, index_col=index_col)

    apt_ids = load_apt_ids(gt_csv)
    process_ids = df_binary.index.tolist()
    y_true = np.array([1 if pid in apt_ids else 0 for pid in process_ids], dtype=int)

    scores, parts = provdetector_style_scores(
        df_binary=df_binary,
        df_parent=df_parent,
        apt_ids=apt_ids,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        num_prop_iters=num_prop_iters,
        prop_decay=prop_decay
    )

    auc = safe_auc(y_true, scores)
    ap = safe_ap(y_true, scores)
    best_f1, best_thr = best_f1_from_scores(y_true, scores)
    ndcg_all = compute_ndcg_all(y_true, scores, k=None)

    print(f"[ProvDetector-style] AUC       = {auc:.6f}")
    print(f"[ProvDetector-style] AP        = {ap:.6f}")
    print(f"[ProvDetector-style] Best F1   = {best_f1:.6f} (threshold={best_thr:.6f})")
    print(f"[ProvDetector-style] nDCG@all  = {ndcg_all:.6f}")

    for k in ndcg_ks:
        k_eff = min(int(k), len(y_true))
        nd = compute_ndcg_all(y_true, scores, k=k_eff)
        print(f"[ProvDetector-style] nDCG@{k_eff} = {nd:.6f}")

    df_rank = pd.DataFrame({
        "process_id": process_ids,
        "provdetector_score": scores,
        "rare_behavior_score": parts["rare_behavior_score_n"],
        "transition_score": parts["transition_score_n"],
        "propagated_score": parts["propagated_score_n"],
        "label_is_apt": y_true
    }).sort_values("provdetector_score", ascending=False).reset_index(drop=True)

    metrics = {
        "auc": auc,
        "ap": ap,
        "best_f1": best_f1,
        "best_f1_threshold": best_thr,
        "ndcg_all": ndcg_all,
    }

    return df_rank, metrics, parts


# ============================================================
# 6) Example usage
# ============================================================

if __name__ == "__main__":
    input_csv ="../clearscope/ProcessEvent.csv"

    gt_csv ="../clearscope/clearscope_bovia_lobiwapp.csv"

  
    
    parent_csv ="../clearscope/ProcessEvent.csv"

  
 
    df_rank, metrics, parts = run_provdetector_pipeline(
        input_csv=input_csv,
        gt_csv=gt_csv,
        parent_csv=parent_csv,
        index_col=None,
        alpha=0.5,
        beta=0.3,
        gamma=0.2,
        num_prop_iters=3,
        prop_decay=0.5,
        ndcg_ks=(100, 500, 1000),
        seed=42
    )

    print("\nTop 10 ranked processes:")
    print(df_rank.head(10))

    print("\nMetrics:")
    print(metrics)
