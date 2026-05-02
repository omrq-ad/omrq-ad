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
# 2) Parent mapping
# rows = child, cols = parent, 1 means parent->child relation
# ============================================================

def build_parent_dict(df_parent, process_ids):
    process_set = set(process_ids)
    parent_dict = {}

    if df_parent is None:
        return parent_dict

    common_children = [pid for pid in df_parent.index if pid in process_set]

    for child in common_children:
        row = df_parent.loc[child]
        active_parents = row[row == 1].index.tolist()
        active_parents = [p for p in active_parents if p in process_set]
        if len(active_parents) > 0:
            # if multiple, keep the first
            parent_dict[child] = active_parents[0]

    return parent_dict


# ============================================================
# 3) WATSON-style scoring
#    score = rarity + deviation + parent inconsistency
# ============================================================

def watson_style_scores(df_binary, df_parent=None, apt_ids=None,
                        alpha=0.4, beta=0.4, gamma=0.2):
    """
    alpha: rarity weight
    beta : deviation-from-normal weight
    gamma: parent inconsistency weight
    """
    process_ids = df_binary.index.tolist()
    X = df_binary.values.astype(np.float32)
    n, d = X.shape

    # --------------------------------------------------------
    # A) Build benign mask for prototype estimation
    # --------------------------------------------------------
    if apt_ids is not None:
        benign_mask = np.array([pid not in apt_ids for pid in process_ids], dtype=bool)
        if benign_mask.sum() == 0:
            benign_mask = np.ones(n, dtype=bool)
    else:
        benign_mask = np.ones(n, dtype=bool)

    X_benign = X[benign_mask]

    # --------------------------------------------------------
    # B) Action rarity: inverse document frequency
    # --------------------------------------------------------
    action_freq = X_benign.mean(axis=0)  # frequency in benign
    rarity = -np.log(action_freq + 1e-8)  # high if rare

    # score per process = weighted sum of active rare actions
    rarity_score = (X * rarity.reshape(1, -1)).sum(axis=1)

    # --------------------------------------------------------
    # C) Deviation from normal prototype
    # --------------------------------------------------------
    centroid = X_benign.mean(axis=0, keepdims=True)  # benign prototype
    deviation_score = np.linalg.norm(X - centroid, axis=1)

    # --------------------------------------------------------
    # D) Parent inconsistency
    # --------------------------------------------------------
    parent_dict = build_parent_dict(df_parent, process_ids)
    pid_to_idx = {pid: i for i, pid in enumerate(process_ids)}

    parent_inconsistency = np.zeros(n, dtype=np.float32)

    for pid, i in pid_to_idx.items():
        if pid in parent_dict:
            parent_pid = parent_dict[pid]
            j = pid_to_idx[parent_pid]
            # one minus cosine similarity
            xi = X[i].reshape(1, -1)
            xj = X[j].reshape(1, -1)
            sim = cosine_similarity(xi, xj)[0, 0]
            parent_inconsistency[i] = 1.0 - sim
        else:
            parent_inconsistency[i] = 0.0

    # --------------------------------------------------------
    # E) Normalize and combine
    # --------------------------------------------------------
    rarity_score_n = minmax(rarity_score)
    deviation_score_n = minmax(deviation_score)
    parent_inconsistency_n = minmax(parent_inconsistency)

    final_score = (
        alpha * rarity_score_n +
        beta  * deviation_score_n +
        gamma * parent_inconsistency_n
    )

    parts = {
        "rarity_score": rarity_score,
        "deviation_score": deviation_score,
        "parent_inconsistency": parent_inconsistency,
        "rarity_score_n": rarity_score_n,
        "deviation_score_n": deviation_score_n,
        "parent_inconsistency_n": parent_inconsistency_n,
        "centroid": centroid.squeeze()
    }

    return final_score, parts


# ============================================================
# 4) End-to-end runner
# ============================================================

def run_watson_pipeline(
    input_csv,
    gt_csv,
    parent_csv=None,
    index_col=None,
    alpha=0.4,
    beta=0.4,
    gamma=0.2,
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

    scores, parts = watson_style_scores(
        df_binary=df_binary,
        df_parent=df_parent,
        apt_ids=apt_ids,
        alpha=alpha,
        beta=beta,
        gamma=gamma
    )

    auc = safe_auc(y_true, scores)
    ap = safe_ap(y_true, scores)
    best_f1, best_thr = best_f1_from_scores(y_true, scores)
    ndcg_all = compute_ndcg_all(y_true, scores, k=None)

    print(f"[WATSON-style] AUC       = {auc:.6f}")
    print(f"[WATSON-style] AP        = {ap:.6f}")
    print(f"[WATSON-style] Best F1   = {best_f1:.6f} (threshold={best_thr:.6f})")
    print(f"[WATSON-style] nDCG@all  = {ndcg_all:.6f}")

    for k in ndcg_ks:
        k_eff = min(int(k), len(y_true))
        nd = compute_ndcg_all(y_true, scores, k=k_eff)
        print(f"[WATSON-style] nDCG@{k_eff} = {nd:.6f}")

    df_rank = pd.DataFrame({
        "process_id": process_ids,
        "watson_score": scores,
        "rarity_score": parts["rarity_score_n"],
        "deviation_score": parts["deviation_score_n"],
        "parent_inconsistency": parts["parent_inconsistency_n"],
        "label_is_apt": y_true
    }).sort_values("watson_score", ascending=False).reset_index(drop=True)

    metrics = {
        "auc": auc,
        "ap": ap,
        "best_f1": best_f1,
        "best_f1_threshold": best_thr,
        "ndcg_all": ndcg_all,
    }

    return df_rank, metrics, parts


# ============================================================
# 5) Example usage
# ============================================================

if __name__ == "__main__":
    input_csv ="../clearscope/ProcessEvent.csv"

   
    gt_csv = "../clearscope/clearscope_bovia_lobiwapp.csv"

   
    parent_csv ="../clearscope/ProcessEvent.csv"

   
   

    df_rank, metrics, parts = run_watson_pipeline(
        input_csv=input_csv,
        gt_csv=gt_csv,
        parent_csv=parent_csv,
        index_col=None,
        alpha=0.4,
        beta=0.4,
        gamma=0.2,
        ndcg_ks=(100, 500, 1000),
        seed=42
    )

    print("\nTop 10 ranked processes:")
    print(df_rank.head(10))

    print("\nMetrics:")
    print(metrics)
