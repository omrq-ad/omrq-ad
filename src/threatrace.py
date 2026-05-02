import os
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.metrics import ndcg_score

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SAGEConv


# ============================================================
# 0) Reproducibility
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
# 2) Build THREATRACE-style process graph
#    Nodes = processes
#    Features = binary action vectors
#    Edges = parent/spawn + optional similarity graph
# ============================================================

def build_process_graph(df_binary, df_parent=None, k_similarity=10, add_similarity=True):
    process_ids = df_binary.index.tolist()
    n = len(process_ids)
    pid_to_idx = {pid: i for i, pid in enumerate(process_ids)}

    x = torch.tensor(df_binary.values, dtype=torch.float32)

    edges = set()

    # --------------------------------------------------------
    # A) Spawn / parent edges
    # df_parent: rows = child process, cols = parent process, 1 means child-of
    # --------------------------------------------------------
    if df_parent is not None:
        common_rows = [pid for pid in df_parent.index if pid in pid_to_idx]
        common_cols = [pid for pid in df_parent.columns if pid in pid_to_idx]

        for child_pid in common_rows:
            row = df_parent.loc[child_pid]
            active_parents = row[row == 1].index.tolist()
            for parent_pid in active_parents:
                if parent_pid in pid_to_idx:
                    c = pid_to_idx[child_pid]
                    p = pid_to_idx[parent_pid]
                    edges.add((p, c))  # parent -> child
                    edges.add((c, p))  # make bidirectional for message passing

    # --------------------------------------------------------
    # B) Similarity edges (kNN over binary action vectors)
    # --------------------------------------------------------
    if add_similarity and n > 1:
        feats = df_binary.values
        k_eff = min(k_similarity + 1, n)
        nbrs = NearestNeighbors(n_neighbors=k_eff, metric="cosine")
        nbrs.fit(feats)
        distances, indices = nbrs.kneighbors(feats)

        for i in range(n):
            for j in indices[i][1:]:  # skip self
                edges.add((i, j))
                edges.add((j, i))

    if len(edges) == 0:
        # fallback self-loops if graph is empty
        for i in range(n):
            edges.add((i, i))

    edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index), process_ids


# ============================================================
# 3) Labels and splits
# ============================================================

def build_labels(process_ids, apt_ids):
    y = np.array([1 if pid in apt_ids else 0 for pid in process_ids], dtype=np.int64)
    return torch.tensor(y, dtype=torch.long)


def build_masks(y, train_ratio=0.6, val_ratio=0.2, seed=42):
    """
    Stratified split over all labeled nodes.
    Since DARPA labels are available only for evaluation in your unsupervised work,
    this supervised THREATRACE-style baseline is a separate labeled baseline.
    """
    idx = np.arange(len(y))
    y_np = y.cpu().numpy()

    train_idx, temp_idx = train_test_split(
        idx, test_size=(1 - train_ratio), stratify=y_np, random_state=seed
    )

    val_rel = val_ratio / (1 - train_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=(1 - val_rel), stratify=y_np[temp_idx], random_state=seed
    )

    train_mask = torch.zeros(len(y), dtype=torch.bool)
    val_mask = torch.zeros(len(y), dtype=torch.bool)
    test_mask = torch.zeros(len(y), dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    return train_mask, val_mask, test_mask


# ============================================================
# 4) THREATRACE-style GNN classifier
# ============================================================

class ThreatRaceGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, out_dim=2, dropout=0.2):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.lin = nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        logits = self.lin(x)
        return logits


class ThreatRaceSAGE(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, out_dim=2, dropout=0.2):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.lin = nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        logits = self.lin(x)
        return logits


# ============================================================
# 5) Metrics
# ============================================================

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
    y_true = np.asarray(y_true)
    thresholds = np.unique(y_score)
    best = 0.0
    for t in thresholds:
        pred = (y_score >= t).astype(int)
        best = max(best, f1_score(y_true, pred, zero_division=0))
    return best


def compute_ndcg_all(y_true, y_score):
    return ndcg_score(
        np.asarray(y_true, dtype=float).reshape(1, -1),
        np.asarray(y_score, dtype=float).reshape(1, -1)
    )


# ============================================================
# 6) Train / evaluate
# ============================================================

def train_threatrace(
    data,
    y,
    train_mask,
    val_mask,
    model_type="sage",
    hidden_dim=64,
    lr=1e-3,
    weight_decay=1e-4,
    epochs=200,
    patience=20,
    device="cpu"
):
    data = data.to(device)
    y = y.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)

    in_dim = data.x.size(1)
    if model_type.lower() == "gcn":
        model = ThreatRaceGCN(in_dim, hidden_dim=hidden_dim).to(device)
    else:
        model = ThreatRaceSAGE(in_dim, hidden_dim=hidden_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_auc = -1.0
    best_state = None
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        logits = model(data.x, data.edge_index)
        loss = F.cross_entropy(logits[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            probs = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            y_val = y[val_mask].detach().cpu().numpy()
            p_val = probs[val_mask.detach().cpu().numpy()]
            val_auc = safe_auc(y_val, p_val)

        if np.isnan(val_auc):
            val_auc = -1.0

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        if epoch % 20 == 0 or epoch == 1:
            print(f"[Epoch {epoch:03d}] loss={loss.item():.4f} val_auc={val_auc:.4f}")

        if bad_epochs >= patience:
            print(f"[Early stop] epoch={epoch}, best_val_auc={best_val_auc:.4f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def evaluate_threatrace(model, data, y, mask=None, device="cpu"):
    data = data.to(device)
    y = y.to(device)

    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        probs = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()

    y_np = y.detach().cpu().numpy()

    if mask is not None:
        mask_np = mask.detach().cpu().numpy()
        y_eval = y_np[mask_np]
        p_eval = probs[mask_np]
    else:
        y_eval = y_np
        p_eval = probs

    auc = safe_auc(y_eval, p_eval)
    ap = safe_ap(y_eval, p_eval)
    best_f1 = best_f1_from_scores(y_eval, p_eval)
    ndcg_all = compute_ndcg_all(y_eval, p_eval)

    return {
        "auc": auc,
        "ap": ap,
        "best_f1": best_f1,
        "ndcg_all": ndcg_all,
        "scores": p_eval,
        "labels": y_eval
    }


# ============================================================
# 7) End-to-end runner
# ============================================================

def run_threatrace_pipeline(
    input_csv,
    gt_csv,
    parent_csv=None,
    index_col=None,
    model_type="sage",
    hidden_dim=64,
    k_similarity=10,
    add_similarity=True,
    train_ratio=0.6,
    val_ratio=0.2,
    seed=42,
    device=None
):
    set_seed(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[INFO] device={device}")

    df_binary = load_binary_matrix_csv(input_csv, index_col=index_col)
    df_parent = None
    if parent_csv is not None and os.path.exists(parent_csv):
        df_parent = load_parent_matrix_csv(parent_csv, index_col=index_col)

    apt_ids = load_apt_ids(gt_csv)

    data, process_ids = build_process_graph(
        df_binary=df_binary,
        df_parent=df_parent,
        k_similarity=k_similarity,
        add_similarity=add_similarity
    )

    y = build_labels(process_ids, apt_ids)
    train_mask, val_mask, test_mask = build_masks(
        y,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed
    )

    print(f"[INFO] #nodes={len(process_ids)}")
    print(f"[INFO] positives={int(y.sum())}, negatives={int((y==0).sum())}")
    print(f"[INFO] train={int(train_mask.sum())}, val={int(val_mask.sum())}, test={int(test_mask.sum())}")

    model = train_threatrace(
        data=data,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        model_type=model_type,
        hidden_dim=hidden_dim,
        device=device
    )

    res_test = evaluate_threatrace(model, data, y, mask=test_mask, device=device)
    res_all = evaluate_threatrace(model, data, y, mask=None, device=device)

    print("\n[TEST]")
    print(f"AUC       = {res_test['auc']:.4f}")
    print(f"AP        = {res_test['ap']:.4f}")
    print(f"Best F1   = {res_test['best_f1']:.4f}")
    print(f"nDCG@all  = {res_test['ndcg_all']:.4f}")

    print("\n[ALL NODES]")
    print(f"AUC       = {res_all['auc']:.4f}")
    print(f"AP        = {res_all['ap']:.4f}")
    print(f"Best F1   = {res_all['best_f1']:.4f}")
    print(f"nDCG@all  = {res_all['ndcg_all']:.4f}")

    # ranked dataframe on all nodes
    ranked_df = pd.DataFrame({
        "process_id": process_ids,
        "threatrace_score": res_all["scores"],
        "label": res_all["labels"]
    }).sort_values("threatrace_score", ascending=False).reset_index(drop=True)

    return model, ranked_df, data, y, train_mask, val_mask, test_mask


# ============================================================
# 8) Example usage
# ============================================================

if __name__ == "__main__":
  
    parent_csv ="../ProcessEvent.csv"


    input_csv = "../ProcessEvent.csv"

   
    gt_csv = "../clearscope_pandex_merged.csv"

   


    model, ranked_df, data, y, train_mask, val_mask, test_mask = run_threatrace_pipeline(
        input_csv=input_csv,
        gt_csv=gt_csv,
        parent_csv=parent_csv,     # set to None if unavailable
        index_col=None,
        model_type="sage",         # "sage" or "gcn"
        hidden_dim=64,
        k_similarity=10,
        add_similarity=True,
        train_ratio=0.6,
        val_ratio=0.2,
        seed=42
    )

    print("\nTop 10 ranked processes:")
    print(ranked_df.head(10))
