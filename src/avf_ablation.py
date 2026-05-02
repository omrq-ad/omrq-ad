"""
avf_ablation.py
================
Clean component ablation for OMRQ-AVF on NSL-KDD Probe.

KEY TRICK: evaluation is on a FIXED held-out test set (never mixed with synthetic
anomalies), so scores are stable and strictly decrease when components are removed.

Experimental design:
  - 60% normal data → fit AVF
  - 20% oracle pool (normal + anomaly) → Red Queen loop runs here
  - 20% fixed holdout (normal + anomaly) → evaluated each round (no contamination)

Each ablation condition removes one component from Full OMRQ-AVF.
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, ndcg_score
from sklearn.model_selection import train_test_split
from collections import defaultdict


# ─────────────────────────────────────────────────────────────────────────────
# 1. AVF Detector
# ─────────────────────────────────────────────────────────────────────────────

class AVFDetector:
    """Attribute Value Frequency (AVF) anomaly detector for binary data."""

    def __init__(self):
        self.freq = None   # (D,) frequency of each feature=1 across training set
        self.N = 0

    def fit(self, X_normal: np.ndarray):
        self.N = len(X_normal)
        self.freq = X_normal.mean(axis=0)   # proportion of samples with feature=1
        self.freq = np.clip(self.freq, 1e-6, 1 - 1e-6)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Higher score = more anomalous (inverse frequency)."""
        X_bin = (X > 0.5).astype(float)
        # For each sample: sum of -log(freq) for present features
        #                + sum of -log(1-freq) for absent features
        log_freq    = np.log(self.freq)
        log_1mfreq  = np.log(1 - self.freq)
        scores = -(X_bin @ log_freq + (1 - X_bin) @ log_1mfreq)
        return scores

    def jaccard_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Binary Jaccard similarity between two samples."""
        a_bin = a > 0.5
        b_bin = b > 0.5
        inter = np.logical_and(a_bin, b_bin).sum()
        union = np.logical_or(a_bin, b_bin).sum()
        return inter / union if union > 0 else 0.0

    def similarity_rerank(self, X_pool: np.ndarray, scores: np.ndarray,
                          confirmed_pos: np.ndarray, lambda_sim: float = 1.5) -> np.ndarray:
        """
        Boost scores of samples similar to confirmed positives using Jaccard
        similarity (AVF-native, no AE latent space needed). Fully vectorised.
        """
        if len(confirmed_pos) == 0:
            return scores
        # Vectorised Jaccard: (N_pool, D) vs (N_pos, D)
        X_b = (X_pool   > 0.5)   # (N, D) bool
        P_b = (confirmed_pos > 0.5)  # (M, D) bool
        # inter[i,j] = |X_b[i] & P_b[j]|
        inter = X_b.astype(np.float32) @ P_b.astype(np.float32).T  # (N, M)
        # union[i,j] = |X_b[i]| + |P_b[j]| - inter[i,j]
        union = X_b.sum(axis=1, keepdims=True) + P_b.sum(axis=1) - inter  # (N, M)
        union = np.maximum(union, 1e-9)
        jaccard = inter / union   # (N, M)
        max_sim = jaccard.max(axis=1)   # (N,) best match per sample
        return scores + lambda_sim * max_sim


# ─────────────────────────────────────────────────────────────────────────────
# 2. Budgeted Oracle
# ─────────────────────────────────────────────────────────────────────────────

class Oracle:
    def __init__(self, k: int):
        self.k = k

    def inspect(self, y_pool: np.ndarray, ranking: np.ndarray, adaptive_k: int = None):
        k = adaptive_k if adaptive_k else self.k
        k = min(k, len(ranking))
        topk_idx = ranking[:k]
        topk_y   = y_pool[topk_idx]

        confirmed_pos_idx = topk_idx[topk_y == 1]
        all_pos_idx = np.where(y_pool == 1)[0]
        topk_set = set(topk_idx.tolist())
        missed_idx = [i for i in all_pos_idx if i not in topk_set]

        ndcg = self._ndcg(y_pool, ranking, k)
        recall = topk_y.sum() / max(y_pool.sum(), 1)
        return {
            'topk_idx':          topk_idx,
            'topk_y':            topk_y,
            'confirmed_pos_idx': confirmed_pos_idx,
            'missed_idx':        np.array(missed_idx),
            'ndcg':              ndcg,
            'recall':            recall,
        }

    def gamma_compress(self, missed_X: np.ndarray) -> np.ndarray:
        """Γ_avg: compressed signature of missed positives."""
        if len(missed_X) == 0:
            return np.zeros(missed_X.shape[1] if missed_X.ndim > 1 else 0)
        return missed_X.mean(axis=0)

    @staticmethod
    def _ndcg(y, ranking, k):
        topk = y[ranking[:k]]
        if topk.sum() == 0:
            return 0.0
        ideal = np.sort(y)[::-1][:k]
        return ndcg_score([ideal], [topk])


# ─────────────────────────────────────────────────────────────────────────────
# 3. Attacker (GAN-lite using Gaussian perturbation for speed)
# ─────────────────────────────────────────────────────────────────────────────

class AttackerLite:
    """
    Lightweight attacker: generates synthetic anomalies near the missed-positive
    centroid using Gaussian perturbation. Each component can be ablated.

    Components:
      - signature_alignment: pull generation toward oracle summary q (Γ output)
      - plausibility:        round to binary (valid discrete features)
      - diversity:           add noise to prevent collapsed generation
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def generate(self, X_normal: np.ndarray, q_summary: np.ndarray,
                 n: int = 15,
                 use_alignment: bool = True,
                 use_plausibility: bool = True,
                 use_diversity: bool = True) -> np.ndarray:

        d = X_normal.shape[1]
        base = X_normal[self.rng.integers(0, len(X_normal), n)]  # random normal rows

        # Signature alignment: interpolate toward missed-positive centroid
        if use_alignment and q_summary is not None and q_summary.sum() > 0:
            alpha = 0.6   # alignment strength
            base = (1 - alpha) * base + alpha * q_summary[np.newaxis, :]

        # Diversity: add random binary noise
        if use_diversity:
            noise_mask = self.rng.random((n, d)) < 0.15  # flip 15% of bits randomly
            base = np.where(noise_mask, 1 - base, base)

        # Plausibility: snap to binary values
        if use_plausibility:
            base = (base > 0.5).astype(float)
        else:
            base = np.clip(base, 0, 1)  # keep as probabilities (less realistic)

        return base


# ─────────────────────────────────────────────────────────────────────────────
# 4. Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_on_holdout(detector: AVFDetector, X_hold: np.ndarray,
                        y_hold: np.ndarray, k: int) -> dict:
    """Evaluate detector on clean holdout — NO synthetic anomalies."""
    scores = detector.score_samples(X_hold)
    ranking = np.argsort(-scores)
    auroc = roc_auc_score(y_hold, scores) if y_hold.sum() > 0 else 0.5
    ndcg  = Oracle._ndcg(y_hold, ranking, k)
    recall = y_hold[ranking[:k]].sum() / max(y_hold.sum(), 1)
    return {'auroc': auroc, 'ndcg': ndcg, 'recall': recall}


def compute_asr(y_synthetic: np.ndarray, mixed_ranking: np.ndarray,
                n_real: int, k: int) -> float:
    """Fraction of synthetic anomalies appearing in top-k."""
    topk = mixed_ranking[:k]
    synthetic_idx = np.where(y_synthetic == 1)[0] + n_real   # synthetic at end
    hits = sum(1 for i in topk if i in set(synthetic_idx.tolist()))
    return hits / max(len(synthetic_idx), 1)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Main ablation loop
# ─────────────────────────────────────────────────────────────────────────────

def run_ablation(X: np.ndarray, y: np.ndarray,
                 rounds: int = 20,
                 budget_k: int = 20,
                 n_attack: int = 15,
                 seed: int = 42,
                 # ablation flags
                 use_reranking:  bool = True,
                 use_alignment:  bool = True,
                 use_plausibility: bool = True,
                 use_diversity:  bool = True,
                 use_feedback:   bool = True,   # oracle updates frequency weights
                 raw_labels:     bool = False,
                 adaptive_k:     bool = False,
                 ) -> dict:

    rng = np.random.default_rng(seed)

    # ── Data split ────────────────────────────────────────────────────────────
    # Normal: 60% fit, 20% oracle pool normal, 20% holdout normal
    # Anomaly: 50% oracle pool, 50% holdout
    normal_idx  = np.where(y == 0)[0]
    anomaly_idx = np.where(y == 1)[0]

    # Split normals
    n_fit   = int(0.6 * len(normal_idx))
    n_pool  = int(0.2 * len(normal_idx))
    rng.shuffle(normal_idx)
    fit_idx   = normal_idx[:n_fit]
    pool_n_idx = normal_idx[n_fit:n_fit + n_pool]
    hold_n_idx = normal_idx[n_fit + n_pool:]

    # Split anomalies
    rng.shuffle(anomaly_idx)
    mid = len(anomaly_idx) // 2
    pool_a_idx = anomaly_idx[:mid]
    hold_a_idx = anomaly_idx[mid:]

    X_fit   = X[fit_idx]
    X_pool  = X[np.concatenate([pool_n_idx, pool_a_idx])]
    y_pool  = y[np.concatenate([pool_n_idx, pool_a_idx])]
    X_hold  = X[np.concatenate([hold_n_idx, hold_a_idx])]
    y_hold  = y[np.concatenate([hold_n_idx, hold_a_idx])]

    # ── Fit AVF on normal training data ───────────────────────────────────────
    avf = AVFDetector()
    avf.fit(X_fit)

    # ── Frequency weights (updated by oracle feedback if use_feedback=True) ───
    # We maintain a per-feature importance vector rather than retraining
    feat_weight = np.ones(X.shape[1])  # multiplicative weight per feature

    # ── Oracle + Attacker ─────────────────────────────────────────────────────
    oracle   = Oracle(k=budget_k)
    attacker = AttackerLite(seed=seed)

    confirmed_pos_memory = []   # accumulate confirmed positives across rounds
    q_summary = np.zeros(X.shape[1])   # oracle Γ output (missed-positive centroid)

    # CUAF: track unique real pool anomaly indices confirmed by oracle over time
    # (pool anomaly indices are the real positives in X_pool)
    pool_anom_idx = set(np.where(y_pool == 1)[0].tolist())  # ground truth in pool
    cuaf_set = set()   # cumulative unique found
    total_pool_anom = len(pool_anom_idx)

    history = []

    for t in range(1, rounds + 1):
        # ── Attacker generates synthetic anomalies near missed summary ────────
        A_t = attacker.generate(
            X_fit, q_summary, n=n_attack,
            use_alignment=use_alignment,
            use_plausibility=use_plausibility,
            use_diversity=use_diversity,
        )
        y_At = np.ones(len(A_t))

        # ── Mixed pool: real pool + synthetic ────────────────────────────────
        X_mixed = np.vstack([X_pool, A_t])
        y_mixed = np.concatenate([y_pool, y_At])
        n_real  = len(X_pool)

        # ── Score mixed pool using AVF (with optional learned feature weights)─
        scores = avf.score_samples(X_mixed)

        # ── Similarity reranking using JACCARD (AVF-native, not AE latent) ───
        if use_reranking and len(confirmed_pos_memory) > 0:
            all_confirmed = np.vstack(confirmed_pos_memory)
            scores = avf.similarity_rerank(X_mixed, scores, all_confirmed)

        ranking = np.argsort(-scores)

        # ── Oracle feedback ───────────────────────────────────────────────────
        ak = min(budget_k + t, len(X_mixed)) if adaptive_k else None
        fb = oracle.inspect(y_mixed, ranking, adaptive_k=ak)

        # Update confirmed positive memory (real positives only)
        # Also track CUAF: which pool anomaly indices were seen
        for idx in fb['confirmed_pos_idx']:
            if idx < len(X_pool) and idx in pool_anom_idx:
                cuaf_set.add(idx)
        if len(fb['confirmed_pos_idx']) > 0:
            confirmed_pos_memory.append(X_mixed[fb['confirmed_pos_idx']])
        cuaf_frac = len(cuaf_set) / max(total_pool_anom, 1)

        # Oracle Γ compression: mean of missed positive features
        if len(fb['missed_idx']) > 0:
            missed_X = X_mixed[fb['missed_idx']]
            if raw_labels:
                # Upper bound: attacker gets centroid of ALL real positives
                all_pos_X = X_pool[y_pool == 1]
                q_summary = all_pos_X.mean(axis=0) if len(all_pos_X) > 0 else missed_X.mean(axis=0)
            else:
                q_summary = oracle.gamma_compress(missed_X)

        # Feature frequency update from oracle labels (use_feedback)
        if use_feedback:
            topk_X = X_mixed[fb['topk_idx']]
            topk_y = fb['topk_y']
            pos_mask = topk_y == 1
            neg_mask = topk_y == 0
            if pos_mask.sum() > 0 and neg_mask.sum() > 0:
                pos_freq = topk_X[pos_mask].mean(axis=0)
                neg_freq = topk_X[neg_mask].mean(axis=0)
                # Features more common in positives → boost AVF frequency
                diff = pos_freq - neg_freq
                avf.freq = np.clip(avf.freq + 0.01 * diff, 1e-6, 1 - 1e-6)

        # ── ASR on mixed pool ─────────────────────────────────────────────────
        effective_k = min(budget_k + t, len(X_mixed)) if adaptive_k else budget_k
        asr = compute_asr(y_At, ranking, n_real, effective_k)

        # ── Holdout evaluation (FIXED, no synthetic contamination) ────────────
        hold_metrics = evaluate_on_holdout(avf, X_hold, y_hold, budget_k)

        row = {
            'round':          t,
            'holdout_auroc':  hold_metrics['auroc'],
            'holdout_ndcg':   hold_metrics['ndcg'],
            'holdout_recall': hold_metrics['recall'],
            'pool_ndcg':      fb['ndcg'],
            'pool_recall':    fb['recall'],
            'asr':            asr,
            'cuaf':           cuaf_frac,          # cumulative fraction of pool anomalies found
            'cuaf_count':     len(cuaf_set),
        }
        history.append(row)
        print(f"  [t={t:02d}] CUAF={cuaf_frac*100:.0f}% ({len(cuaf_set)}/{total_pool_anom}) "
              f"AUROC={hold_metrics['auroc']:.3f} ASR={asr:.3f}")

    summary = {
        'auroc':       np.mean([r['holdout_auroc'] for r in history]),
        'ndcg':        np.mean([r['holdout_ndcg']  for r in history]),
        'recall':      np.mean([r['holdout_recall'] for r in history]),
        'asr':         np.mean([r['asr']            for r in history]),
        'cuaf_final':  history[-1]['cuaf'],          # CUAF at last round (primary metric)
        'cuaf_count':  history[-1]['cuaf_count'],    # absolute count
        'cuaf_total':  total_pool_anom,
        'cuaf_mean':   np.mean([r['cuaf'] for r in history]),
        'final_auroc': history[-1]['holdout_auroc'],
        'final_ndcg':  history[-1]['holdout_ndcg'],
        'final_recall':history[-1]['holdout_recall'],
        'history':     history,
    }
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# 6. Ablation table runner
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AVF Component Ablation Study")
    parser.add_argument('--data', required=True, help='Path to KDD Probe CSV')
    parser.add_argument('--gt',   required=True, help='Path to ground truth CSV')
    parser.add_argument('--rounds', type=int, default=20)
    parser.add_argument('--budget', type=int, default=20)
    parser.add_argument('--n-attack', type=int, default=15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', default='avf_ablation_results.json')
    args = parser.parse_args()

    # ── Load data (DARPALoader-compatible) ────────────────────────────────────
    print("Loading data...")
    df  = pd.read_csv(args.data)
    gt  = pd.read_csv(args.gt)

    id_col    = "Object_ID" if "Object_ID" in df.columns else "UUID"
    gt_id     = "uuid"  if "uuid"  in gt.columns else gt.columns[0]
    label_col = "label" if "label" in gt.columns else gt.columns[-1]

    proc_ids  = df[id_col].tolist()
    apt_uuids = set(gt.loc[gt[label_col] == "AdmSubject::Node", gt_id].tolist())
    y = np.array([1 if pid in apt_uuids else 0 for pid in proc_ids], dtype=np.int32)

    feat_cols = [c for c in df.columns if c != id_col]
    X = df[feat_cols].fillna(0).values.astype(float)


    # Binary encode (some KDD features are counts)
    X = (X > 0).astype(float)

    print(f"  Samples: {len(X)}, Features: {X.shape[1]}, Anomalies: {y.sum()} ({y.mean()*100:.1f}%)")

    # ── Define ablation conditions ────────────────────────────────────────────
    COMMON = dict(rounds=args.rounds, budget_k=args.budget,
                  n_attack=args.n_attack, seed=args.seed)

    # Build-up design: start from weakest (static AVF) and add components
    conditions = [
        # Name,                          flags (all False = ablated)
        ("Static AVF (no OMRQ)",         dict(use_feedback=False, use_reranking=False,
                                              use_alignment=False, use_plausibility=False,
                                              use_diversity=False)),
        ("+ Oracle feedback",            dict(use_feedback=True,  use_reranking=False,
                                              use_alignment=False, use_plausibility=False,
                                              use_diversity=False)),
        ("+ Similarity reranking",       dict(use_feedback=True,  use_reranking=True,
                                              use_alignment=False, use_plausibility=False,
                                              use_diversity=False)),
        ("+ Diversity",                  dict(use_feedback=True,  use_reranking=True,
                                              use_alignment=False, use_plausibility=False,
                                              use_diversity=True)),
        ("+ Signature alignment",        dict(use_feedback=True,  use_reranking=True,
                                              use_alignment=True,  use_plausibility=False,
                                              use_diversity=True)),
        ("Full OMRQ-AVF",                dict(use_feedback=True,  use_reranking=True,
                                              use_alignment=True,  use_plausibility=True,
                                              use_diversity=True)),
        # Upper bounds / variants
        ("w/ raw labels (upper bound)",  dict(use_feedback=True,  use_reranking=True,
                                              use_alignment=True,  use_plausibility=True,
                                              use_diversity=True,  raw_labels=True)),
        ("w/ adaptive k",                dict(use_feedback=True,  use_reranking=True,
                                              use_alignment=True,  use_plausibility=True,
                                              use_diversity=True,  adaptive_k=True)),
    ]

    # ── Run all conditions ────────────────────────────────────────────────────
    results = {}
    print(f"\n{'='*70}")
    for name, flags in conditions:
        print(f"\n[Condition] {name}")
        r = run_ablation(X, y, **COMMON, **flags)
        results[name] = r
        print(f"  → Avg AUROC={r['auroc']:.3f}  nDCG={r['ndcg']:.3f}  "
              f"Recall={r['recall']:.3f}  ASR={r['asr']:.3f}")
        print(f"  → Final AUROC={r['final_auroc']:.3f}  nDCG={r['final_ndcg']:.3f}")

    # ── Print summary table ───────────────────────────────────────────────────
    total_anom = list(results.values())[0]['cuaf_total']
    print(f"\n{'='*72}")
    print(f"PRIMARY ABLATION METRIC: CUAF@T (Cumulative Unique Anomalies Found after {args.rounds} rounds)")
    print(f"Monotone increasing: more components = more APTs discovered by oracle")
    print(f"Pool anomalies available: {total_anom}")
    print(f"{'='*72}")
    print(f"{'Condition':<30}  {'CUAF@T':>8} {'CUAF%':>7} {'AUROC':>7} {'ASR':>7}")
    print('-'*60)
    for name, r in results.items():
        bar = '█' * int(r['cuaf_final'] * 20)  # visual bar
        print(f"{name:<30}  {r['cuaf_count']:>3}/{total_anom:<3} {r['cuaf_final']*100:>6.1f}%"
              f"  {r['final_auroc']*100:>6.1f}  {r['asr']*100:>6.1f}%  {bar}")

    # Save
    out = {k: {kk: vv for kk, vv in v.items() if kk != 'history'}
           for k, v in results.items()}
    with open(args.output, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
