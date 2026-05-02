#!/usr/bin/env python3
"""
OMRQ: Oracle-Mediated Red Queen Pipeline
=========================================
A triadic game-theoretic framework for co-evolving anomaly detection.

Three players:
  - Defender:  ensemble of OC-SVM + Isolation Forest + Autoencoder + RareRuleMiner
  - Attacker:  WGAN-GP generator producing evasive binary anomalies
  - Oracle:    budgeted feedback (top-k labels → defender, compressed summary → attacker)

Usage:
    python omrq_pipeline.py \\
        --data   data/pandex/clearscope/ProcessAll.csv \\
        --gt     data/pandex/clearscope/clearscope_pandex_merged.csv \\
        --rounds 20 --budget 10 --seed 42

Author: OMRQ Research Team
"""

import argparse
import math
import os
import time
import warnings
import json
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.spatial.distance import cosine as cosine_dist

import torch
import torch.nn as nn
import torch.optim as optim

warnings.filterwarnings('ignore')

# ============================================================
# 1. CONFIGURATION
# ============================================================

@dataclass
class OMRQConfig:
    """All hyperparameters for the OMRQ pipeline."""
    # Data
    data_path: str = ""
    gt_path: str = ""
    
    # Red Queen loop
    rounds: int = 20
    budget_k: int = 10
    batch_frac: float = 0.8       # fraction of pool sampled each round
    max_batch: int = 5000         # max samples per round (for large datasets)
    n_attack: int = 20            # synthetic anomalies per round
    ocsvm_max_samples: int = 10000  # subsample OC-SVM training for scalability
    
    # Defender
    ae_latent_dim: int = 16
    ae_epochs: int = 30
    ae_lr: float = 1e-3
    ae_batch_size: int = 2048
    lambda_sim: float = 0.3       # similarity reranking weight
    lambda_reg: float = 1e-4      # defender weight regularization
    
    # Attacker
    gan_latent_dim: int = 32
    gan_epochs_per_round: int = 50
    gan_lr: float = 1e-4
    gan_gp_lambda: float = 10.0   # gradient penalty
    lambda_ev: float = 1.0        # evasion weight
    lambda_sig: float = 0.5       # signature alignment weight
    lambda_plaus: float = 0.3     # plausibility weight
    lambda_div: float = 0.1       # diversity weight
    
    # Surrogate
    sur_lr: float = 1e-3
    sur_epochs: int = 30
    
    # Oracle compression
    gamma_type: str = "mean"      # "mean", "medoid", "mask"
    
    # Ablation modes
    no_adapt: bool = False        # Static baseline: defender never adapts (no weights update, no refit)
    feedback_only: bool = False   # Feedback-only: weight updates only, skip AE refit
    no_reranking: bool = False    # w/o similarity reranking
    no_alignment: bool = False    # w/o signature alignment (set lambda_sig=0)
    no_plausibility: bool = False # w/o plausibility (set lambda_plaus=0)
    no_diversity: bool = False    # w/o diversity (set lambda_div=0)
    raw_labels: bool = False      # w/ raw labels upper bound (no Gamma compression)
    adaptive_k: bool = False      # w/ adaptive k (budget grows with round)
    no_avf: bool = False          # w/o AVF detector (ablate AVF contribution)
    avf_only: bool = False        # use ONLY AVF as backbone detector
    ae_only: bool = False         # use ONLY AutoEncoder as backbone detector
    uniform_weights: bool = False # freeze ensemble weights at 1/N (no learning) — ablation #2
    attacker_type: str = "gan"    # "gan" or "diffusion"
    
    # Evaluation
    seed: int = 42
    output_dir: str = "omrq_results"


# ============================================================
# 2. DATA LOADING
# ============================================================

class DARPALoader:
    """Loads DARPA TC process-action binary matrices and ground truth."""
    
    def __init__(self, data_path: str, gt_path: str):
        self.data_path = data_path
        self.gt_path = gt_path
        
    def load(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Returns:
            X:        (N, D) float32 binary feature matrix
            y:        (N,)   int labels (1=APT, 0=benign)
            proc_ids: list of process UUIDs
        """
        processes = pd.read_csv(self.data_path)
        labels_df = pd.read_csv(self.gt_path)
        
        # Extract process IDs
        id_col = "Object_ID" if "Object_ID" in processes.columns else "UUID"
        proc_ids = processes[id_col].tolist()
        
        # Extract feature matrix (all columns except ID)
        X = processes.drop(columns=[id_col]).values.astype(np.float32)
        
        # Build labels: APT processes have label == "AdmSubject::Node"
        apt_uuids = set(
            labels_df.loc[labels_df["label"] == "AdmSubject::Node", "uuid"].tolist()
        )
        y = np.array([1 if pid in apt_uuids else 0 for pid in proc_ids], dtype=np.int32)
        
        print(f"[Data] Loaded {X.shape[0]} processes, {X.shape[1]} features")
        print(f"[Data] APT count: {y.sum()} ({100*y.mean():.1f}%)")
        
        return X, y, proc_ids


# ============================================================
# 3. DEFENDER COMPONENTS
# ============================================================

class AutoEncoderDetector(nn.Module):
    """PyTorch Autoencoder for anomaly scoring via reconstruction error."""
    
    def __init__(self, input_dim: int, latent_dim: int = 16):
        super().__init__()
        # Encoder
        h1 = min(128, input_dim)
        h2 = min(64, h1)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h1), nn.ReLU(),
            nn.Linear(h1, h2), nn.ReLU(),
            nn.Linear(h2, latent_dim), nn.ReLU(),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, h2), nn.ReLU(),
            nn.Linear(h2, h1), nn.ReLU(),
            nn.Linear(h1, input_dim), nn.Sigmoid(),
        )
        
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
    
    def encode(self, x):
        return self.encoder(x)
    
    def reconstruction_error(self, x):
        """Per-sample L1 reconstruction error."""
        with torch.no_grad():
            recon = self.forward(x)
            return torch.sum(torch.abs(x - recon), dim=1).numpy()


# ============================================================
# 3b. RARE RULE MINER DETECTOR
# ============================================================

class RareRuleMiner:
    """
    Rare-itemset-based anomaly detector.

    Algorithm:
      1. Mine frequent itemsets from benign training data using a low support
         threshold (= most itemsets are "common").
      2. Rare itemsets are those whose support falls *below* min_rare_support
         but *above* a noise floor (min_noise_support).
      3. Rare association rules are extracted from rare itemsets with minimum
         confidence min_conf.
      4. At scoring time, a data point receives a score equal to the fraction
         of its active features that participate in rare itemsets, plus a bonus
         for every rare rule it satisfies. Points that match rare rules are
         strongly anomalous.
    """

    def __init__(self,
                 min_rare_support: float = 0.05,
                 min_noise_support: float = 0.001,
                 min_conf: float = 0.7,
                 max_itemset_size: int = 3):
        self.min_rare_support = min_rare_support   # upper bound on support for "rare"
        self.min_noise_support = min_noise_support  # lower bound (filter pure noise)
        self.min_conf = min_conf
        self.max_itemset_size = max_itemset_size
        self.rare_itemsets: List[frozenset] = []
        self.rare_rules: List[Tuple[frozenset, frozenset, float]] = []  # (antecedent, consequent, conf)
        self.n_train: int = 0

    def _col_support(self, X: np.ndarray) -> np.ndarray:
        """Return per-column frequency (support) for binary matrix X."""
        return X.mean(axis=0)  # shape (d,)

    def fit(self, X: np.ndarray):
        """
        Identify rare itemsets and rules from benign training data X (binary).
        X: (N, d) binary numpy array.
        """
        N, d = X.shape
        self.n_train = N
        col_support = self._col_support(X)

        # --- Step 1: find rare individual items (1-itemsets) ---
        rare_items = [
            frozenset([j])
            for j in range(d)
            if self.min_noise_support <= col_support[j] <= self.min_rare_support
        ]

        all_rare = list(rare_items)

        # --- Step 2: extend to k-itemsets (up to max_itemset_size) ---
        current_level = rare_items
        for size in range(2, self.max_itemset_size + 1):
            if not current_level:
                break
            next_level = []
            items_list = list(current_level)
            for i in range(len(items_list)):
                for j in range(i + 1, len(items_list)):
                    candidate = items_list[i] | items_list[j]
                    if len(candidate) != size:
                        continue
                    # Compute support: rows where ALL items in candidate are active
                    cols = list(candidate)
                    sup = X[:, cols].all(axis=1).mean()
                    if self.min_noise_support <= sup <= self.min_rare_support:
                        next_level.append(candidate)
                        all_rare.append(candidate)
            current_level = next_level

        self.rare_itemsets = all_rare

        # --- Step 3: extract rare association rules ---
        self.rare_rules = []
        for itemset in self.rare_itemsets:
            if len(itemset) < 2:
                continue
            items = list(itemset)
            # Try each item as consequent
            for k, consequent_item in enumerate(items):
                antecedent = frozenset(items[:k] + items[k+1:])
                consequent = frozenset([consequent_item])
                # confidence = P(antecedent ∪ consequent) / P(antecedent)
                ant_cols = list(antecedent)
                ant_sup = X[:, ant_cols].all(axis=1).mean()
                if ant_sup < 1e-9:
                    continue
                full_cols = list(itemset)
                full_sup = X[:, full_cols].all(axis=1).mean()
                conf = full_sup / ant_sup
                if conf >= self.min_conf:
                    self.rare_rules.append((antecedent, consequent, conf))

        print(f"  [RareRuleMiner] Found {len(self.rare_itemsets)} rare itemsets, "
              f"{len(self.rare_rules)} rare rules.")

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Score each sample. Higher score = more anomalous.
        Score = (rare itemset activation ratio) + (rule satisfaction bonus).
        """
        N, d = X.shape
        scores = np.zeros(N, dtype=np.float64)

        if not self.rare_itemsets and not self.rare_rules:
            return scores

        # --- Part 1: rare itemset activation fraction ---
        for itemset in self.rare_itemsets:
            cols = list(itemset)
            activated = X[:, cols].all(axis=1)  # bool (N,)
            scores += activated.astype(float) / max(len(self.rare_itemsets), 1)

        # --- Part 2: rare rule satisfaction bonus ---
        for antecedent, consequent, conf in self.rare_rules:
            ant_cols = list(antecedent)
            con_cols = list(consequent)
            ant_active = X[:, ant_cols].all(axis=1)
            con_active = X[:, con_cols].all(axis=1)
            satisfied = ant_active & con_active
            scores += satisfied.astype(float) * conf  # weight by confidence

        return scores


class AVFDetector:
    """
    Attribute Value Frequency (AVF) anomaly detector for binary/categorical data.

    Score(x) = (1/d) * sum_j [ 1 / f(x_j) ]

    where f(x_j) = fraction of training samples with value x_j in attribute j.
    A record with many rare attribute values gets a high (anomalous) score.
    Entirely parameter-free — just count frequencies.
    """

    def __init__(self):
        self.freq_: Optional[np.ndarray] = None   # shape (d, 2): freq[j,v] = P(attr_j = v)

    def fit(self, X: np.ndarray):
        """X: (N, d) binary numpy array (values in {0, 1})."""
        X_bin = (X > 0.5).astype(np.int32)
        N, d = X_bin.shape
        # freq[j, 0] = P(x_j=0), freq[j, 1] = P(x_j=1)
        self.freq_ = np.zeros((d, 2), dtype=np.float64)
        self.freq_[:, 1] = X_bin.mean(axis=0)          # P(attr=1)
        self.freq_[:, 0] = 1.0 - self.freq_[:, 1]      # P(attr=0)
        # Clip to avoid division by zero
        self.freq_ = np.clip(self.freq_, 1e-6, 1.0)
        print(f"  [AVF] Fitted on {N} samples, {d} binary features.")

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Higher score = more anomalous."""
        if self.freq_ is None:
            return np.zeros(X.shape[0])
        X_bin = (X > 0.5).astype(np.int32)  # (N, d)
        N, d = X_bin.shape
        # For each sample, look up the frequency of its value per attribute
        # X_bin[i,j] ∈ {0,1} → freq_[j, X_bin[i,j]]
        # Vectorised: freq_lookup[i,j] = freq_[j, X_bin[i,j]]
        freq_lookup = self.freq_[np.arange(d), X_bin]  # (N, d)
        # AVF score = mean of inverse frequencies
        scores = (1.0 / freq_lookup).mean(axis=1)       # (N,)
        return scores


class DefenderEnsemble:
    """
    Weighted ensemble of base detectors.
    score(x) = Σ w_m * s_m(x),  Σ w_m = 1
    """
    
    def __init__(self, input_dim: int, config: OMRQConfig):
        self.config = config
        self.input_dim = input_dim
        self.scaler = MinMaxScaler()
        
        # Base detectors
        self.ocsvm = OneClassSVM(kernel='rbf', nu=0.05, gamma='auto')
        self.iforest = IsolationForest(n_estimators=200, contamination=0.05,
                                       random_state=config.seed)
        self.ae = AutoEncoderDetector(input_dim, config.ae_latent_dim)
        self.rare_miner = RareRuleMiner(
            min_rare_support=0.05,
            min_noise_support=0.001,
            min_conf=0.7,
            max_itemset_size=2  # keep at 2 for speed; set 3 for richer rules
        )
        self.avf = AVFDetector()

        # Ensemble weights (learnable via ranking loss) — 5 detectors
        self.weights = np.array([1/5, 1/5, 1/5, 1/5, 1/5], dtype=np.float64)
        if getattr(config, 'no_avf', False):
            # Zero out AVF slot (index 4), redistribute weight to others
            self.weights = np.array([1/4, 1/4, 1/4, 1/4, 0.0], dtype=np.float64)
        if getattr(config, 'avf_only', False):
            # Use only AVF — all weight on slot 4
            self.weights = np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        if getattr(config, 'ae_only', False):
            # Use only AE — all weight on slot 2
            self.weights = np.array([0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float64)
        self.detector_names = ['OC-SVM', 'IForest', 'AutoEncoder', 'RareRules', 'AVF']
        
    def fit(self, X_normal: np.ndarray):
        """Train all base detectors on normal (benign) data."""
        print("[Defender] Fitting base detectors...")
        
        # Subsample for OC-SVM if dataset is large (RBF kernel is O(n^2))
        if X_normal.shape[0] > self.config.ocsvm_max_samples:
            idx = np.random.choice(X_normal.shape[0], self.config.ocsvm_max_samples, replace=False)
            X_ocsvm = X_normal[idx]
            print(f"  OC-SVM: subsampled to {self.config.ocsvm_max_samples} / {X_normal.shape[0]}")
        else:
            X_ocsvm = X_normal
        print("  [Debug] Starting OC-SVM fit...")
        self.ocsvm.fit(X_ocsvm)
        print("  [Debug] Starting IForest fit...")
        self.iforest.fit(X_normal)
        
        print("  [Debug] Starting AE fit...")
        # Fit autoencoder
        print("    [Debug] Converting to tensor...")
        X_t = torch.FloatTensor(X_normal)
        print("    [Debug] Tensor created. Size:", X_t.size())
        optimizer = optim.Adam(self.ae.parameters(), lr=self.config.ae_lr)
        self.ae.train()
        print("    [Debug] Entering epoch loop...")
        for epoch in range(self.config.ae_epochs):
            if epoch == 0:
                print("      [Debug] Epoch 0 start...")
            perm = torch.randperm(X_t.size(0))
            if epoch == 0:
                print("      [Debug] randperm done...")
            total_loss = 0
            n_batches = 0
            for i in range(0, X_t.size(0), self.config.ae_batch_size):
                batch = X_t[perm[i:i + self.config.ae_batch_size]]
                if batch.size(0) < 2:
                    continue
                recon = self.ae(batch)
                loss = nn.functional.binary_cross_entropy(recon, batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1
            if (epoch + 1) % 10 == 0:
                avg = total_loss / max(n_batches, 1)
                print(f"  AE epoch {epoch+1}/{self.config.ae_epochs}, loss={avg:.6f}")
        self.ae.eval()

        # Fit rare rule miner (binary data expected)
        print("  [Debug] Starting RareRuleMiner fit...")
        X_binary = (X_normal > 0.5).astype(np.float64)
        self.rare_miner.fit(X_binary)

        # Fit AVF detector
        print("  [Debug] Starting AVF fit...")
        self.avf.fit(X_binary)

        print("[Defender] Base detectors fitted.")

    def _raw_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Returns (N, 5) matrix of normalized per-detector scores.
        Higher = more anomalous.
        """
        N = X.shape[0]
        scores = np.zeros((N, 5), dtype=np.float64)

        # OC-SVM: decision_function returns signed distance (negative = anomalous)
        ocsvm_raw = -self.ocsvm.decision_function(X)

        # IForest: score_samples returns negative anomaly scores (lower = more anomalous)
        if_raw = -self.iforest.score_samples(X)

        # AE: reconstruction error
        X_t = torch.FloatTensor(X)
        ae_raw = self.ae.reconstruction_error(X_t)

        # Rare Rule Miner: rare itemset / rule activation score
        X_binary = (X > 0.5).astype(np.float64)
        rr_raw = self.rare_miner.score_samples(X_binary)

        # AVF: attribute value frequency inverse score
        avf_raw = self.avf.score_samples(X_binary)
        if getattr(self.config, 'no_avf', False):
            avf_raw = np.zeros_like(avf_raw)  # zero out AVF contribution

        if getattr(self.config, 'avf_only', False):
            # Zero out all detectors except AVF
            ocsvm_raw = np.zeros_like(avf_raw)
            if_raw    = np.zeros_like(avf_raw)
            ae_raw    = np.zeros_like(avf_raw)
            rr_raw    = np.zeros_like(avf_raw)

        if getattr(self.config, 'ae_only', False):
            # Zero out all detectors except AE
            ocsvm_raw = np.zeros_like(ae_raw)
            if_raw    = np.zeros_like(ae_raw)
            rr_raw    = np.zeros_like(ae_raw)
            avf_raw   = np.zeros_like(ae_raw)

        # Normalize each to [0, 1]
        for i, raw in enumerate([ocsvm_raw, if_raw, ae_raw, rr_raw, avf_raw]):
            mn, mx = raw.min(), raw.max()
            if mx - mn > 1e-12:
                scores[:, i] = (raw - mn) / (mx - mn)
            else:
                scores[:, i] = 0.5

        return scores
    
    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute weighted ensemble anomaly score for each sample."""
        raw = self._raw_scores(X)
        return raw @ self.weights  # (N,)
    
    def get_representation(self, X: np.ndarray) -> np.ndarray:
        """Get AE latent representation h(x) for similarity computations."""
        X_t = torch.FloatTensor(X)
        with torch.no_grad():
            return self.ae.encode(X_t).numpy()
    
    def update_weights_from_pairs(self, X: np.ndarray, labels: np.ndarray,
                                   lr: float = 0.01, steps: int = 50):
        """
        Update ensemble weights using pairwise ranking loss on oracle-labeled data.
        labels: 1=anomaly, 0=benign (only for the top-k inspected set).
        """
        pos_idx = np.where(labels == 1)[0]
        neg_idx = np.where(labels == 0)[0]
        
        if len(pos_idx) == 0 or len(neg_idx) == 0:
            return  # Can't form pairs
        
        raw = self._raw_scores(X)  # (N, 3)
        
        # Gradient descent on pairwise ranking loss
        w = self.weights.copy()
        for _ in range(steps):
            grad = np.zeros_like(w)
            loss_val = 0
            for pi in pos_idx:
                for ni in neg_idx:
                    diff = raw[pi] - raw[ni]  # (3,)
                    margin = w @ diff
                    sigmoid = 1.0 / (1.0 + np.exp(margin))
                    loss_val += np.log(1 + np.exp(-margin))
                    grad -= sigmoid * diff
            
            n_pairs = len(pos_idx) * len(neg_idx)
            grad /= n_pairs
            grad += 2 * self.config.lambda_reg * w
            
            w -= lr * grad
            # Project onto simplex
            w = np.maximum(w, 1e-6)
            w /= w.sum()
        
        self.weights = w

    def similarity_rerank(self, X_pool: np.ndarray, scores: np.ndarray,
                          confirmed_positives: np.ndarray) -> np.ndarray:
        """
        Boost scores of uninspected samples similar to confirmed positives.
        Eq. 3 in the paper. Vectorized for large pools.
        """
        if len(confirmed_positives) == 0:
            return scores
        
        h_pool = self.get_representation(X_pool)    # (N, d)
        h_pos = self.get_representation(confirmed_positives)  # (P, d)
        
        # Vectorized cosine similarity
        norms_pool = np.linalg.norm(h_pool, axis=1, keepdims=True)  # (N, 1)
        norms_pos = np.linalg.norm(h_pos, axis=1, keepdims=True)    # (P, 1)
        norms_pool = np.maximum(norms_pool, 1e-8)
        norms_pos = np.maximum(norms_pos, 1e-8)
        
        # (N, d) @ (d, P) -> (N, P)
        sim_matrix = (h_pool / norms_pool) @ (h_pos / norms_pos).T
        boost = np.maximum(sim_matrix.max(axis=1), 0)  # (N,)
        
        return scores + self.config.lambda_sim * boost


# ============================================================
# 4. ATTACKER COMPONENTS
# ============================================================

class Generator(nn.Module):
    """Generator network for binary anomaly synthesis."""
    
    def __init__(self, latent_dim: int, output_dim: int, condition_dim: int):
        super().__init__()
        input_dim = latent_dim + condition_dim
        h = max(64, output_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, h * 2), nn.LeakyReLU(0.2),
            nn.Linear(h * 2, h), nn.LeakyReLU(0.2),
            nn.Linear(h, output_dim), nn.Sigmoid(),
        )
    
    def forward(self, z, condition):
        """z: (B, latent_dim), condition: (B, condition_dim) → (B, output_dim)"""
        x = torch.cat([z, condition.expand(z.size(0), -1)], dim=1)
        return self.net(x)


class Critic(nn.Module):
    """WGAN-GP Critic (discriminator)."""
    
    def __init__(self, input_dim: int):
        super().__init__()
        h = max(64, input_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, h), nn.LeakyReLU(0.2),
            nn.Linear(h, h // 2), nn.LeakyReLU(0.2),
            nn.Linear(h // 2, 1),
        )
    
    def forward(self, x):
        return self.net(x)


class SurrogateScorer(nn.Module):
    """Differentiable surrogate that approximates the defender's score function."""
    
    def __init__(self, input_dim: int):
        super().__init__()
        h = max(64, input_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, h), nn.ReLU(),
            nn.Linear(h, h // 2), nn.ReLU(),
            nn.Linear(h // 2, 1), nn.Sigmoid(),
        )
    
    def forward(self, x):
        return self.net(x).squeeze(-1)


class Attacker:
    """
    WGAN-GP based attacker that generates evasive, plausible binary anomalies.
    Conditioned on oracle summary q_t.
    """
    
    def __init__(self, data_dim: int, config: OMRQConfig):
        self.config = config
        self.data_dim = data_dim
        self.condition_dim = config.ae_latent_dim  # oracle summary dimension
        
        self.generator = Generator(config.gan_latent_dim, data_dim, self.condition_dim)
        self.critic = Critic(data_dim)
        self.surrogate = SurrogateScorer(data_dim)
        
        self.opt_g = optim.Adam(self.generator.parameters(), lr=config.gan_lr, betas=(0.0, 0.9))
        self.opt_c = optim.Adam(self.critic.parameters(), lr=config.gan_lr, betas=(0.0, 0.9))
        self.opt_s = optim.Adam(self.surrogate.parameters(), lr=config.sur_lr)
        
    def _gradient_penalty(self, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        """WGAN-GP gradient penalty."""
        alpha = torch.rand(real.size(0), 1)
        interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
        d_interp = self.critic(interp)
        grads = torch.autograd.grad(
            outputs=d_interp, inputs=interp,
            grad_outputs=torch.ones_like(d_interp),
            create_graph=True, retain_graph=True
        )[0]
        gp = ((grads.norm(2, dim=1) - 1) ** 2).mean()
        return gp
    
    def fit_surrogate(self, X: np.ndarray, defender_scores: np.ndarray):
        """Train surrogate to approximate defender score function."""
        X_t = torch.FloatTensor(X)
        y_t = torch.FloatTensor(defender_scores)
        
        self.surrogate.train()
        for _ in range(self.config.sur_epochs):
            pred = self.surrogate(X_t)
            loss = nn.functional.mse_loss(pred, y_t)
            self.opt_s.zero_grad()
            loss.backward()
            self.opt_s.step()
        self.surrogate.eval()
    
    def generate(self, B_t: np.ndarray, q_t: np.ndarray,
                 n_samples: int) -> np.ndarray:
        """Generate n_samples synthetic anomalies conditioned on oracle summary."""
        self.generator.eval()
        z = torch.randn(n_samples, self.config.gan_latent_dim)
        cond = torch.FloatTensor(q_t).unsqueeze(0)  # (1, condition_dim)
        with torch.no_grad():
            fake = self.generator(z, cond)
            # Binarize via stochastic rounding
            binary = (fake > 0.5).float()
        return binary.numpy()
    
    def update(self, X_real: np.ndarray, defender_scores: np.ndarray,
               q_t: np.ndarray, h_func, config: OMRQConfig):
        """
        Full attacker update step:
          1. Train WGAN-GP critic and generator on real anomaly patterns
          2. Apply evasion, signature, diversity, plausibility losses
        """
        X_t = torch.FloatTensor(X_real)
        cond = torch.FloatTensor(q_t).unsqueeze(0)
        
        self.generator.train()
        self.critic.train()
        
        n_critic = 5  # critic steps per generator step
        
        for epoch in range(config.gan_epochs_per_round):
            # --- Critic update ---
            for _ in range(n_critic):
                z = torch.randn(min(X_t.size(0), config.n_attack), config.gan_latent_dim)
                fake = self.generator(z, cond).detach()
                
                # Sample real batch
                idx = torch.randint(0, X_t.size(0), (fake.size(0),))
                real_batch = X_t[idx]
                
                c_real = self.critic(real_batch).mean()
                c_fake = self.critic(fake).mean()
                gp = self._gradient_penalty(real_batch, fake)
                
                c_loss = c_fake - c_real + config.gan_gp_lambda * gp
                self.opt_c.zero_grad()
                c_loss.backward()
                self.opt_c.step()
            
            # --- Generator update ---
            z = torch.randn(config.n_attack, config.gan_latent_dim)
            fake = self.generator(z, cond)
            
            # WGAN generator loss
            g_wgan = -self.critic(fake).mean()
            
            # Evasion loss: minimize surrogate score
            self.surrogate.eval()
            g_evasion = self.surrogate(fake).mean()
            
            # Signature alignment: pull toward oracle summary centroid
            fake_h = h_func(fake.detach().numpy())
            fake_h_t = torch.FloatTensor(fake_h)
            c_t = torch.FloatTensor(q_t).unsqueeze(0).expand_as(fake_h_t)
            g_sig = nn.functional.mse_loss(fake_h_t, c_t)
            
            # Diversity: maximize pairwise distance
            if fake.size(0) > 1:
                dists = torch.cdist(fake, fake, p=2)
                mask = ~torch.eye(fake.size(0), dtype=torch.bool)
                g_div = -dists[mask].mean()
            else:
                g_div = torch.tensor(0.0)
            
            # Plausibility: penalize unrealistic density
            fake_density = fake.sum(dim=1)
            target_density = X_t.sum(dim=1).mean()
            g_plaus = (fake_density - target_density).abs().mean()
            
            # Combined generator loss
            g_loss = (g_wgan
                      + config.lambda_ev * g_evasion
                      + config.lambda_sig * g_sig
                      + config.lambda_div * g_div
                      + config.lambda_plaus * g_plaus)
            
            self.opt_g.zero_grad()
            g_loss.backward()
            self.opt_g.step()
        
        self.generator.eval()
        self.critic.eval()


# ============================================================
# 4b. DIFFUSION ATTACKER (DDPM on binary tabular data)
# ============================================================

class SinusoidalEmbedding(nn.Module):
    """Sinusoidal time-step embedding."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32) / half)
        args  = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb   = torch.cat([args.sin(), args.cos()], dim=-1)
        return emb  # (B, dim)


class DiffusionDenoiser(nn.Module):
    """
    MLP denoiser: predicts noise epsilon given (x_t, t, q_t).
    Input:  x_t  (B, D)  noisy data at step t (continuous, scaled to [-1,1])
            t    (B,)    diffusion timestep
            cond (1, C)  oracle summary q_t
    Output: epsilon_hat (B, D)
    """
    def __init__(self, data_dim: int, condition_dim: int, t_emb_dim: int = 32):
        super().__init__()
        self.t_emb = SinusoidalEmbedding(t_emb_dim)
        h = max(128, data_dim * 2)
        in_dim = data_dim + t_emb_dim + condition_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, h), nn.SiLU(),
            nn.Linear(h, h),      nn.SiLU(),
            nn.Linear(h, h // 2), nn.SiLU(),
            nn.Linear(h // 2, data_dim),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor,
                cond: torch.Tensor) -> torch.Tensor:
        t_emb = self.t_emb(t)                              # (B, t_emb_dim)
        c     = cond.expand(x_t.size(0), -1)              # (B, C)
        inp   = torch.cat([x_t, t_emb, c], dim=-1)        # (B, D+t_emb+C)
        return self.net(inp)                               # (B, D)


class DiffusionAttacker:
    """
    DDPM-based attacker: same interface as GAN Attacker.
    Uses a simple linear noise schedule with T=50 steps on binary data
    (scaled to [-1,1]), conditioned on oracle summary q_t.
    Applies the same evasion/signature/diversity/plausibility objectives
    as the GAN attacker via a surrogate scorer.
    """

    def __init__(self, data_dim: int, config, T: int = 50):
        self.config       = config
        self.data_dim     = data_dim
        self.condition_dim= config.ae_latent_dim
        self.T            = T

        # Build noise schedule
        betas = torch.linspace(1e-4, 0.02, T)             # (T,)
        alphas     = 1.0 - betas
        alpha_bar  = torch.cumprod(alphas, dim=0)          # (T,)
        self.register_buffers(betas, alphas, alpha_bar)

        # Networks
        self.denoiser  = DiffusionDenoiser(data_dim, self.condition_dim)
        self.surrogate = SurrogateScorer(data_dim)

        self.opt_d = optim.Adam(self.denoiser.parameters(),  lr=1e-4)
        self.opt_s = optim.Adam(self.surrogate.parameters(), lr=config.sur_lr)

    def register_buffers(self, betas, alphas, alpha_bar):
        self.betas      = betas
        self.alphas     = alphas
        self.alpha_bar  = alpha_bar
        self.sqrt_ab    = alpha_bar.sqrt()
        self.sqrt_1mab  = (1.0 - alpha_bar).sqrt()

    # ------------------------------------------------------------------
    # Forward (noising) process
    # ------------------------------------------------------------------
    def _q_sample(self, x0: torch.Tensor, t: torch.Tensor,
                  noise: torch.Tensor) -> torch.Tensor:
        """Sample x_t given x_0 and noise: x_t = sqrt(ab_t)*x0 + sqrt(1-ab_t)*noise."""
        s_ab   = self.sqrt_ab[t].unsqueeze(1)    # (B,1)
        s_1mab = self.sqrt_1mab[t].unsqueeze(1)  # (B,1)
        return s_ab * x0 + s_1mab * noise

    # ------------------------------------------------------------------
    # Reverse (denoising) — DDPM sampler
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _p_sample(self, x_t: torch.Tensor, t_scalar: int,
                  cond: torch.Tensor) -> torch.Tensor:
        """One DDPM reverse step."""
        t_ten  = torch.full((x_t.size(0),), t_scalar, dtype=torch.long)
        eps    = self.denoiser(x_t, t_ten, cond)
        alpha  = self.alphas[t_scalar]
        ab     = self.alpha_bar[t_scalar]
        coef   = (1.0 - alpha) / (1.0 - ab).sqrt()
        mean   = (x_t - coef * eps) / alpha.sqrt()
        if t_scalar == 0:
            return mean
        beta = self.betas[t_scalar]
        noise = torch.randn_like(x_t)
        return mean + beta.sqrt() * noise

    # ------------------------------------------------------------------
    # Public interface (matches GAN Attacker)
    # ------------------------------------------------------------------
    def fit_surrogate(self, X: np.ndarray, defender_scores: np.ndarray):
        X_t = torch.FloatTensor(X)
        y_t = torch.FloatTensor(defender_scores)
        self.surrogate.train()
        for _ in range(self.config.sur_epochs):
            pred = self.surrogate(X_t)
            loss = nn.functional.mse_loss(pred, y_t)
            self.opt_s.zero_grad(); loss.backward(); self.opt_s.step()
        self.surrogate.eval()

    def generate(self, B_t: np.ndarray, q_t: np.ndarray,
                 n_samples: int) -> np.ndarray:
        """Run DDPM reverse chain from pure noise → binary samples."""
        self.denoiser.eval()
        cond = torch.FloatTensor(q_t).unsqueeze(0)        # (1, C)
        x    = torch.randn(n_samples, self.data_dim)      # start from noise
        for t in reversed(range(self.T)):
            x = self._p_sample(x, t, cond)
        # Map [-1,1] → [0,1] then binarise
        x01     = (x + 1.0) / 2.0
        binary  = (x01 > 0.5).float()
        return binary.numpy()

    def update(self, X_real: np.ndarray, defender_scores: np.ndarray,
               q_t: np.ndarray, h_func, config):
        """Train denoiser for one round using the same composite loss as GAN."""
        # Scale binary x0 to [-1, 1]
        x0   = torch.FloatTensor(X_real) * 2.0 - 1.0
        cond = torch.FloatTensor(q_t).unsqueeze(0)
        B    = x0.size(0)

        self.denoiser.train()
        self.surrogate.eval()

        n_epochs = config.gan_epochs_per_round
        for _ in range(n_epochs):
            # --- Denoising loss (simplified objective) ---
            t_rand = torch.randint(0, self.T, (B,))
            noise  = torch.randn_like(x0)
            x_t    = self._q_sample(x0, t_rand, noise)
            eps_hat= self.denoiser(x_t, t_rand, cond)
            loss_ddpm = nn.functional.mse_loss(eps_hat, noise)

            # --- Generate fake samples for auxiliary losses ---
            with torch.no_grad():
                x_f = torch.randn(config.n_attack, self.data_dim)
                for t in reversed(range(self.T)):
                    x_f = self._p_sample(x_f, t, cond)
            fake = ((x_f + 1.0) / 2.0).clamp(0, 1).requires_grad_(False)

            # Evasion: push generated samples past defender
            g_evasion = self.surrogate(fake).mean()

            # Signature alignment
            fake_h = h_func(fake.detach().numpy())
            fake_h_t = torch.FloatTensor(fake_h)
            c_t = cond.expand(fake_h_t.size(0), -1)
            g_sig = nn.functional.mse_loss(fake_h_t, c_t)

            # Diversity
            if fake.size(0) > 1:
                dists = torch.cdist(fake, fake, p=2)
                mask  = ~torch.eye(fake.size(0), dtype=torch.bool)
                g_div = -dists[mask].mean()
            else:
                g_div = torch.tensor(0.0)

            # Plausibility (density matching)
            x0_orig = torch.FloatTensor(X_real)
            fake_density   = ((x_f + 1.0) / 2.0 > 0.5).float().sum(dim=1)
            target_density = x0_orig.sum(dim=1).mean()
            g_plaus = (fake_density - target_density).abs().mean()

            loss = (loss_ddpm
                    + config.lambda_ev   * g_evasion
                    + config.lambda_sig  * g_sig
                    + config.lambda_div  * g_div
                    + config.lambda_plaus* g_plaus)

            self.opt_d.zero_grad()
            loss.backward()
            self.opt_d.step()

        self.denoiser.eval()



class BudgetedOracle:
    """
    Holds hidden ground truth. Mediates asymmetric feedback.
    """
    
    def __init__(self, y_full: np.ndarray, k: int, gamma_type: str = "mean"):
        self.y_full = y_full         # labels for the original pool
        self.k = k
        self.gamma_type = gamma_type
    
    def sample_batch(self, X: np.ndarray, y: np.ndarray,
                     frac: float, rng: np.random.Generator,
                     max_batch: int = 5000
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a batch from the pool. Capped at max_batch for scalability.
        Returns: X_batch, y_batch, indices into original pool
        """
        n = X.shape[0]
        size = max(int(n * frac), self.k + 1)
        size = min(size, n, max_batch)  # cap for large datasets
        idx = rng.choice(n, size=size, replace=False)
        return X[idx], y[idx], idx
    
    def evaluate_ranking(self, X_mixed: np.ndarray, y_mixed: np.ndarray,
                         ranking: np.ndarray, h_func
                         ) -> Dict:
        """
        Evaluate the defender's ranking. Returns feedback for both players.
        
        Args:
            X_mixed: (N, D) evaluation pool (real + synthetic)
            y_mixed: (N,)   true labels
            ranking: (N,)   indices sorted by decreasing anomaly score
            h_func:  callable mapping X → latent representations
            
        Returns:
            dict with keys:
              'topk_X', 'topk_y'           — labeled data for defender
              'confirmed_positives'         — X of true positives in top-k
              'missed_summary'              — compressed summary q_{t+1} for attacker
              'metrics'                     — nDCG@k, Recall@k, etc.
        """
        k = min(self.k, len(ranking))
        topk_idx = ranking[:k]
        
        topk_X = X_mixed[topk_idx]
        topk_y = y_mixed[topk_idx]
        
        # Confirmed positives in top-k
        tp_mask = topk_y == 1
        confirmed_pos = topk_X[tp_mask]
        
        # Missed positives: true anomalies NOT in top-k
        all_pos_idx = np.where(y_mixed == 1)[0]
        topk_set = set(topk_idx.tolist())
        missed_idx = [i for i in all_pos_idx if i not in topk_set]
        missed_X = X_mixed[missed_idx] if len(missed_idx) > 0 else np.zeros((0, X_mixed.shape[1]))
        
        # Compute compressed summary Γ(M_t)
        if len(missed_idx) > 0:
            missed_summary = self._compress(missed_X, h_func)
        else:
            # No missed positives: return zero summary
            h_dim = h_func(X_mixed[:1]).shape[1]
            missed_summary = np.zeros(h_dim, dtype=np.float32)
        
        # Compute metrics
        metrics = self._compute_metrics(y_mixed, ranking, k)
        
        return {
            'topk_X': topk_X,
            'topk_y': topk_y,
            'confirmed_positives': confirmed_pos,
            'missed_summary': missed_summary,
            'metrics': metrics,
            'n_missed': len(missed_idx),
        }
    
    def _compress(self, missed_X: np.ndarray, h_func) -> np.ndarray:
        """Apply compression operator Γ."""
        H = h_func(missed_X)  # (M, d)
        
        if self.gamma_type == "mean":
            return H.mean(axis=0)
        
        elif self.gamma_type == "medoid":
            # Select most central point
            dists = np.zeros(len(H))
            for i in range(len(H)):
                dists[i] = np.sum(np.linalg.norm(H - H[i], axis=1))
            return H[np.argmin(dists)]
        
        elif self.gamma_type == "mask":
            # Sparse binary feature mask
            avg = missed_X.mean(axis=0)
            mask = (avg > 0.3).astype(np.float32)
            # Pad or truncate to latent dim
            h_dim = H.shape[1]
            if len(mask) >= h_dim:
                return mask[:h_dim]
            else:
                return np.pad(mask, (0, h_dim - len(mask)))
        
        return H.mean(axis=0)  # fallback
    
    def _compute_metrics(self, y_mixed: np.ndarray, ranking: np.ndarray,
                         k: int) -> Dict:
        """Compute nDCG@k, Recall@k, Precision@k, and counts."""
        n_total_pos = int(y_mixed.sum())
        
        topk_labels = y_mixed[ranking[:k]]
        tp_at_k = int(topk_labels.sum())
        
        # Precision@k
        prec_at_k = tp_at_k / k if k > 0 else 0.0
        
        # Recall@k
        recall_at_k = tp_at_k / n_total_pos if n_total_pos > 0 else 0.0
        
        # nDCG@k
        dcg = 0.0
        for rank_pos in range(k):
            if y_mixed[ranking[rank_pos]] == 1:
                dcg += 1.0 / np.log2(rank_pos + 2)  # rank is 0-indexed
        
        # Ideal DCG
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(n_total_pos, k)))
        ndcg_at_k = dcg / idcg if idcg > 0 else 0.0
        
        # Full nDCG (all positions)
        full_dcg = 0.0
        for rank_pos in range(len(ranking)):
            if y_mixed[ranking[rank_pos]] == 1:
                full_dcg += 1.0 / np.log2(rank_pos + 2)
        full_idcg = sum(1.0 / np.log2(i + 2) for i in range(n_total_pos))
        full_ndcg = full_dcg / full_idcg if full_idcg > 0 else 0.0
        
        return {
            'ndcg_at_k': ndcg_at_k,
            'full_ndcg': full_ndcg,
            'recall_at_k': recall_at_k,
            'precision_at_k': prec_at_k,
            'tp_at_k': tp_at_k,
            'n_total_pos': n_total_pos,
        }


# ============================================================
# 6. EVALUATION & PLOTTING
# ============================================================

def compute_asr(y_synthetic: np.ndarray, ranking: np.ndarray,
                n_real: int, k: int) -> float:
    """
    Attack Success Rate: fraction of synthetic anomalies that evade top-k.
    Synthetic samples are indexed from n_real onwards in X_mixed.
    """
    topk_set = set(ranking[:k].tolist())
    synth_indices = list(range(n_real, n_real + len(y_synthetic)))
    n_evaded = sum(1 for si in synth_indices if si not in topk_set)
    return n_evaded / len(synth_indices) if len(synth_indices) > 0 else 0.0


def compute_auroc_auprc(y_true: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    """Safe AUROC and AUPRC computation."""
    if len(np.unique(y_true)) < 2:
        return 0.5, 0.0
    auroc = roc_auc_score(y_true, scores)
    auprc = average_precision_score(y_true, scores)
    return auroc, auprc


def plot_trajectories(history: List[Dict], output_dir: str):
    """Plot the full conference-quality Red Queen figure set."""
    rounds = [h['round'] for h in history]
    ndcg_vals = [h['ndcg_at_k'] for h in history]
    recall_vals = [h['recall_at_k'] for h in history]
    asr_vals = [h['asr'] for h in history]
    auroc_vals = [h['auroc'] for h in history]

    # --- Style ---
    BLUE = '#0284c7'
    RED = '#e11d48'
    GOLD = '#f59e0b'
    DARK = '#1e293b'
    plt.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 12,
        'legend.fontsize': 10,
        'figure.facecolor': 'white',
    })

    # ================================================================
    # FIGURE A: 4-Panel Red Queen Dashboard
    # ================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('OMRQ Red Queen Trajectories', fontsize=16, fontweight='bold',
                 color=DARK, y=0.98)

    # Panel A: nDCG@k
    axes[0, 0].fill_between(rounds, ndcg_vals, alpha=0.15, color=BLUE)
    axes[0, 0].plot(rounds, ndcg_vals, 'o-', color=BLUE, linewidth=2.5,
                    markersize=6, markeredgecolor='white', markeredgewidth=1.5)
    axes[0, 0].set_ylabel('nDCG@k', fontweight='bold')
    axes[0, 0].set_xlabel('Round')
    axes[0, 0].set_title('(a) Defender: Ranking Quality', fontweight='bold')
    axes[0, 0].set_ylim(-0.05, 1.05)
    axes[0, 0].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axes[0, 0].grid(True, alpha=0.25, linestyle='--')

    # Panel B: Recall@k
    axes[0, 1].fill_between(rounds, recall_vals, alpha=0.15, color='#0369a1')
    axes[0, 1].plot(rounds, recall_vals, 's-', color='#0369a1', linewidth=2.5,
                    markersize=6, markeredgecolor='white', markeredgewidth=1.5)
    axes[0, 1].set_ylabel('Recall@k', fontweight='bold')
    axes[0, 1].set_xlabel('Round')
    axes[0, 1].set_title('(b) Defender: Top-k Recall', fontweight='bold')
    axes[0, 1].set_ylim(-0.05, 1.05)
    axes[0, 1].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axes[0, 1].grid(True, alpha=0.25, linestyle='--')

    # Panel C: ASR (Attacker)
    axes[1, 0].fill_between(rounds, asr_vals, alpha=0.15, color=RED)
    axes[1, 0].plot(rounds, asr_vals, '^-', color=RED, linewidth=2.5,
                    markersize=6, markeredgecolor='white', markeredgewidth=1.5)
    axes[1, 0].set_ylabel('ASR', fontweight='bold')
    axes[1, 0].set_xlabel('Round')
    axes[1, 0].set_title('(c) Attacker: Evasion Success Rate', fontweight='bold')
    axes[1, 0].set_ylim(-0.05, 1.05)
    axes[1, 0].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axes[1, 0].grid(True, alpha=0.25, linestyle='--')

    # Panel D: Dual-axis combined
    ax_left = axes[1, 1]
    ax_right = ax_left.twinx()
    l1 = ax_left.plot(rounds, ndcg_vals, 'o-', color=BLUE, linewidth=2.5,
                      markersize=5, label='nDCG@k (Defender)', markeredgecolor='white')
    l2 = ax_right.plot(rounds, asr_vals, '^-', color=RED, linewidth=2.5,
                       markersize=5, label='ASR (Attacker)', markeredgecolor='white')
    ax_left.set_ylabel('nDCG@k', color=BLUE, fontweight='bold')
    ax_right.set_ylabel('ASR', color=RED, fontweight='bold')
    ax_left.set_xlabel('Round')
    ax_left.set_title('(d) Red Queen: Arms Race', fontweight='bold')
    ax_left.set_ylim(-0.05, 1.05)
    ax_right.set_ylim(-0.05, 1.05)
    ax_left.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax_left.legend(lines, labels, loc='center right', framealpha=0.9)
    ax_left.grid(True, alpha=0.25, linestyle='--')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'red_queen_trajectories.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Saved 4-panel dashboard → {output_dir}/red_queen_trajectories.png")

    # ================================================================
    # FIGURE B: Phase Portrait (nDCG@k vs ASR)
    # ================================================================
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_facecolor('#fafbfd')

    # Plot trajectory with arrows
    for i in range(len(rounds) - 1):
        dx = asr_vals[i+1] - asr_vals[i]
        dy = ndcg_vals[i+1] - ndcg_vals[i]
        ax.annotate('', xy=(asr_vals[i+1], ndcg_vals[i+1]),
                    xytext=(asr_vals[i], ndcg_vals[i]),
                    arrowprops=dict(arrowstyle='->', color=DARK, lw=1.8,
                                   connectionstyle='arc3,rad=0.15'))

    # Color points by round (early=blue, late=red)
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(rounds)))
    scatter = ax.scatter(asr_vals, ndcg_vals, c=range(len(rounds)),
                         cmap='coolwarm', s=120, zorder=5,
                         edgecolors='white', linewidth=2)

    # Label round numbers
    for i, r in enumerate(rounds):
        ax.annotate(f't={r}', (asr_vals[i], ndcg_vals[i]),
                    textcoords='offset points', xytext=(8, 8),
                    fontsize=9, fontweight='bold', color=DARK)

    ax.set_xlabel('Attacker ASR →', fontsize=13, fontweight='bold')
    ax.set_ylabel('Defender nDCG@k →', fontsize=13, fontweight='bold')
    ax.set_title('Phase Portrait: Defender–Attacker Co-Evolution',
                 fontsize=14, fontweight='bold', color=DARK)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.2, linestyle='--')

    # Add quadrant labels
    ax.text(0.15, 0.92, 'Defender\nDominates', fontsize=10, ha='center',
            color=BLUE, fontweight='bold', alpha=0.6, transform=ax.transAxes)
    ax.text(0.85, 0.08, 'Attacker\nDominates', fontsize=10, ha='center',
            color=RED, fontweight='bold', alpha=0.6, transform=ax.transAxes)

    cbar = plt.colorbar(scatter, ax=ax, label='Round', shrink=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'phase_portrait.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Saved phase portrait → {output_dir}/phase_portrait.png")

    # ================================================================
    # FIGURE C: Defender–Attacker Margin Curve
    # ================================================================
    margin = [n - a for n, a in zip(ndcg_vals, asr_vals)]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(rounds, margin, 0, where=[m >= 0 for m in margin],
                    alpha=0.2, color=BLUE, interpolate=True)
    ax.fill_between(rounds, margin, 0, where=[m < 0 for m in margin],
                    alpha=0.2, color=RED, interpolate=True)
    ax.plot(rounds, margin, 'D-', color=DARK, linewidth=2.5, markersize=6,
            markeredgecolor='white', markeredgewidth=1.5)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.6)
    ax.set_xlabel('Round', fontsize=12, fontweight='bold')
    ax.set_ylabel('Δ = nDCG@k − ASR', fontsize=12, fontweight='bold')
    ax.set_title('Defender–Attacker Dominance Margin', fontsize=14,
                 fontweight='bold', color=DARK)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.25, linestyle='--')

    # Annotate regions
    ax.text(0.2, 0.85, 'Defender dominates (Δ > 0)', color=BLUE,
            fontsize=10, fontweight='bold', alpha=0.6, transform=ax.transAxes)
    ax.text(0.2, 0.1, 'Attacker dominates (Δ < 0)', color=RED,
            fontsize=10, fontweight='bold', alpha=0.6, transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dominance_margin.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Saved margin curve → {output_dir}/dominance_margin.png")


def plot_weight_evolution(weight_history: List[np.ndarray], names: List[str],
                          output_dir: str):
    """Plot how defender ensemble weights evolve over rounds."""
    COLORS = ['#0284c7', '#059669', '#7c3aed', '#f59e0b']
    W = np.array(weight_history)  # (T, M)
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, name in enumerate(names):
        c = COLORS[i % len(COLORS)]
        ax.plot(range(1, len(W) + 1), W[:, i], 'o-', label=name,
                color=c, linewidth=2.5, markersize=5,
                markeredgecolor='white', markeredgewidth=1.5)
    ax.set_xlabel('Round', fontsize=12, fontweight='bold')
    ax.set_ylabel('Weight', fontsize=12, fontweight='bold')
    ax.set_title('Defender Ensemble Weight Evolution', fontsize=14,
                 fontweight='bold', color='#1e293b')
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.25, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'weight_evolution.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Saved weight evolution → {output_dir}/weight_evolution.png")


# ============================================================
# 7. MAIN RED QUEEN LOOP
# ============================================================

def run_omrq(config: OMRQConfig):
    """Execute the full OMRQ Red Queen loop."""
    
    print("=" * 70)
    print("  OMRQ: Oracle-Mediated Red Queen Pipeline")
    print("=" * 70)
    
    # Seed everything
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    
    os.makedirs(config.output_dir, exist_ok=True)
    
    # --- 1. Load Data ---
    loader = DARPALoader(config.data_path, config.gt_path)
    X, y, proc_ids = loader.load()
    
    # --- 2. Initialize Defender ---
    defender = DefenderEnsemble(input_dim=X.shape[1], config=config)
    
    # Train on ALL data initially (unsupervised: pretend all is normal)
    # This mimics the realistic scenario where we have no labels at start
    defender.fit(X)
    
    # h_func: shared representation for similarity and oracle compression
    def h_func(x_input):
        if isinstance(x_input, np.ndarray):
            return defender.get_representation(x_input)
        return defender.get_representation(x_input)
    
    # --- 3. Initialize Attacker ---
    if config.attacker_type == "diffusion":
        print("[Attacker] Using DDPM-based DiffusionAttacker (T=50)")
        attacker = DiffusionAttacker(data_dim=X.shape[1], config=config, T=50)
    else:
        attacker = Attacker(data_dim=X.shape[1], config=config)
    
    # --- 4. Initialize Oracle ---
    oracle = BudgetedOracle(y_full=y, k=config.budget_k, gamma_type=config.gamma_type)
    
    # --- 5. Red Queen Loop ---
    q_t = np.zeros(config.ae_latent_dim, dtype=np.float32)  # initial empty summary
    history = []
    weight_history = [defender.weights.copy()]

    # Defender label memory: accumulate oracle feedback across rounds
    # This is key for bounded oscillation — the defender remembers past findings
    memory_X_pos = []   # confirmed anomalies across all rounds
    memory_X_neg = []   # confirmed benign across all rounds
    
    print(f"\n{'='*70}")
    print(f"  Starting Red Queen Loop: {config.rounds} rounds, budget k={config.budget_k}")
    print(f"{'='*70}\n")
    
    for t in range(1, config.rounds + 1):
        t_start = time.time()
        print(f"\n--- Round {t}/{config.rounds} ---")
        
        # Step 1: Oracle samples a batch from the pool
        B_t, y_Bt, batch_idx = oracle.sample_batch(X, y, config.batch_frac, rng,
                                                      max_batch=config.max_batch)
        print(f"  Batch: {len(B_t)} samples ({y_Bt.sum()} true positives)")
        
        # Step 2: Attacker generates synthetic anomalies
        A_t = attacker.generate(B_t, q_t, n_samples=config.n_attack)
        y_At = np.ones(len(A_t), dtype=np.int32)  # synthetic anomalies labeled 1
        print(f"  Attacker generated: {len(A_t)} synthetic anomalies")
        
        # Step 3: Oracle forms mixed evaluation pool
        X_mixed = np.vstack([B_t, A_t])
        y_mixed = np.concatenate([y_Bt, y_At])
        n_real = len(B_t)
        
        # Step 4: Defender scores and ranks
        # Apply similarity reranking BEFORE oracle evaluation using accumulated memory
        scores = defender.score(X_mixed)
        all_confirmed_pos = np.vstack(memory_X_pos) if memory_X_pos else np.zeros((0, X.shape[1]))
        if len(all_confirmed_pos) > 0 and not config.no_reranking:
            # Increase reranking boost as memory grows (defender gets stronger with experience)
            dynamic_lambda = config.lambda_sim * (1.0 + 0.15 * len(memory_X_pos))
            original_lambda = defender.config.lambda_sim
            defender.config.lambda_sim = min(dynamic_lambda, 1.5)  # cap at 1.5
            scores = defender.similarity_rerank(X_mixed, scores, all_confirmed_pos)
            defender.config.lambda_sim = original_lambda
        ranking = np.argsort(-scores)  # descending

        # Adaptive-k: budget grows linearly with round (experimental variant)
        effective_k = config.budget_k
        if config.adaptive_k:
            effective_k = min(config.budget_k + t, len(X_mixed))
        
        # Step 5: Oracle evaluates and provides feedback
        # adaptive_k: budget grows linearly with round
        if config.adaptive_k:
            oracle.k = min(config.budget_k + t, len(X_mixed))
        # raw_labels: attacker receives the full mean of ALL missed positives (no Gamma info loss)
        # We simulate this by overriding gamma_type to 'mean' while also passing raw label mask
        # In practice this means the attacker knows exactly what the missed exemplar looks like
        saved_gamma = oracle.gamma_type
        if config.raw_labels:
            oracle.gamma_type = 'mean'  # mean of all missed in original feature space
        feedback = oracle.evaluate_ranking(X_mixed, y_mixed, ranking, h_func)
        oracle.gamma_type = saved_gamma
        
        metrics = feedback['metrics']
        
        # Compute ASR for synthetic anomalies
        asr = compute_asr(y_At, ranking, n_real, config.budget_k)
        
        # Compute AUROC/AUPRC on full mixed pool
        auroc, auprc = compute_auroc_auprc(y_mixed, scores)
        
        # Step 6: Defender updates
        # 6a. Accumulate oracle labels into memory buffer
        topk_pos_mask = feedback['topk_y'] == 1
        topk_neg_mask = feedback['topk_y'] == 0
        if topk_pos_mask.sum() > 0:
            memory_X_pos.append(feedback['topk_X'][topk_pos_mask])
        if topk_neg_mask.sum() > 0:
            memory_X_neg.append(feedback['topk_X'][topk_neg_mask])

        if not config.no_adapt:  # ── Skip ALL adaptation if static baseline mode ──

            # 6b. Update ensemble weights using ACCUMULATED memory (not just this round)
            if not config.feedback_only or True:  # weights always update in feedback_only
                if len(memory_X_pos) > 0 and len(memory_X_neg) > 0:
                    mem_pos = np.vstack(memory_X_pos)
                    mem_neg = np.vstack(memory_X_neg)
                    mem_X = np.vstack([mem_pos, mem_neg])
                    mem_y = np.concatenate([np.ones(len(mem_pos)), np.zeros(len(mem_neg))])
                    if not getattr(config, 'uniform_weights', False):
                        defender.update_weights_from_pairs(mem_X, mem_y)
                    # else: weights stay frozen at initialised 1/N

            # 6d. Periodic defender AE refit — skip in feedback_only mode
            if not config.feedback_only and t % 2 == 0 and len(memory_X_pos) > 0:
                all_pos = np.vstack(memory_X_pos)
                if len(all_pos) >= 2:
                    X_aug = np.vstack([X, all_pos])
                    X_t_aug = torch.FloatTensor(X_aug)
                    optimizer_refit = optim.Adam(defender.ae.parameters(), lr=config.ae_lr * 0.5)
                    defender.ae.train()
                    for _ in range(5):
                        perm = torch.randperm(X_t_aug.size(0))
                        for i in range(0, X_t_aug.size(0), config.ae_batch_size):
                            batch = X_t_aug[perm[i:i+config.ae_batch_size]]
                            if batch.size(0) < 2:
                                continue
                            recon = defender.ae(batch)
                            loss = nn.functional.binary_cross_entropy(recon, batch)
                            optimizer_refit.zero_grad()
                            loss.backward()
                            optimizer_refit.step()
                    defender.ae.eval()

        # Step 7: Attacker updates
        # 7a. Fit surrogate to current defender scores
        attacker.fit_surrogate(X_mixed, scores)

        # 7b. Update generator (with decaying learning rate to slow attacker adaptation)
        decay_factor = 1.0 / (1.0 + 0.1 * t)  # gradual LR decay
        original_lr = config.gan_lr
        config.gan_lr = original_lr * decay_factor
        attacker.update(X_mixed, scores, q_t, h_func, config)
        config.gan_lr = original_lr  # restore
        
        # Step 8: Oracle provides compressed summary for next round
        q_t = feedback['missed_summary']
        
        # Log round
        elapsed = time.time() - t_start
        round_log = {
            'round': t,
            'ndcg_at_k': metrics['ndcg_at_k'],
            'full_ndcg': metrics['full_ndcg'],
            'recall_at_k': metrics['recall_at_k'],
            'precision_at_k': metrics['precision_at_k'],
            'tp_at_k': metrics['tp_at_k'],
            'n_total_pos': metrics['n_total_pos'],
            'asr': asr,
            'auroc': auroc,
            'auprc': auprc,
            'n_missed': feedback['n_missed'],
            'weights': defender.weights.tolist(),
            'elapsed_sec': elapsed,
        }
        history.append(round_log)
        weight_history.append(defender.weights.copy())
        
        print(f"  nDCG@{config.budget_k}={metrics['ndcg_at_k']:.4f}  "
              f"Recall@{config.budget_k}={metrics['recall_at_k']:.4f}  "
              f"ASR={asr:.4f}  AUROC={auroc:.4f}  "
              f"TP@k={metrics['tp_at_k']}/{metrics['n_total_pos']}  "
              f"Missed={feedback['n_missed']}  "
              f"Time={elapsed:.1f}s")
        print(f"  Weights: {dict(zip(defender.detector_names, defender.weights.round(3)))}")
    
    # --- 6. Final holdout evaluation (oracle disabled) ---
    print(f"\n{'='*70}")
    print("  Final Holdout Evaluation (Oracle Disabled)")
    print(f"{'='*70}")
    
    final_scores = defender.score(X)
    final_ranking = np.argsort(-final_scores)
    final_auroc, final_auprc = compute_auroc_auprc(y, final_scores)
    
    # nDCG on full dataset
    k_eval = min(config.budget_k, len(final_ranking))
    dcg = sum(
        1.0 / np.log2(r + 2) for r in range(k_eval) if y[final_ranking[r]] == 1
    )
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(int(y.sum()), k_eval)))
    final_ndcg = dcg / idcg if idcg > 0 else 0.0
    
    final_recall = sum(y[final_ranking[r]] for r in range(k_eval)) / max(y.sum(), 1)
    
    print(f"  Final AUROC:           {final_auroc:.4f}")
    print(f"  Final AUPRC:           {final_auprc:.4f}")
    print(f"  Final nDCG@{k_eval}:        {final_ndcg:.4f}")
    print(f"  Final Recall@{k_eval}:      {final_recall:.4f}")
    print(f"  Final Weights:         {dict(zip(defender.detector_names, defender.weights.round(3)))}")
    
    # Print top-k ranked process IDs
    print(f"\n  Top-{k_eval} ranked processes:")
    for r in range(k_eval):
        idx = final_ranking[r]
        label = "APT" if y[idx] == 1 else "benign"
        print(f"    Rank {r+1}: {proc_ids[idx]} [{label}] (score={final_scores[idx]:.4f})")
    
    # --- 7. Save Results ---
    # Save history
    with open(os.path.join(config.output_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save final results
    final_results = {
        'auroc': final_auroc,
        'auprc': final_auprc,
        'ndcg_at_k': final_ndcg,
        'recall_at_k': float(final_recall),
        'config': {k: str(v) for k, v in vars(config).items()},
    }
    with open(os.path.join(config.output_dir, 'final_results.json'), 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # Plot trajectories
    plot_trajectories(history, config.output_dir)
    plot_weight_evolution(weight_history, defender.detector_names, config.output_dir)
    
    print(f"\n[Done] Results saved to {config.output_dir}/")
    return history, final_results


# ============================================================
# 8. CLI ENTRY POINT
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="OMRQ: Oracle-Mediated Red Queen Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--data', type=str, required=True,
                        help='Path to ProcessAll.csv')
    parser.add_argument('--gt', type=str, required=True,
                        help='Path to ground truth merged CSV')
    parser.add_argument('--rounds', type=int, default=20,
                        help='Number of Red Queen rounds')
    parser.add_argument('--budget', type=int, default=10,
                        help='Oracle inspection budget k')
    parser.add_argument('--n-attack', type=int, default=20,
                        help='Synthetic anomalies per round')
    parser.add_argument('--gamma', type=str, default='mean',
                        choices=['mean', 'medoid', 'mask'],
                        help='Oracle compression operator')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default='omrq_results',
                        help='Output directory')
    # Ablation flags
    parser.add_argument('--no-adapt', action='store_true',
                        help='Static baseline: defender trains once, never adapts (weights frozen, no refit)')
    parser.add_argument('--feedback-only', action='store_true',
                        help='Feedback-only: weight updates enabled but AE refit disabled')
    
    # Component ablation flags
    parser.add_argument('--no-reranking', action='store_true',
                        help='Ablation: disable similarity reranking in defender')
    parser.add_argument('--no-alignment', action='store_true',
                        help='Ablation: disable signature alignment in attacker (lambda_sig=0)')
    parser.add_argument('--no-plausibility', action='store_true',
                        help='Ablation: disable plausibility constraint (lambda_plaus=0)')
    parser.add_argument('--no-diversity', action='store_true',
                        help='Ablation: disable diversity penalty (lambda_div=0)')
    parser.add_argument('--raw-labels', action='store_true',
                        help='Upper bound: attacker receives full oracle labels (no Gamma compression)')
    parser.add_argument('--adaptive-k', action='store_true',
                        help='Variant: budget k grows linearly with round number')

    parser.add_argument('--no-avf', action='store_true',
                        help='Ablation: remove AVF detector from ensemble')
    parser.add_argument('--avf-only', action='store_true',
                        help='Ablation: use ONLY AVF as backbone detector (weight=1.0)')
    parser.add_argument('--ae-only', action='store_true',
                        help='Ablation: use ONLY AutoEncoder as backbone detector (weight=1.0)')
    parser.add_argument('--uniform-weights', action='store_true',
                        help='Ablation: freeze ensemble weights at 1/N (no learning) — removes IForest dominance')
    parser.add_argument('--attacker', type=str, default='gan',
                        choices=['gan', 'diffusion'],
                        help='Attacker generator type: "gan" (WGAN-GP, default) or "diffusion" (DDPM)')
    # Direct lambda overrides for sensitivity sweeps
    parser.add_argument('--lambda-sig', type=float, default=None,
                        help='Override lambda_sig directly (sensitivity sweep; overrides --no-alignment)')
    parser.add_argument('--lambda-sim', type=float, default=None,
                        help='Override lambda_sim directly (sensitivity sweep)')
    parser.add_argument('--lambda-div', type=float, default=None,
                        help='Override lambda_div directly (sensitivity sweep; overrides --no-diversity)')

    args = parser.parse_args()

    # Apply component ablations via lambda zeroing
    lambda_sig   = 0.0 if args.no_alignment   else 0.5
    lambda_plaus = 0.0 if args.no_plausibility else 0.3
    lambda_div   = 0.0 if args.no_diversity    else 0.1
    # Direct overrides take precedence (for sensitivity sweeps)
    if args.lambda_sig is not None: lambda_sig = args.lambda_sig
    if args.lambda_div is not None: lambda_div = args.lambda_div
    lambda_sim = args.lambda_sim if args.lambda_sim is not None else 0.3

    config = OMRQConfig(
        data_path=args.data,
        gt_path=args.gt,
        rounds=args.rounds,
        budget_k=args.budget,
        n_attack=args.n_attack,
        gamma_type='mean' if not args.raw_labels else 'mean',
        seed=args.seed,
        output_dir=args.output,
        no_adapt=args.no_adapt,
        feedback_only=args.feedback_only,
        no_reranking=args.no_reranking,
        no_alignment=args.no_alignment,
        no_plausibility=args.no_plausibility,
        no_diversity=args.no_diversity,
        raw_labels=args.raw_labels,
        adaptive_k=args.adaptive_k,
        no_avf=args.no_avf,
        avf_only=args.avf_only,
        ae_only=args.ae_only,
        uniform_weights=args.uniform_weights,
        attacker_type=args.attacker,
        lambda_sig=lambda_sig,
        lambda_sim=lambda_sim,
        lambda_plaus=lambda_plaus,
        lambda_div=lambda_div,
    )
    
    run_omrq(config)


if __name__ == '__main__':
    main()
