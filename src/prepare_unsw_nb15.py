#!/usr/bin/env python3
"""
prepare_unsw_nb15.py
====================
Downloads (or uses a local copy of) UNSW-NB15 and converts it into the
two-file format expected by omrq_pipeline.py:

  <out_dir>/unsw_nb15_features.csv   — Object_ID + normalised feature columns
  <out_dir>/unsw_nb15_gt.csv         — uuid + label  (label='AdmSubject::Node'
                                        for every attack row, mimicking DARPA format)

Usage
-----
  # Auto-download (tries public mirrors):
  python prepare_unsw_nb15.py --out data/unsw_nb15

  # Use files you already downloaded manually:
  python prepare_unsw_nb15.py --files UNSW_NB15_training-set.csv UNSW_NB15_testing-set.csv \
                               --out data/unsw_nb15

Manual download (if auto-download fails)
-----------------------------------------
  1. Go to https://research.unsw.edu.au/projects/unsw-nb15-dataset
  2. Download:  UNSW_NB15_training-set.csv  (175 MB)
                UNSW_NB15_testing-set.csv   (48 MB)
  3. Re-run with --files pointing to those files.

  OR download the smaller Kaggle version (~50 MB total):
  kaggle datasets download -d mrwellsdavid/unsw-nb15

Notes on the dataset
---------------------
  - 257,673 records (training + testing combined)
  - 42 raw features, mix of continuous + categorical (proto, service, state)
  - Binary label: 0 = normal, 1 = attack (~14% attack rate)
  - 9 attack categories: Fuzzers, Analysis, Backdoor, DoS, Exploits,
    Generic, Reconnaissance, Shellcode, Worms
"""

import argparse
import os
import sys
import urllib.request
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ── Download strategy (tried in order) ───────────────────────────────────────
# 1. kaggle CLI  (fastest if user has API key set up)
# 2. Direct CSV mirrors on GitHub (small processed subsets)
# 3. Manual download instructions

_KAGGLE_DATASET = "mrwellsdavid/unsw-nb15"   # kaggle datasets download -d ...

_DIRECT_MIRRORS = [
    # Smaller pre-processed merged version (training + testing) from known repos
    ("https://raw.githubusercontent.com/jmnwong/UNSW-NB15-Dataset/"
     "master/UNSW_NB15_training-set.csv",
     "https://raw.githubusercontent.com/jmnwong/UNSW-NB15-Dataset/"
     "master/UNSW_NB15_testing-set.csv"),
]

# ── columns to drop (identifiers / leakage / non-feature) ───────────────────
_DROP_COLS = [
    'id', 'label', 'attack_cat',      # id + targets
    'srcip', 'dstip',                  # IP addresses — too specific
    'sport', 'dsport',                 # raw port numbers
]

# ── categorical columns to one-hot encode ───────────────────────────────────
_CAT_COLS = ['proto', 'service', 'state']

# ── UNSW-NB15 official feature names (42 features + label) ──────────────────
#    Used as fallback if CSV has no header
_FEATURE_NAMES = [
    'srcip','sport','dstip','dsport','proto','state','dur','sbytes','dbytes',
    'sttl','dttl','sloss','dloss','service','sload','dload','spkts','dpkts',
    'swin','dwin','stcpb','dtcpb','smeansz','dmeansz','trans_depth','res_bdy_len',
    'sjit','djit','stime','ltime','sintpkt','dintpkt','tcprtt','synack','ackdat',
    'is_sm_ips_ports','ct_state_ttl','ct_flw_http_mthd','is_ftp_login',
    'ct_ftp_cmd','ct_srv_src','ct_srv_dst','ct_dst_ltm','ct_src_ltm',
    'ct_src_dport_ltm','ct_dst_sport_ltm','ct_dst_src_ltm',
    'attack_cat','label',
]


# =============================================================================
# Download helpers
# =============================================================================

def _try_download(url: str, dest: Path) -> bool:
    """Try to download url → dest. Returns True on success."""
    try:
        print(f"  Trying: {url[:80]}...")
        urllib.request.urlretrieve(url, dest)
        if dest.stat().st_size < 1024:
            dest.unlink()
            return False
        print(f"  ✓ Downloaded → {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        if dest.exists():
            dest.unlink()
        return False


def _try_kaggle(out_dir: Path) -> list:
    """Try to download via kaggle CLI. Returns list of CSV paths on success."""
    import shutil, subprocess, zipfile
    if not shutil.which('kaggle'):
        return []
    print("[Kaggle] Found kaggle CLI — attempting download...")
    try:
        result = subprocess.run(
            ['kaggle', 'datasets', 'download', '-d', _KAGGLE_DATASET,
             '-p', str(out_dir), '--unzip'],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode != 0:
            print(f"  Kaggle error: {result.stderr[:200]}")
            return []
        csvs = list(out_dir.glob('*.csv'))
        if csvs:
            print(f"  ✓ Kaggle download complete: {[f.name for f in csvs]}")
            return csvs
    except Exception as e:
        print(f"  Kaggle exception: {e}")
    return []


def download_files(out_dir: Path) -> list:
    """Acquire UNSW-NB15 files via Kaggle CLI, direct mirrors, or fail with instructions."""
    existing = list(out_dir.glob('UNSW_NB15*.csv'))
    if existing:
        print(f"[Cache] Found existing files: {[f.name for f in existing]}")
        return existing

    # ── Try Kaggle CLI ───────────────────────────────────────────────────────
    result = _try_kaggle(out_dir)
    if result:
        return result

    # ── Try direct mirrors ───────────────────────────────────────────────────
    for train_url, test_url in _DIRECT_MIRRORS:
        train_out = out_dir / 'UNSW_NB15_training-set.csv'
        test_out  = out_dir / 'UNSW_NB15_testing-set.csv'
        print(f"\n[Direct] Trying GitHub mirror...")
        ok_t = _try_download(train_url, train_out)
        ok_e = _try_download(test_url,  test_out)
        if ok_t and ok_e:
            return [train_out, test_out]

    # ── All failed ───────────────────────────────────────────────────────────
    print("""
[!] Auto-download failed. Choose one of these options:

Option A — Kaggle CLI (recommended):
  1. pip install kaggle
  2. Get your API key from https://www.kaggle.com/settings → 'Create New Token'
  3. Save it to  ~/.kaggle/kaggle.json
  4. Re-run this script (it will use kaggle automatically)

Option B — Manual download from UNSW:
  1. Go to https://research.unsw.edu.au/projects/unsw-nb15-dataset
  2. Download UNSW_NB15_training-set.csv and UNSW_NB15_testing-set.csv
  3. Re-run with:
     python prepare_unsw_nb15.py \\
         --files UNSW_NB15_training-set.csv UNSW_NB15_testing-set.csv \\
         --out data/unsw_nb15
""")
    sys.exit(1)


# =============================================================================
# Preprocessing
# =============================================================================

def load_raw(paths) -> pd.DataFrame:
    frames = []
    for p in paths:
        p = Path(p)
        print(f"[Load] {p.name}  ({p.stat().st_size / 1e6:.1f} MB)")

        # Try UTF-8 first, fall back to latin-1 (raw files often have encoding issues)
        for enc in ('utf-8', 'latin-1'):
            try:
                df = pd.read_csv(p, low_memory=False, encoding=enc)
                break
            except UnicodeDecodeError:
                continue

        # Detect headerless raw files: if first column looks like an IP address
        # the raw 4-file UNSW release has NO header row
        first_col = str(df.columns[0]).strip()
        is_headerless = (
            first_col[0].isdigit() or           # starts with digit (IP or port)
            first_col.startswith('ï»¿') or       # BOM artefact
            '.' in first_col[:12]                # looks like IP
        )
        if is_headerless:
            print(f"  Detected headerless format — assigning official column names")
            # The raw files have 49 columns: 47 features + attack_cat + label
            if len(df.columns) == len(_FEATURE_NAMES):
                df = pd.read_csv(p, header=None, names=_FEATURE_NAMES,
                                 low_memory=False, encoding='latin-1')
            else:
                # Fallback: prepend synthetic names, last col = label
                df = pd.read_csv(p, header=None, low_memory=False, encoding='latin-1')
                df.columns = [f"F{i}" for i in range(len(df.columns) - 1)] + ['label']
        else:
            # Normalise column names (strip spaces, lowercase)
            df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]

        frames.append(df)
        print(f"  Shape: {df.shape}")

    combined = pd.concat(frames, ignore_index=True)
    # Normalise all column names after concat
    combined.columns = [c.strip().lower().replace(' ', '_') for c in combined.columns]
    print(f"[Load] Combined: {combined.shape}")
    return combined


def preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns
    -------
    features_df : DataFrame with Object_ID + normalised feature columns
    gt_df       : DataFrame with uuid + label  (attack rows only)
    """
    # ── Identify label column ────────────────────────────────────────────────
    label_col = None
    for c in ['label', 'class', 'attack']:
        if c in df.columns:
            label_col = c
            break
    if label_col is None:
        raise ValueError("Cannot find label column. Columns: " + str(df.columns.tolist()))

    # Make binary label
    labels = (df[label_col].astype(str).str.strip().str.lower()
              .map(lambda x: 0 if x in ('0', 'normal', 'benign', '') else 1)
              .fillna(0).astype(int))

    # ── Assign Object_ID ─────────────────────────────────────────────────────
    ids = pd.Series([f"Entity{i}" for i in range(len(df))], name="Object_ID")

    # ── Drop non-feature columns ─────────────────────────────────────────────
    drop = [c for c in _DROP_COLS if c in df.columns]
    feat = df.drop(columns=drop, errors='ignore')

    # ── Clean infinities / NaN ───────────────────────────────────────────────
    feat = feat.replace([np.inf, -np.inf], np.nan)

    # ── One-hot encode categoricals ──────────────────────────────────────────
    cat_present = [c for c in _CAT_COLS if c in feat.columns]
    if cat_present:
        feat = pd.get_dummies(feat, columns=cat_present, prefix=cat_present,
                              drop_first=False, dtype=float)
        print(f"  One-hot encoded {cat_present} → {len(feat.columns)} cols")

    # ── Force numeric (drop any remaining string columns) ───────────────────
    feat = feat.apply(pd.to_numeric, errors='coerce')

    # ── Fill remaining NaN with column median ────────────────────────────────
    feat = feat.fillna(feat.median(numeric_only=True))
    feat = feat.fillna(0)

    # ── MinMaxScale to [0, 1] ────────────────────────────────────────────────
    scaler = MinMaxScaler()
    feat_scaled = pd.DataFrame(
        scaler.fit_transform(feat),
        columns=feat.columns
    )

    n_attack = labels.sum()
    n_total  = len(labels)
    print(f"  Attacks: {n_attack} / {n_total} ({100*n_attack/n_total:.2f}%)")
    print(f"  Final feature dims: {feat_scaled.shape[1]}")

    # ── Assemble outputs ─────────────────────────────────────────────────────
    features_df = pd.concat([ids, feat_scaled], axis=1)

    gt_df = pd.DataFrame({
        'uuid':  ids[labels == 1],
        'label': 'AdmSubject::Node',
    }).reset_index(drop=True)

    return features_df, gt_df


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Download + preprocess UNSW-NB15 for omrq_pipeline.py"
    )
    parser.add_argument('--files', nargs='+', default=None,
                        help='Paths to local UNSW-NB15 CSV files (skip download)')
    parser.add_argument('--out', type=str, default='data/unsw_nb15',
                        help='Output directory')
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Acquire raw files ────────────────────────────────────────────────────
    if args.files:
        raw_files = args.files
        print(f"[Info] Using provided files: {raw_files}")
    else:
        raw_files = download_files(out_dir)

    # ── Load + preprocess ────────────────────────────────────────────────────
    df = load_raw(raw_files)
    features_df, gt_df = preprocess(df)

    # ── Save ─────────────────────────────────────────────────────────────────
    feat_path = out_dir / 'unsw_nb15_features.csv'
    gt_path   = out_dir / 'unsw_nb15_gt.csv'

    features_df.to_csv(feat_path, index=False)
    gt_df.to_csv(gt_path, index=False)

    print(f"\n[Done] Files written:")
    print(f"  Features : {feat_path}  ({features_df.shape})")
    print(f"  GT       : {gt_path}   ({len(gt_df)} attack records)")
    print(f"\nRun OMRQ with:")
    print(f"  python omrq_pipeline.py \\")
    print(f"    --data {feat_path} \\")
    print(f"    --gt   {gt_path} \\")
    print(f"    --rounds 20 --budget 30 --n-attack 15 --seed 42")


if __name__ == '__main__':
    main()
