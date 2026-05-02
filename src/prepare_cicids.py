#!/usr/bin/env python3
"""
prepare_cicids.py
=================
Downloads (or uses local copies of) CICIDS-2017 and converts it into the
two-file format expected by omrq_pipeline.py:

  <out_dir>/cicids_features.csv   — Object_ID + normalised feature columns
  <out_dir>/cicids_gt.csv         — uuid + label  (label='AdmSubject::Node'
                                     for every attack flow, mimicking DARPA fmt)

The script works with ANY of the standard CICIDS-2017 release formats:
  • The UNB "GeneratedLabelledFlows" CSVs (one file per day, ~1GB total)
  • The Kaggle "MachineLearningCSV" processed single CSV
  • The "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv" etc. split files

Usage
-----
  # Auto-download (tries public mirrors):
  python prepare_cicids.py --out data/cicids

  # Use files you already downloaded:
  python prepare_cicids.py \
      --files /path/to/Monday-WorkingHours.pcap_ISCX.csv \
              /path/to/Tuesday-WorkingHours.pcap_ISCX.csv \
      --out data/cicids

  # Use Kaggle single-file version:
  python prepare_cicids.py --files /path/to/cicids2017.csv --out data/cicids

Manual download
---------------
  Option A — UNB official (requires form):
    https://www.unb.ca/cic/datasets/ids-2017.html

  Option B — Kaggle (requires API key):
    kaggle datasets download -d cicdataset/cicids2017
    # or
    kaggle datasets download -d mlg-ulb/creditcardfraud   (different - just an example)

  Option C — Direct CSV mirror (small processed version, ~150 MB):
    See URLs in _MIRRORS below.

Notes
-----
  - ~2.8M flow records (all days combined)
  - 78 continuous features per flow (duration, bytes, packets, IATs, flags, etc.)
  - Binary label: BENIGN vs attack class string
  - Attack types: DoS, DDoS, PortScan, BruteForce, XSS, SQL Injection,
                  Infiltration, Botnet, Heartbleed
  - Anomaly rate varies by file: ~6-25% overall
  - Pipeline note: AE uses MSE loss on continuous data (auto-detected)
"""

import argparse
import os
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ── Download strategy ──────────────────────────────────────────────────────
_KAGGLE_DATASET = "cicdataset/cicids2017"  # kaggle datasets download -d ...

# ── Columns to drop (identifiers / timestamps / leakage) ────────────────────
_DROP_COLS = [
    'Flow ID', ' Flow ID',
    ' Source IP', 'Source IP', 'Src IP',
    ' Destination IP', 'Destination IP', 'Dst IP',
    ' Source Port', 'Source Port', 'Src Port',
    ' Destination Port', 'Destination Port', 'Dst Port',
    ' Timestamp', 'Timestamp',
    'Label', ' Label',              # target — removed after extraction
    'External IP',
]

# ── Columns that are categorical and need one-hot encoding ───────────────────
_CAT_COLS = ['Protocol', ' Protocol']  # 0=TCP, 6=UDP, 17=ICMP — treat as cat


# =============================================================================
# Download helpers
# =============================================================================

def _try_download(url: str, dest: Path) -> bool:
    try:
        print(f"  Trying: {url[:90]}...")
        urllib.request.urlretrieve(url, dest)
        size = dest.stat().st_size
        if size < 500:
            dest.unlink(); return False
        print(f"  ✓ Downloaded ({size / 1e6:.1f} MB)")
        return True
    except Exception as e:
        print(f"  ✗ {e}")
        if dest.exists():
            dest.unlink()
        return False


def _try_kaggle(out_dir: Path) -> list:
    """Try kaggle CLI download. Returns list of found CSVs on success."""
    import shutil, subprocess
    if not shutil.which('kaggle'):
        return []
    print("[Kaggle] Found kaggle CLI — attempting download...")
    try:
        result = subprocess.run(
            ['kaggle', 'datasets', 'download', '-d', _KAGGLE_DATASET,
             '-p', str(out_dir), '--unzip'],
            capture_output=True, text=True, timeout=600
        )
        if result.returncode != 0:
            print(f"  Kaggle error: {result.stderr[:300]}")
            return []
        csvs = sorted(out_dir.glob('*.csv'))
        if csvs:
            print(f"  ✓ Downloaded {len(csvs)} CSV file(s)")
            return csvs
    except Exception as e:
        print(f"  Kaggle exception: {e}")
    return []


def download_files(out_dir: Path) -> list:
    """Acquire CICIDS-2017 via Kaggle CLI or fail with manual instructions."""
    existing = sorted(out_dir.glob('*.csv'))
    if existing:
        print(f"[Cache] Found {len(existing)} existing CSV file(s)")
        return existing

    # Try Kaggle first
    result = _try_kaggle(out_dir)
    if result:
        return result

    # Try sample mirror (one day file)
    sample = out_dir / "cicids_friday_ddos.csv"
    sample_url = ("https://raw.githubusercontent.com/abhishekdabas31/"
                  "Network-Intrusion-Detection-System/master/Dataset/"
                  "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
    print("\n[Direct] Trying sample mirror (Friday DDoS file)...")
    if _try_download(sample_url, sample):
        print("  Note: This is a single-day sample. For full dataset use Kaggle.")
        return [sample]

    # All failed
    print("""
[!] Auto-download failed. Choose one of these options:

Option A — Kaggle CLI (recommended, full dataset ~450 MB):
  1. pip install kaggle
  2. Get API key from https://www.kaggle.com/settings → 'Create New Token'
  3. Save to ~/.kaggle/kaggle.json
  4. Re-run this script

Option B — UNB official (requires registration form):
  1. https://www.unb.ca/cic/datasets/ids-2017.html
  2. Download 'GeneratedLabelledFlows.zip' (~450 MB)
  3. Unzip and re-run:
     python prepare_cicids.py \\
         --files MachineLearningCVE/*.csv \\
         --out data/cicids

Option C — Use any single day file you have:
     python prepare_cicids.py \\
         --files Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv \\
         --out data/cicids
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
        try:
            df = pd.read_csv(p, encoding='utf-8',  low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(p, encoding='latin-1', low_memory=False)

        # Normalise column names: strip whitespace
        df.columns = [c.strip() for c in df.columns]
        print(f"  Shape: {df.shape} | Cols sample: {list(df.columns[:5])}")
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    print(f"[Load] Combined shape: {combined.shape}")
    return combined


def preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Preprocess CICIDS dataframe → (features_df, gt_df)."""

    # ── Find label column ────────────────────────────────────────────────────
    label_col = None
    for c in df.columns:
        if c.strip().lower() in ('label', 'class', 'attack', 'attack type', 'attack_type', 'type'):
            label_col = c
            break
    if label_col is None:
        raise ValueError("Cannot find 'Label' column. Found: " + str(df.columns.tolist()[:15]))

    raw_labels = df[label_col].astype(str).str.strip()
    BENIGN_VALUES = {'benign', '0', 'normal', 'normal traffic', 'normaltraffic', ''}
    labels = (~raw_labels.str.lower().isin(BENIGN_VALUES)).astype(int)

    print(f"  Label distribution:\n{raw_labels.value_counts().head(10)}")
    print(f"  → Binary: {labels.sum()} attacks / {len(labels)} total "
          f"({100*labels.mean():.2f}%)")

    # ── Assign Object_ID ─────────────────────────────────────────────────────
    ids = pd.Series([f"Entity{i}" for i in range(len(df))], name="Object_ID")

    # ── Drop non-feature cols ────────────────────────────────────────────────
    drop = [c for c in _DROP_COLS if c in df.columns] + [label_col]
    feat = df.drop(columns=drop, errors='ignore')

    # ── One-hot encode protocol if still categorical (sometimes integer) ──────
    for c in list(feat.columns):
        if c.strip() in [cc.strip() for cc in _CAT_COLS]:
            feat = pd.get_dummies(feat, columns=[c],
                                   prefix=c.strip(), drop_first=False, dtype=float)

    # ── Force numeric ────────────────────────────────────────────────────────
    feat = feat.apply(pd.to_numeric, errors='coerce')

    # ── Replace Inf / NaN ────────────────────────────────────────────────────
    feat = feat.replace([np.inf, -np.inf], np.nan)

    # Cap extreme values at 99th percentile per column before scaling
    for col in feat.columns:
        cap = feat[col].quantile(0.99)
        if pd.notna(cap) and cap > 0:
            feat[col] = feat[col].clip(upper=cap)

    feat = feat.fillna(feat.median(numeric_only=True)).fillna(0)

    print(f"  Feature dims after encoding: {feat.shape[1]}")

    # ── MinMaxScale to [0, 1] ────────────────────────────────────────────────
    scaler = MinMaxScaler()
    feat_scaled = pd.DataFrame(
        scaler.fit_transform(feat),
        columns=feat.columns,
        dtype=np.float32
    )

    # ── Subsample if very large (>500K rows, preserving original class ratio) ─
    MAX_ROWS = 500_000
    if len(feat_scaled) > MAX_ROWS:
        attack_rate = labels.mean()
        n_attack = int(MAX_ROWS * attack_rate)
        n_normal = MAX_ROWS - n_attack
        print(f"  [Subsample] {len(feat_scaled):,} → {MAX_ROWS:,} rows "
              f"(stratified, preserving {100*attack_rate:.1f}% attack rate)")
        rng = np.random.default_rng(42)
        normal_idx = np.where(labels == 0)[0]
        attack_idx = np.where(labels == 1)[0]
        chosen_attack = rng.choice(attack_idx, min(n_attack, len(attack_idx)), replace=False)
        chosen_normal = rng.choice(normal_idx, min(n_normal, len(normal_idx)), replace=False)
        chosen = np.concatenate([chosen_attack, chosen_normal])
        chosen.sort()
        feat_scaled = feat_scaled.iloc[chosen].reset_index(drop=True)
        labels      = labels.iloc[chosen].reset_index(drop=True)
        ids         = ids.iloc[chosen].reset_index(drop=True)

    # ── Assemble outputs ─────────────────────────────────────────────────────
    features_df = pd.concat([ids.reset_index(drop=True),
                              feat_scaled.reset_index(drop=True)], axis=1)

    gt_df = pd.DataFrame({
        'uuid':  ids[labels == 1].reset_index(drop=True),
        'label': 'AdmSubject::Node',
    })

    print(f"  Final features shape : {features_df.shape}")
    print(f"  GT records (attacks) : {len(gt_df)}")
    return features_df, gt_df


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Download + preprocess CICIDS-2017 for omrq_pipeline.py"
    )
    parser.add_argument('--files', nargs='+', default=None,
                        help='Local CICIDS CSV file(s) — skip download')
    parser.add_argument('--out', type=str, default='data/cicids',
                        help='Output directory')
    parser.add_argument('--max-rows', type=int, default=500_000,
                        help='Max rows to keep (stratified subsample if exceeded)')
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Acquire raw files ────────────────────────────────────────────────────
    if args.files:
        raw_files = args.files
        print(f"[Info] Using provided files: {[str(p) for p in raw_files]}")
    else:
        raw_files = download_files(out_dir)

    # ── Load + preprocess ────────────────────────────────────────────────────
    df = load_raw(raw_files)
    features_df, gt_df = preprocess(df)

    # ── Save ─────────────────────────────────────────────────────────────────
    feat_path = out_dir / 'cicids_features.csv'
    gt_path   = out_dir / 'cicids_gt.csv'

    features_df.to_csv(feat_path, index=False)
    gt_df.to_csv(gt_path,        index=False)

    print(f"\n[Done] Files written:")
    print(f"  Features : {feat_path}  {features_df.shape}")
    print(f"  GT       : {gt_path}   ({len(gt_df)} attack records)")
    print(f"\nRun OMRQ with:")
    print(f"  python omrq_pipeline.py \\")
    print(f"    --data {feat_path} \\")
    print(f"    --gt   {gt_path} \\")
    print(f"    --rounds 20 --budget 30 --n-attack 20 --seed 42")


if __name__ == '__main__':
    main()
