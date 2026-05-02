#!/bin/bash
# Two-Player Game (--raw-labels) — DARPA TC datasets only
# NSL-KDD U2R and UNSW-NB15 use a separate network pipeline (not covered here)
# Each run ~2 min sequentially

set -e
BASE=/Users/sb/Documents/Projects/GAN/GansBinarySequence2/APT-AutoEncoders/AE-APT
SRC=$BASE/src
DATA=$BASE/data
OUT=$SRC/omrq_results_twoplayer

mkdir -p $OUT

cd $SRC

echo "========================================="
echo "[1/4] Clearscope (Bovia)"
echo "========================================="
python3 omrq_pipeline.py \
  --data  $DATA/bovia/clearscope/ProcessAll.csv \
  --gt    $DATA/bovia/clearscope/clearscope_bovia_lobiwapp.csv \
  --rounds 20 --budget 50 --n-attack 20 \
  --raw-labels \
  --output $OUT/clearscope \
  2>&1 | tee $OUT/clearscope.log
echo "Done Clearscope"

echo "========================================="
echo "[2/4] Cadets (Bovia)"
echo "========================================="
python3 omrq_pipeline.py \
  --data  $DATA/bovia/cadets/ProcessAll.csv \
  --gt    $DATA/bovia/cadets/cadets_bovia_webshell.csv \
  --rounds 20 --budget 50 --n-attack 20 \
  --raw-labels \
  --output $OUT/cadets \
  2>&1 | tee $OUT/cadets.log
echo "Done Cadets"

echo "========================================="
echo "[3/4] Trace/Linux (Bovia)"
echo "========================================="
python3 omrq_pipeline.py \
  --data  $DATA/bovia/trace/ProcessAll.csv \
  --gt    $DATA/bovia/trace/trace_bovia_simple.csv \
  --rounds 20 --budget 50 --n-attack 20 \
  --raw-labels \
  --output $OUT/trace \
  2>&1 | tee $OUT/trace.log
echo "Done Trace"

echo "========================================="
echo "[4/4] 5Dir (Bovia)"
echo "========================================="
python3 omrq_pipeline.py \
  --data  $DATA/bovia/5dir/ProcessAll.csv \
  --gt    $DATA/bovia/5dir/5dir_bovia_simple.csv \
  --rounds 20 --budget 50 --n-attack 20 \
  --raw-labels \
  --output $OUT/5dir \
  2>&1 | tee $OUT/5dir.log
echo "Done 5Dir"

echo ""
echo "========================================="
echo "ALL DARPA RUNS DONE."
echo "Printing per-run summary from logs..."
echo "========================================="
for ds in clearscope cadets trace 5dir; do
  echo ""
  echo "--- $ds ---"
  grep -E "AvgAUROC|AvgNDCG|AvgASR|avg_auroc|avg_ndcg|avg_asr|\[Summary\]|AUROC|nDCG.*avg|ASR.*avg" $OUT/${ds}.log 2>/dev/null | tail -6
done
