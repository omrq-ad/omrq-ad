#!/bin/bash
# Feedback-Only (--feedback-only) runs on all datasets missing it
set -e

BASE=/Users/sb/Documents/Projects/GAN/GansBinarySequence2/APT-AutoEncoders/AE-APT
SRC=$BASE/src
DATA=$BASE/data
MI=/Users/sb/Documents/Projects/GAN/Machine-Intelligence-v1.0/data/MiscCategoricalData
OUT=$SRC/omrq_results_feedback

mkdir -p $OUT
cd $SRC

echo "[1/5] Clearscope (Bovia) -- Feedback-Only"
python3 omrq_pipeline.py \
  --data $DATA/bovia/clearscope/ProcessAll.csv \
  --gt   $DATA/bovia/clearscope/clearscope_bovia_lobiwapp.csv \
  --rounds 20 --budget 50 --n-attack 20 \
  --feedback-only --output $OUT/clearscope \
  2>&1 | tee $OUT/clearscope.log
echo "Done Clearscope"

echo "[2/5] Cadets (Bovia) -- Feedback-Only"
python3 omrq_pipeline.py \
  --data $DATA/bovia/cadets/ProcessAll.csv \
  --gt   $DATA/bovia/cadets/cadets_bovia_webshell.csv \
  --rounds 20 --budget 50 --n-attack 20 \
  --feedback-only --output $OUT/cadets \
  2>&1 | tee $OUT/cadets.log
echo "Done Cadets"

echo "[3/5] 5Dir (Bovia) -- Feedback-Only"
python3 omrq_pipeline.py \
  --data $DATA/bovia/5dir/ProcessEvent.csv \
  --gt   $DATA/bovia/5dir/5dir_bovia_simple.csv \
  --rounds 20 --budget 50 --n-attack 20 \
  --feedback-only --output $OUT/5dir \
  2>&1 | tee $OUT/5dir.log
echo "Done 5Dir"

echo "[4/5] KDD U2R -- Feedback-Only"
python3 omrq_pipeline.py \
  --data $MI/kddcup99-corrected-u2rvsnormal-nominal-cleaned.csv \
  --gt   $MI/kddcup99-corrected-u2rvsnormal-nominal-cleaned_gt.csv \
  --rounds 20 --budget 30 --n-attack 15 \
  --feedback-only --output $OUT/kdd_u2r \
  2>&1 | tee $OUT/kdd_u2r.log
echo "Done KDD U2R"

echo "[5/5] UNSW-NB15 -- Feedback-Only"
python3 omrq_pipeline.py \
  --data $SRC/data/unsw_nb15/unsw_nb15_features.csv \
  --gt   $SRC/data/unsw_nb15/unsw_nb15_gt.csv \
  --rounds 20 --budget 50 --n-attack 20 \
  --feedback-only --output $OUT/unsw \
  2>&1 | tee $OUT/unsw.log
echo "Done UNSW"

echo "ALL DONE"
