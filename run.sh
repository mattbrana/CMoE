#!/usr/bin/env bash
# CMoE energy comparison pipeline.
#
# Stages (each stage logs to W&B):
#   PHASE 0  dense_baseline_eval     dense model PPL + GPU energy
#   PHASE 1  conversion              MoE carving energy
#   PHASE 2  post_conversion_eval    MoE PPL + energy (training-free)
#   PHASE 3  finetune                LoRA fine-tuning energy
#   PHASE 4  post_finetune_eval      MoE PPL + energy (after fine-tune)
#
# Usage:
#   bash run.sh download              # one-time: pre-download the HF model
#   bash run.sh                       # full pipeline
#   bash run.sh -- --skip-dense-baseline    # forward extra args to run_cmoe.py
#
# Override any variable from the environment, e.g.:
#   MODEL=meta-llama/Meta-Llama-3-8B DATASET=wikitext2 bash run.sh

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODEL="${MODEL:-meta-llama/Llama-2-7b-hf}"
DATASET="${DATASET:-wikitext2}"
NSHARED="${NSHARED:-2}"
NACTIVATED="${NACTIVATED:-2}"
NEXPERTS="${NEXPERTS:-16}"
NSAMPLES="${NSAMPLES:-64}"
EPOCH="${EPOCH:-1}"
EXTRA_LR="${EXTRA_LR:-0.001}"
BIAS_SPEED="${BIAS_SPEED:-0.001}"
WANDB_PROJECT="${WANDB_PROJECT:-cmoe-energy}"

if [ "${1:-}" = "download" ]; then
    shift
    python download_model.py "$MODEL" "$@"
    exit $?
fi

if [ "${1:-}" = "--" ]; then
    shift
fi

python run_cmoe.py "$MODEL" "$DATASET" \
    --new-eval \
    --nshared "$NSHARED" \
    --nactivated "$NACTIVATED" \
    --nexperts "$NEXPERTS" \
    --nsamples "$NSAMPLES" \
    --epoch "$EPOCH" \
    --extra-lr "$EXTRA_LR" \
    --bias-speed "$BIAS_SPEED" \
    --wandb-project "$WANDB_PROJECT" \
    "$@"
