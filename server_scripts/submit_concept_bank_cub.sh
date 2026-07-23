#!/usr/bin/env bash
# Build concept bank for CUB using CREAM backbone.
# Run before submit_slcbm.sh
#
# USAGE:
#   ./server_scripts/submit_concept_bank_cub.sh

set -euo pipefail

PYTHON="/home/dani00003/.venvs/guide/bin/python"
SLCBM_DIR="/home/dani00003/sl-cbm"
CREAM_CKPT="/home/dani00003/GUIDE-cbm-WIP/models/resnet18_finetuned.pth"

mkdir -p "$SLCBM_DIR/logs"
mkdir -p "$SLCBM_DIR/concept_banks"

echo "universe                = docker
docker_image            = pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime
executable              = $PYTHON
arguments               = training_tools/learn_concepts_dataset.py --backbone-name resnet18_cream --backbone-ckpt $CREAM_CKPT --dataset-name cub --out-dir concept_banks/ --C 0.1 --n-samples 100
initialdir              = $SLCBM_DIR

output                  = $SLCBM_DIR/logs/concept_bank_cub.\$(ClusterId).\$(ProcId).out
error                   = $SLCBM_DIR/logs/concept_bank_cub.\$(ClusterId).\$(ProcId).err
log                     = $SLCBM_DIR/logs/concept_bank_cub.\$(ClusterId).log

request_GPUs            = 1
request_CPUs            = 4
request_memory          = 16G
requirements            = UidDomain == \"cs.uni-saarland.de\"
+WantGPUHomeMounted     = true
queue 1" | condor_submit

echo "CUB concept bank job submitted"
echo "Output: concept_banks/cub_resnet18_cream_0.1_100.pkl"
echo "When done → run: ./server_scripts/submit_slcbm.sh"
condor_q
