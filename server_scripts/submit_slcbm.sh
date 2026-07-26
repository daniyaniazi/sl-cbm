#!/usr/bin/env bash
# SL-CBM baseline on CUB-200-2011 using resnet18_cub (pytorchcv, auto-downloaded).
# This is a fair comparison with GUIDE-CBM also running resnet18_cub.
#
# PREREQUISITES (run once):
#   1. Download concept_banks.zip from SL-CBM Google Drive (README link)
#      unzip concept_banks.zip -d ~/sl-cbm/
#      → provides: concept_banks/cub_resnet18_cub_0.1_100.pkl
#   2. mkdir -p ~/sl-cbm/model_zoo/resnet_cub
#      (pytorchcv will auto-download resnet18_cub weights here on first run)
#
# USAGE:
#   ./server_scripts/submit_slcbm.sh

set -euo pipefail

PYTHON="/home/dani00003/miniconda3/envs/sl-cbm/bin/python"
SLCBM_DIR="/home/dani00003/sl-cbm"

mkdir -p "$SLCBM_DIR/logs"

echo "universe                = docker
docker_image            = pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime
executable              = $PYTHON
arguments               = spss_vl_cbm_train.py --backbone-name resnet18_cream --backbone-ckpt /home/dani00003/GUIDE-cbm-WIP/models/resnet18_finetuned.pth --concept-bank concept_banks/cub_resnet18_cream_0.1_100.pkl --dataset spss_cub --target-dataset spss_cub --pcbm-arch spss_pcbm --batch-size 8 --epoch 450 --lr 1e-4 --lambda1 1.0 --lambda2 100.0 --lambda3 1.0 --intervention --explain-method builtin_explain
environment             = \"PYTHONPATH=/home/dani00003/sl-cbm:/home/dani00003/pcbm-module\"
initialdir              = $SLCBM_DIR

output                  = $SLCBM_DIR/logs/slcbm_resnet18cub.\$(ClusterId).\$(ProcId).out
error                   = $SLCBM_DIR/logs/slcbm_resnet18cub.\$(ClusterId).\$(ProcId).err
log                     = $SLCBM_DIR/logs/slcbm_resnet18cub.\$(ClusterId).log

request_GPUs            = 1
request_CPUs            = 8
request_memory          = 32G
requirements            = UidDomain == \"cs.uni-saarland.de\"
+WantGPUHomeMounted     = true
queue 1" | condor_submit

echo "SL-CBM job submitted (resnet18_cub, pytorchcv weights)"
condor_q
