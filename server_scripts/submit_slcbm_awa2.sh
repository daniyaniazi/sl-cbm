#!/usr/bin/env bash
# SL-CBM on AwA2 with CREAM backbone.
#
# PREREQUISITES (run once before submitting):
#   1. AwA2 images downloaded + all_features.pkl generated (see GUIDE-cbm-WIP AwA2 setup)
#   2. Build AwA2 concept bank (ResNet101 ImageNet weights, auto-downloaded):
#      cd ~/sl-cbm
#      python training_tools/learn_concepts_awa2.py \
#        --backbone-name resnet101_imagenet \
#        --backbone-ckpt none \
#        --out-dir concept_banks/ \
#        --C 0.1 --n-samples 50
#      → produces: concept_banks/awa2_resnet101_imagenet_0.1_50.pkl
#
# USAGE:
#   ./server_scripts/submit_slcbm_awa2.sh

set -euo pipefail

PYTHON="/home/dani00003/miniconda3/envs/sl-cbm/bin/python"
SLCBM_DIR="/home/dani00003/sl-cbm"
mkdir -p "$SLCBM_DIR/logs"

echo "universe                = docker
docker_image            = pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime
executable              = $PYTHON
arguments               = spss_vl_cbm_train.py --backbone-name resnet101_imagenet --backbone-ckpt none --concept-bank concept_banks/awa2_resnet101_imagenet_0.1_236.pkl --dataset spss_awa2 --target-dataset spss_awa2 --pcbm-arch spss_pcbm --batch-size 8 --epoch 450 --lr 1e-4 --lambda1 1.0 --lambda2 100.0 --lambda3 1.0 --intervention --explain-method builtin_explain
environment             = "PYTHONPATH=/home/dani00003/sl-cbm:/home/dani00003/pcbm-module"
initialdir              = $SLCBM_DIR

output                  = $SLCBM_DIR/logs/slcbm_awa2_cream.\$(ClusterId).\$(ProcId).out
error                   = $SLCBM_DIR/logs/slcbm_awa2_cream.\$(ClusterId).\$(ProcId).err
log                     = $SLCBM_DIR/logs/slcbm_awa2_cream.\$(ClusterId).log

request_GPUs            = 1
request_CPUs            = 8
request_memory          = 32G
requirements            = UidDomain == \"cs.uni-saarland.de\"
+WantGPUHomeMounted     = true
queue 1" | condor_submit

echo "SL-CBM AwA2 job submitted (CREAM backbone)"
condor_q
