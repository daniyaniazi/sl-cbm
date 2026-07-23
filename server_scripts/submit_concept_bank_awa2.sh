#!/usr/bin/env bash
# Build concept bank for AwA2 using ResNet101 ImageNet backbone.
# Run before submit_slcbm_awa2.sh.

set -euo pipefail

PYTHON="/home/dani00003/miniconda3/envs/sl-cbm/bin/python"
SLCBM_DIR="/home/dani00003/sl-cbm"

mkdir -p "$SLCBM_DIR/logs"
mkdir -p "$SLCBM_DIR/concept_banks"

echo "universe                = docker
docker_image            = pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime
executable              = $PYTHON
arguments               = training_tools/learn_concepts_awa2.py --backbone-name resnet101_imagenet --backbone-ckpt none --out-dir concept_banks/ --C 0.1 --n-samples 10
environment             = \"PYTHONPATH=/home/dani00003/pcbm-module:/home/dani00003/sl-cbm\"
initialdir              = $SLCBM_DIR

output                  = $SLCBM_DIR/logs/concept_bank_awa2.\$(ClusterId).\$(ProcId).out
error                   = $SLCBM_DIR/logs/concept_bank_awa2.\$(ClusterId).\$(ProcId).err
log                     = $SLCBM_DIR/logs/concept_bank_awa2.\$(ClusterId).log

request_GPUs            = 1
request_CPUs            = 4
request_memory          = 16G
requirements            = UidDomain == \"cs.uni-saarland.de\"
+WantGPUHomeMounted     = true

queue 1" | condor_submit

echo "AwA2 concept bank job submitted"
echo "Output: $SLCBM_DIR/concept_banks/awa2_resnet101_imagenet_0.1_20.pkl"
echo "When done: ./server_scripts/submit_slcbm_awa2.sh"

condor_q