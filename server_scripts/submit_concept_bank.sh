#!/usr/bin/env bash
# Submit concept bank generation for SL-CBM with CREAM weights.
# Run ONCE before submitting the training job.
# Output: ~/sl-cbm/concept_banks/cub_resnet18_cream_0.1_100.pkl

set -euo pipefail

PYTHON="/home/dani00003/.venvs/guide/bin/python"
SLCBM_DIR="/home/dani00003/sl-cbm"

mkdir -p "$SLCBM_DIR/logs"
mkdir -p "$SLCBM_DIR/concept_banks"

echo "universe                = docker
docker_image            = pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime
executable              = $PYTHON
arguments               = training_tools/learn_concepts_dataset.py --backbone-name resnet18_cream --backbone-ckpt /home/dani00003/GUIDE-cbm-WIP/models/resnet18_finetuned.pth --dataset-name cub --out-dir concept_banks/ --C 0.1 --n-samples 100
initialdir              = $SLCBM_DIR

output                  = $SLCBM_DIR/logs/concept_bank.\$(ClusterId).\$(ProcId).out
error                   = $SLCBM_DIR/logs/concept_bank.\$(ClusterId).\$(ProcId).err
log                     = $SLCBM_DIR/logs/concept_bank.\$(ClusterId).log

request_GPUs            = 1
request_CPUs            = 4
request_memory          = 16G
requirements            = UidDomain == \"cs.uni-saarland.de\"
+WantGPUHomeMounted     = true
queue 1" | condor_submit

echo "Concept bank job submitted. Monitor with: condor_q"
echo "Output will be: $SLCBM_DIR/concept_banks/cub_resnet18_cream_0.1_100.pkl"
