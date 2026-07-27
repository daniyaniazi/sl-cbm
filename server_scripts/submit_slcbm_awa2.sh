#!/usr/bin/env bash
# SL-CBM on AwA2 with ResNet101 ImageNet weights — 3 seeds.
#
# USAGE:
#   ./server_scripts/submit_slcbm_awa2.sh

set -euo pipefail

PYTHON="/home/dani00003/miniconda3/envs/sl-cbm/bin/python"
SLCBM_DIR="/home/dani00003/sl-cbm"
mkdir -p "$SLCBM_DIR/logs"

SEEDS=(42 55 27)

for SEED in "${SEEDS[@]}"; do

    LOGBASE="$SLCBM_DIR/logs/slcbm_awa2_seed${SEED}"

    echo "  → Submitting AwA2 seed=$SEED"

    echo "universe                = docker
docker_image            = pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime
executable              = $PYTHON
arguments               = spss_vl_cbm_train.py --backbone-name resnet101_imagenet --backbone-ckpt none --concept-bank concept_banks/awa2_resnet101_imagenet_0.1_236.pkl --dataset spss_awa2 --target-dataset spss_awa2 --pcbm-arch spss_pcbm --batch-size 8 --epoch 450 --lr 1e-4 --lambda1 1.0 --lambda2 100.0 --lambda3 1.0 --intervention --explain-method builtin_explain --universal-seed $SEED --exp-name slcbm_awa2_seed${SEED}
environment             = "PYTHONPATH=$SLCBM_DIR:/home/dani00003/pcbm-module"
initialdir              = $SLCBM_DIR

output                  = ${LOGBASE}.\$(ClusterId).\$(ProcId).out
error                   = ${LOGBASE}.\$(ClusterId).\$(ProcId).err
log                     = ${LOGBASE}.\$(ClusterId).log

request_GPUs            = 1
request_CPUs            = 8
request_memory          = 32G
requirements            = UidDomain == \"cs.uni-saarland.de\"
+WantGPUHomeMounted     = true
queue 1" | condor_submit

done

echo ""
echo "Submitted ${#SEEDS[@]} AwA2 jobs (seeds: ${SEEDS[*]})"
condor_q
