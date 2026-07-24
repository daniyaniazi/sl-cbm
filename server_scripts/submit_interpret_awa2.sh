#!/usr/bin/env bash
# Submit SL-CBM concept interpretability jobs for AwA2.
# Runs concept_interpretability.py for top-5 AwA2 concepts × 3 methods.
# Output: ~/sl-cbm/outputs/interpret_awa2_<concept>_<method>/
#
# PREREQUISITES:
#   - AwA2 training complete (trainable_weights.pt must exist)
#   - Update CKPT below with the actual output folder timestamp
#
# USAGE:
#   ./server_scripts/submit_interpret_awa2.sh

set -euo pipefail

PYTHON="/home/dani00003/miniconda3/envs/sl-cbm/bin/python"
SLCBM_DIR="/home/dani00003/sl-cbm"

# ── UPDATE THIS after training finishes ──────────────────────────────────────
CKPT="$SLCBM_DIR/outputs/20260723180418/trainable_weights.pt"
# ─────────────────────────────────────────────────────────────────────────────

CONCEPT_BANK="$SLCBM_DIR/concept_banks/awa2_resnet101_imagenet_0.1_236.pkl"
BACKBONE_CKPT="none"

mkdir -p "$SLCBM_DIR/logs"

# Top-5 AwA2 concepts — UPDATE after running notebook cell to get real top-5
# from model weights (same procedure as CUB)
CONCEPTS=("black" "furry" "quadrapedal" "tail" "fast")

METHODS=("layer_grad_cam" "integrated_gradient" "saliency_map")

for CONCEPT in "${CONCEPTS[@]}"; do
for METHOD in "${METHODS[@]}"; do

    EXPNAME="interpret_awa2_${CONCEPT}_${METHOD}"
    LOGBASE="$SLCBM_DIR/logs/${EXPNAME}"

    echo "  → $EXPNAME"

    echo "universe                = docker
docker_image            = pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime
executable              = $PYTHON
arguments               = concept_interpretability.py --backbone-name resnet101_imagenet --backbone-ckpt $BACKBONE_CKPT --concept-bank $CONCEPT_BANK --pcbm-ckpt $CKPT --pcbm-arch spss_pcbm --dataset awa2 --concept-target $CONCEPT --explain-method $METHOD --concept-pooling max_pooling_class_wise --exp-name $EXPNAME --save-100-local --universal-seed 24
environment             = \"PYTHONPATH=$SLCBM_DIR:/home/dani00003/pcbm-module\"
initialdir              = $SLCBM_DIR

output                  = ${LOGBASE}.\$(ClusterId).\$(ProcId).out
error                   = ${LOGBASE}.\$(ClusterId).\$(ProcId).err
log                     = ${LOGBASE}.\$(ClusterId).log

request_GPUs            = 1
request_CPUs            = 4
request_memory          = 16G
requirements            = UidDomain == \"cs.uni-saarland.de\"
+WantGPUHomeMounted     = true
queue 1" | condor_submit

done
done

echo ""
echo "Submitted $((${#CONCEPTS[@]} * ${#METHODS[@]})) jobs (5 concepts × 3 methods)"
echo "Results: $SLCBM_DIR/outputs/interpret_awa2_<concept>_<method>/"
condor_q
