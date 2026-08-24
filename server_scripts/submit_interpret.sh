#!/usr/bin/env bash
# Submit SL-CBM concept interpretability jobs (saliency map visualization).
# Runs concept_interpretability.py for top-5 CUB concept groups × 3 methods.
# Output: ~/sl-cbm/outputs/interpret_<concept>_<method>/
#
# PREREQUISITES: training must be complete — trainable_weights.pt must exist.
#
# USAGE:
#   ./server_scripts/submit_interpret.sh

set -euo pipefail

PYTHON="/home/dani00003/miniconda3/envs/sl-cbm/bin/python"
SLCBM_DIR="/home/dani00003/sl-cbm"
CKPT="$SLCBM_DIR/outputs/slcbm_cub_seed42/trainable_weights.pt"
CONCEPT_BANK="$SLCBM_DIR/concept_banks/cub_resnet18_cream_0.1_100.pkl"
BACKBONE_CKPT="/home/dani00003/GUIDE-cbm-WIP/models/resnet18_finetuned.pth"
SAMPLE_IDS="$SLCBM_DIR/data/CUB_IDs.txt"

mkdir -p "$SLCBM_DIR/logs"

# Top-5 CUB concept groups — update after running notebook cell 6
CONCEPTS=("has_wing_color" "has_bill_shape" "has_breast_color" "has_crown_color" "has_tail_shape")

# GradCAM only (as requested by Felipe)
METHODS=("layer_grad_cam")

for CONCEPT in "${CONCEPTS[@]}"; do
for METHOD in "${METHODS[@]}"; do

    EXPNAME="interpret_${CONCEPT}_${METHOD}"
    LOGBASE="$SLCBM_DIR/logs/${EXPNAME}"

    echo "  → $EXPNAME"

    echo "universe                = docker
docker_image            = pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime
executable              = $PYTHON
arguments               = concept_interpretability.py --backbone-name resnet18_cream --backbone-ckpt $BACKBONE_CKPT --concept-bank $CONCEPT_BANK --pcbm-ckpt $CKPT --pcbm-arch spss_pcbm --dataset cub --concept-target $CONCEPT --explain-method $METHOD --concept-pooling max_pooling_class_wise --exp-name $EXPNAME --save-100-local --sample-ids $SAMPLE_IDS --universal-seed 42
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
echo "Submitted $((${#CONCEPTS[@]} * ${#METHODS[@]})) jobs (5 concepts × GradCAM, 20 images from CUB_IDs.txt)"
echo "Results: $SLCBM_DIR/outputs/evals/interpret_<concept>_layer_grad_cam/"
condor_q
