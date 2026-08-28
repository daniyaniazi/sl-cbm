#!/usr/bin/env bash
# Submit SL-CBM per-sample interpretability for CUB.
# For each image in CUB_IDs.txt: finds its top-10 concepts, generates heatmaps.
# Output: ~/sl-cbm/outputs/evals/interpret_cub_topk/<method>/<img_id>/

set -euo pipefail

PYTHON="/home/dani00003/miniconda3/envs/sl-cbm/bin/python"
SLCBM_DIR="/home/dani00003/sl-cbm"
CKPT="$SLCBM_DIR/outputs/slcbm_cub_seed42/trainable_weights.pt"
CONCEPT_BANK="$SLCBM_DIR/concept_banks/cub_resnet18_cream_0.1_100.pkl"
BACKBONE_CKPT="/home/dani00003/GUIDE-cbm-WIP/models/resnet18_finetuned.pth"
SAMPLE_IDS="$SLCBM_DIR/data/CUB_IDs.txt"
EXPNAME="interpret_cub_topk"
LOGBASE="$SLCBM_DIR/logs/${EXPNAME}"

mkdir -p "$SLCBM_DIR/logs"

echo "  → $EXPNAME"

echo "universe                = docker
docker_image            = pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime
executable              = $PYTHON
arguments               = concept_interpretability.py --backbone-name resnet18_cream --backbone-ckpt $BACKBONE_CKPT --concept-bank $CONCEPT_BANK --pcbm-ckpt $CKPT --pcbm-arch spss_pcbm --dataset cub --explain-method layer_grad_cam --concept-pooling max_pooling_class_wise --top-k 10 --exp-name $EXPNAME --save-100-local --sample-ids $SAMPLE_IDS --universal-seed 42
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

echo ""
echo "Submitted 1 job — top-10 concepts per image, 20 images from CUB_IDs.txt"
echo "Results: $SLCBM_DIR/outputs/evals/${EXPNAME}/layer_grad_cam/<img_id>/"
condor_q
