#!/bin/bash
# =============================================================================
# BigCodeBench Sampling Script
# Models: gemma2-2b, gemma2-9b, qwen3-8b
# GPU:    RTX PRO 4090 (RunPod)
# =============================================================================
# PRE-REQUISITES (run these manually before executing this script):
#
# 1. HUGGING FACE PERMISSIONS:
#    - gemma2-2b → request access at: https://huggingface.co/google/gemma-2-2b-it
#    - gemma2-9b → request access at: https://huggingface.co/google/gemma-2-9b-it
#    - qwen3-8b  → publicly available, no approval needed
#    - Then set your HF token:
#        export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
#
# 2. DOCKER (usually pre-installed on RunPod GPU pods):
#    Check with: docker --version
#    If missing:  apt-get update && apt-get install -y docker.io
#
# 3. NVIDIA CONTAINER TOOLKIT (needed for --gpus flag):
#    Check with: nvidia-smi
#    If missing:
#        distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
#        curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | apt-key add -
#        curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list \
#            | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
#        apt-get update && apt-get install -y nvidia-container-toolkit
#        systemctl restart docker
#
# 4. PULL THE DOCKER IMAGE (do this once before the run to avoid mid-job delays):
#    docker pull bigcodebench/bigcodebench-generate:latest
#
# NOTE: You do NOT need to clone the BigCodeBench GitHub repo — the Docker
#       image bundles everything needed. Just run this script from any directory.
# =============================================================================

set -e  # Exit immediately on any error

# --- Config ------------------------------------------------------------------
HF_TOKEN="${HF_TOKEN:?ERROR: HF_TOKEN is not set. Run: export HF_TOKEN=hf_xxx}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"   # Default to GPU 0
OUTPUT_DIR="$(pwd)"  # BigCodeBench writes to ./bcb_results/ by default
SPLIT="instruct"
SUBSET="full"
TEMPERATURE="1"
N_SAMPLES="5"
BACKEND="vllm"
BATCH_SIZE="1"       # Conservative default; increase if VRAM allows
TP="1"               # Tensor parallel size; 1 is correct for a single GPU


# --- Helper ------------------------------------------------------------------
run_model() {
    local model_id="$1"       # HuggingFace model ID
    local model_tag="$2"      # Short name used for log file

    echo ""
    echo "============================================================"
    echo " Sampling: $model_id"
    echo " Samples:  $N_SAMPLES  |  Temp: $TEMPERATURE  |  Split: $SPLIT"
    echo "============================================================"

    docker run \
        --gpus "\"device=$CUDA_VISIBLE_DEVICES\"" \
        -e HF_TOKEN="$HF_TOKEN" \
        -e HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" \
        -v "$OUTPUT_DIR":/app \
        -t bigcodebench/bigcodebench-generate:latest \
        --model "$model_id" \
        --split "$SPLIT" \
        --subset "$SUBSET" \
        --temperature "$TEMPERATURE" \
        --n_samples "$N_SAMPLES" \
        --bs "$BATCH_SIZE" \
        --backend "$BACKEND" \
        --tp "$TP" \
        --resume \
        2>&1 | tee "${model_tag}_sampling.log"

    echo "✓ Done: $model_id — results in ./bcb_results/ | log: ${model_tag}_sampling.log"
}

# --- Run models --------------------------------------------------------------
run_model "google/gemma-2-2b-it"   "gemma2_2b"
run_model "google/gemma-2-9b-it"   "gemma2_9b"
run_model "Qwen/Qwen3-8B-Instruct" "qwen3_8b"

echo ""
echo "============================================================"
echo " All models sampled. Results in: ./bcb_results/"
echo "============================================================"
