#!/bin/bash
# =============================================================================
# BigCodeBench Sampling Script — No Docker (direct pip install)
# Models: gemma2-2b, gemma2-9b, qwen3-8b
# GPU:    RTX PRO 4090 (RunPod)
# =============================================================================

set -e

# --- Config ------------------------------------------------------------------
HF_TOKEN="${HF_TOKEN:?ERROR: HF_TOKEN is not set. Run: export HF_TOKEN=hf_xxx}"
export HF_TOKEN
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

SPLIT="instruct"
SUBSET="full"
TEMPERATURE="1"
N_SAMPLES="5"
BACKEND="vllm"
BATCH_SIZE="1"
TP="1"

# --- Install dependencies (only runs once) -----------------------------------
install_deps() {
    echo ">>> Installing BigCodeBench and dependencies..."

    pip install bigcodebench --upgrade -q

    # Install the generate entrypoint (not included in the base pip package)
    pip install "git+https://github.com/bigcode-project/bigcodebench.git" --upgrade -q

    # flash-attn speeds up generation significantly — skip if it fails to build
    pip install packaging ninja -q
    pip install flash-attn --no-build-isolation -q || echo "⚠ flash-attn install failed — skipping (generation will still work)"

    # vLLM (needed for --backend vllm)
    pip install vllm -q

    echo ">>> Dependencies installed."
}

# --- Helper ------------------------------------------------------------------
run_model() {
    local model_id="$1"
    local model_tag="$2"
    local extra_args="${3:-}"

    echo ""
    echo "============================================================"
    echo " Sampling: $model_id"
    echo " Samples:  $N_SAMPLES  |  Temp: $TEMPERATURE  |  Split: $SPLIT"
    echo "============================================================"

    python -m bigcodebench.generate \
        --model "$model_id" \
        --split "$SPLIT" \
        --subset "$SUBSET" \
        --temperature "$TEMPERATURE" \
        --n_samples "$N_SAMPLES" \
        --backend "$BACKEND" \
        --resume \
        $extra_args \
        2>&1 | tee "${model_tag}_sampling.log"

    echo "✓ Done: $model_id — results in ./bcb_results/ | log: ${model_tag}_sampling.log"
}

# vLLM v1 engine stability fixes
#export VLLM_WORKER_MULTIPROC_METHOD=spawn   # Fixes engine core init failures on some pods
#export VLLM_USE_V1=0 

# --- Main --------------------------------------------------------------------
install_deps

run_model "google/gemma-2-2b"   "gemma2_2b"  "--max_model_len 8192"
run_model "google/gemma-2-9b"   "gemma2_9b"  "--max_model_len 8192"
run_model "Qwen/Qwen3-8B"       "qwen3_8b"

echo ""
echo "============================================================"
echo " All models sampled. Results in: ./bcb_results/"
echo "============================================================"