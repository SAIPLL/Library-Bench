#!/bin/bash

# --- Args ---
# Usage: ./evaluate.sh <model> <id> [split_point] [total]
# Example: ./evaluate.sh gemma2:2b 0 569 1139
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <model> <id> [split_point=569] [total=1139]"
    echo "Example: $0 gemma2:2b 0 569 1139"
    exit 1
fi

MODEL="$1"
ID="$2"
SPLIT_POINT="${3:-569}"
TOTAL="${4:-1139}"

# --- Config ---
SAMPLES_DIR="resultsb-sanitized"
SAMPLES_FILE="${MODEL}-${ID}-no-constrained-1-answers-sanitized"

# --- Generate comma-separated ranges ---
RANGE1=$(seq 1 $SPLIT_POINT | paste -sd ',')
RANGE2=$(seq $((SPLIT_POINT + 1)) $TOTAL | paste -sd ',')

# --- Run evaluations ---
bigcodebench.evaluate \
    --execution e2b \
    --split instruct \
    --subset full \
    --samples "${SAMPLES_DIR}/${SAMPLES_FILE}_part1.jsonl" \
    --selective_evaluate "$RANGE1"

bigcodebench.evaluate \
    --execution e2b \
    --split instruct \
    --subset full \
    --samples "${SAMPLES_DIR}/${SAMPLES_FILE}_part2.jsonl" \
    --selective_evaluate "$RANGE2"