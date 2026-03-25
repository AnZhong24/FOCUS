#!/bin/bash

# Copyright (c) 2026 SANDS Lab. All rights reserved.

# Run Fast-dLLM-v2-style sub-block cache reuse throughput benchmarks.
# Usage:
#   ./run_sub_block_cache_reuse_benchmark.sh <dataset_id> <model_id> [sub_block_size] [dllm_block_length]
# Examples:
#   ./run_sub_block_cache_reuse_benchmark.sh anon8231489123/ShareGPT_Vicuna_unfiltered JetLM/SDAR-8B-Chat-b32
#   ./run_sub_block_cache_reuse_benchmark.sh anon8231489123/ShareGPT_Vicuna_unfiltered inclusionAI/LLaDA2.0-mini 8 32

if [ -z "$1" ]; then
    echo "Error: Dataset ID is required as the first argument."
    echo "Usage: $0 <dataset_id> <model_id> [sub_block_size] [dllm_block_length]"
    exit 1
fi

if [ -z "$2" ]; then
    echo "Error: Model ID is required as the second argument."
    echo "Usage: $0 <dataset_id> <model_id> [sub_block_size] [dllm_block_length]"
    exit 1
fi

DATASET=$1
MODEL=$2
SUB_BLOCK_SIZE=${3:-8}
BLOCK_LENGTH=${4:-32}
CACHE_BLOCK_SEQ_LEN=${BLOCK_LENGTH}

DATASET_ARGS=()
if [[ "${DATASET}" == *"hendrycks-MATH"* ]]; then
    DATASET_ARGS+=(--dataset-format math)
fi

mkdir -p ./results

BATCH_SIZES=(32 64 128 256)

COMMON_PARAMS="--eager-mode \
    --backend pytorch \
    --skip-tokenize \
    --skip-detokenize \
    --num-prompts 5000 \
    --use-uvloop \
    --dllm-block-length ${BLOCK_LENGTH} \
    --dllm-denoising-steps ${BLOCK_LENGTH} \
    --cache-block-seq-len ${CACHE_BLOCK_SEQ_LEN} \
    --max-new-tokens 2048 \
    --repeat-block-detect \
    --repeat-block-window ${BLOCK_LENGTH}"

SUB_BLOCK_PARAMS="${COMMON_PARAMS} \
    --dllm-confidence-threshold 0.9 \
    --dllm-enable-delayed-cache \
    --dllm-enable-sub-block-cache-reuse \
    --dllm-sub-block-size ${SUB_BLOCK_SIZE}"

MODEL_NAME=$(basename "${MODEL}")
DATASET_NAME=$(basename "${DATASET}")

for BATCH_SIZE in "${BATCH_SIZES[@]}"
do
    OUTPUT_FILE="./results/sub_block_reuse_${DATASET_NAME}_${MODEL_NAME}_sb_${SUB_BLOCK_SIZE}_batch_${BATCH_SIZE}.log"
    ERROR_FILE="./results/sub_block_reuse_${DATASET_NAME}_${MODEL_NAME}_sb_${SUB_BLOCK_SIZE}_batch_${BATCH_SIZE}.err"

    echo "Running sub-block reuse benchmark: dataset=${DATASET_NAME}, model=${MODEL_NAME}, sub_block=${SUB_BLOCK_SIZE}, batch=${BATCH_SIZE}"
    python benchmark/profile_throughput.py ${SUB_BLOCK_PARAMS} "${DATASET_ARGS[@]}" \
        --concurrency ${BATCH_SIZE} ${DATASET} ${MODEL} > "${OUTPUT_FILE}" 2> "${ERROR_FILE}"

    if [ $? -eq 0 ]; then
        echo "Benchmark completed successfully: ${OUTPUT_FILE}"
    else
        echo "Benchmark failed: ${ERROR_FILE}"
    fi
done
