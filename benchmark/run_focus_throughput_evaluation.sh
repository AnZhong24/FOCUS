#!/bin/bash

# Script to run throughput benchmark with different batch sizes, datasets, and models
# This script executes the profile_throughput.py benchmark for multiple configurations:
# - Dataset: Specified as command line argument
# - 2 experiment types: Focus (threshold 0.8) and Base (threshold 0.9)
# - 4 batch sizes: 32, 64, 128, and 256
# - 2 models: SDAR-8B-Chat-b32 and LLaDA2.0-mini
# Total per dataset: 2 experiment types × 4 batch sizes × 2 models = 16 benchmarks
# Output is saved to ./results directory
#
# Usage: ./run_focus_throughput_evaluation.sh <dataset_id>
# Example: ./run_focus_throughput_evaluation.sh anon8231489123/ShareGPT_Vicuna_unfiltered
#          ./run_focus_throughput_evaluation.sh allenai/WildChat

# Check if dataset argument is provided
if [ -z "$1" ]; then
    echo "Error: Dataset ID is required as a command line argument"
    echo "Usage: $0 <dataset_id>"
    echo "Example: $0 anon8231489123/ShareGPT_Vicuna_unfiltered"
    exit 1
fi

# Get dataset from command line argument
DATASET=$1

# Create results directory if it doesn't exist
mkdir -p ./results

# Array of models to test
MODELS=("JetLM/SDAR-8B-Chat-b32" "inclusionAI/LLaDA2.0-mini")

# Array of batch sizes to test
BATCH_SIZES=(32 64 128 256)

# ============================================
# Common parameters for all experiments
# ============================================
# --eager-mode: Use eager execution mode
# --backend pytorch: Use PyTorch as the backend
# --skip-tokenize: Skip tokenization step
# --skip-detokenize: Skip detokenization step
# --num-prompts 5000: Number of prompts to process
# --use-uvloop: Use uvloop for async operations
# --dllm-block-length 32: DLLM block length parameter
# --dllm-denoising-steps 32: DLLM denoising steps
# --max-new-tokens 2048: Maximum number of new tokens to generate
# --repeat-block-detect: Enable repeat block detection
# --repeat-block-window 64: Window size for repeat block detection

COMMON_PARAMS="--eager-mode \
    --backend pytorch \
    --skip-tokenize \
    --skip-detokenize \
    --num-prompts 5000 \
    --use-uvloop \
    --dllm-block-length 32 \
    --dllm-denoising-steps 32 \
    --max-new-tokens 2048 \
    --repeat-block-detect \
    --repeat-block-window 64"

# ============================================
# Loop through each model
# ============================================
for MODEL in "${MODELS[@]}"
do
    # Extract model name from path for output file naming
    MODEL_NAME=$(basename ${MODEL})
    
    echo "========================================="
    echo "Processing model: ${MODEL}"
    echo "========================================="
    echo ""
    
    # Extract dataset name from path for output file naming
    DATASET_NAME=$(basename ${DATASET})
    
    echo "========================================="
    echo "Processing dataset: ${DATASET} with model: ${MODEL_NAME}"
    echo "========================================="
    echo ""
        
        # ============================================
        # Run Focus Experiment for this dataset and model
        # ============================================
        echo "========================================="
        echo "Starting FOCUS EXPERIMENT for ${DATASET_NAME} - ${MODEL_NAME}"
        echo "========================================="
        
        # Focus experiment specific parameters
        # --dllm-confidence-threshold 0.8: DLLM confidence threshold
        # --dllm-enable-delayed-cache: Enable DLLM delayed cache
        # --dllm-enable-focus: Enable DLLM focus mechanism
        # --dllm-focus-alpha 1.5: DLLM focus alpha parameter
        
        FOCUS_PARAMS="${COMMON_PARAMS} \
            --dllm-confidence-threshold 0.8 \
            --dllm-enable-delayed-cache \
            --dllm-enable-focus \
            --dllm-focus-alpha 1.5"
        
        for BATCH_SIZE in "${BATCH_SIZES[@]}"
        do
            echo "========================================="
            echo "Running FOCUS benchmark: ${DATASET_NAME}, ${MODEL_NAME}, batch size: ${BATCH_SIZE}"
            echo "========================================="
            
            # Define output file paths for stdout and stderr
            OUTPUT_FILE="./results/focus_${DATASET_NAME}_${MODEL_NAME}_batch_${BATCH_SIZE}.log"
            ERROR_FILE="./results/focus_${DATASET_NAME}_${MODEL_NAME}_batch_${BATCH_SIZE}.err"
            
            # Execute the command with the current batch size
            # --concurrency parameter controls the batch size
            # Redirect stdout to OUTPUT_FILE and stderr to ERROR_FILE
            python benchmark/profile_throughput.py ${FOCUS_PARAMS} --concurrency ${BATCH_SIZE} ${DATASET} ${MODEL} > ${OUTPUT_FILE} 2> ${ERROR_FILE}
            
            # Check if the command executed successfully
            if [ $? -eq 0 ]; then
                echo "Benchmark completed successfully"
                echo "Output saved to: ${OUTPUT_FILE}"
            else
                echo "Benchmark failed"
                echo "Check error log: ${ERROR_FILE}"
            fi
            
            echo ""
        done
        
        # ============================================
        # Run Base Experiment for this dataset and model
        # ============================================
        echo "========================================="
        echo "Starting BASE EXPERIMENT for ${DATASET_NAME} - ${MODEL_NAME}"
        echo "========================================="
        
        # Base experiment specific parameters
        # --dllm-confidence-threshold 0.9: Higher confidence threshold for base experiment
        # Note: delayed cache and focus are disabled by default (no flags)
        
        BASE_PARAMS="${COMMON_PARAMS} \
            --dllm-confidence-threshold 0.9"
        
        for BATCH_SIZE in "${BATCH_SIZES[@]}"
        do
            echo "========================================="
            echo "Running BASE benchmark: ${DATASET_NAME}, ${MODEL_NAME}, batch size: ${BATCH_SIZE}"
            echo "========================================="
            
            # Define output file paths for stdout and stderr
            OUTPUT_FILE="./results/base_${DATASET_NAME}_${MODEL_NAME}_batch_${BATCH_SIZE}.log"
            ERROR_FILE="./results/base_${DATASET_NAME}_${MODEL_NAME}_batch_${BATCH_SIZE}.err"
            
            # Execute the command with the current batch size
            # --concurrency parameter controls the batch size
            # Redirect stdout to OUTPUT_FILE and stderr to ERROR_FILE
            python benchmark/profile_throughput.py ${BASE_PARAMS} --concurrency ${BATCH_SIZE} ${DATASET} ${MODEL} > ${OUTPUT_FILE} 2> ${ERROR_FILE}
            
            # Check if the command executed successfully
            if [ $? -eq 0 ]; then
                echo "Benchmark completed successfully"
                echo "Output saved to: ${OUTPUT_FILE}"
            else
                echo "Benchmark failed"
                echo "Check error log: ${ERROR_FILE}"
            fi
            
            echo ""
        done
        
    echo "========================================="
    echo "Completed all benchmarks for ${DATASET_NAME} - ${MODEL_NAME}"
    echo "========================================="
    echo ""
done

echo "========================================="
echo "Completed all benchmarks for model: ${MODEL_NAME}"
echo "========================================="
echo ""

echo "========================================="
echo "All batch size comparisons completed!"
echo "========================================="
echo ""
echo "Results summary:"
echo "  Dataset: ${DATASET}"
echo "  Total benchmarks run: 16 (2 experiment types × 4 batch sizes × 2 models)"
echo "  Focus experiments: ./results/focus_${DATASET_NAME}_*_batch_*.log"
echo "  Base experiments: ./results/base_${DATASET_NAME}_*_batch_*.log"
echo "========================================="
