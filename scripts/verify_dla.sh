#!/bin/bash

# UNINA-DLA Verification Script
# Usage: ./verify_dla.sh <onnx_file>

ONNX_FILE=${1:-"unina_dla.onnx"}
ENGINE_FILE="${ONNX_FILE%.*}.engine"

echo "Compiling $ONNX_FILE to $ENGINE_FILE for DLA..."

# Critical Flags for DLA:
# --useDLACore=0: Target DLA Core 0
# --allowGPUFallback: Allow fallback (we want to check if it happens, not crash)
# --inputIOFormats=fp16:chw: Avoids GPU reformatting
# --profilingVerbosity=detailed: Show layer placement

trtexec --onnx=$ONNX_FILE \
        --saveEngine=$ENGINE_FILE \
        --useDLACore=0 \
        --int8 \
        --fp16 \
        --inputIOFormats=fp16:chw \
        --outputIOFormats=fp16:chw \
        --profilingVerbosity=detailed \
        --noDataTransfers \
        --allowGPUFallback > dla_log.txt 2>&1

echo "Compilation finished. Checking for GPU fallbacks..."

# Grep for GPU Fallbacks in the log
grep "GPU" dla_log.txt | grep "Layer"

FALLBACK_COUNT=$(grep "GPU" dla_log.txt | grep "Layer" | wc -l)

if [ $FALLBACK_COUNT -eq 0 ]; then
    echo "SUCCESS: 100% DLA Residency Achieved!"
else
    echo "WARNING: $FALLBACK_COUNT layers fell back to GPU. Check dla_log.txt."
fi

echo "Latency Benchmark:"
tail -n 20 dla_log.txt | grep "mean"
