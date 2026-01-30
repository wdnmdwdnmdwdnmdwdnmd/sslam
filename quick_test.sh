#!/bin/bash
# Quick test to verify both versions work correctly
# Author: lyh

VOCAB="Vocabulary/ORBvoc.txt"
SETTINGS="./myvideo/redmi.yaml"
DATASET="./myvideo"

echo "Testing original version..."
./Examples/Monocular/mono_tum $VOCAB $SETTINGS $DATASET 2>&1 | grep -E "mean tracking|median tracking|Using|LocalBA"

echo ""
echo "Testing NEON-optimized version..."
./Examples/Monocular/mono_tum_neon $VOCAB $SETTINGS $DATASET 2>&1 | grep -E "mean tracking|median tracking|Using|LocalBA|FakeLDLT"
