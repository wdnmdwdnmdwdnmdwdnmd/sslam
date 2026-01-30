#!/bin/bash
# Compare mono_tum vs mono_tum_neon performance with multiple runs
# Author: lyh

echo "========================================"
echo "  ORB-SLAM3 Performance Comparison"
echo "  Original vs NEON-Optimized Solver"
echo "  LBA (Local Bundle Adjustment) Time"
echo "  Multiple Runs with Warmup"
echo "========================================"
echo ""

VOCAB="Vocabulary/ORBvoc.txt"
SETTINGS="Examples/Monocular/TUM1.yaml"
DATASET="./dataset/rgbd_dataset_freiburg1_xyz"
NUM_RUNS=$1 # Number of benchmark runs (after warmup)

# Check if executables exist
if [ ! -f "Examples/Monocular/mono_tum" ]; then
    echo "Error: mono_tum not found. Please run ./build.sh first."
    exit 1
fi

if [ ! -f "Examples/Monocular/mono_tum_neon" ]; then
    echo "Error: mono_tum_neon not found. Please run ./build.sh first."
    exit 1
fi

# Function to calculate mean and std
calculate_stats() {
    local values=("$@")
    local sum=0
    local count=${#values[@]}
    
    # Calculate mean
    for val in "${values[@]}"; do
        sum=$(echo "$sum + $val" | bc)
    done
    local mean=$(echo "scale=6; $sum / $count" | bc)
    
    # Calculate standard deviation
    local sq_diff_sum=0
    for val in "${values[@]}"; do
        local diff=$(echo "$val - $mean" | bc)
        local sq_diff=$(echo "$diff * $diff" | bc)
        sq_diff_sum=$(echo "$sq_diff_sum + $sq_diff" | bc)
    done
    local variance=$(echo "scale=6; $sq_diff_sum / $count" | bc)
    local std=$(echo "scale=6; sqrt($variance)" | bc -l)
    
    echo "$mean $std"
}

echo "=== Warmup Phase ==="
echo "Running warmup for Original version..."
./Examples/Monocular/mono_tum $VOCAB $SETTINGS $DATASET > /dev/null 2>&1
echo "Running warmup for NEON version..."
./Examples/Monocular/mono_tum_neon $VOCAB $SETTINGS $DATASET > /dev/null 2>&1
echo "Warmup completed."
echo ""

# Arrays to store results
declare -a ORIGINAL_TIMES
declare -a NEON_TIMES

echo "=== Benchmarking NEON-Optimized Version ==="
for i in $(seq 1 $NUM_RUNS); do
    echo -n "Run $i/$NUM_RUNS: "
    ./Examples/Monocular/mono_tum_neon $VOCAB $SETTINGS $DATASET > /tmp/mono_tum_neon_run${i}.log 2>&1
    
    # Wait for all threads and resources to be fully released
    sleep 2
    pkill -9 -f mono_tum_neon 2>/dev/null  # Force kill any hanging processes
    sleep 1
    
    TIME=$(grep "^LBA:" /tmp/mono_tum_neon_run${i}.log | sed 's/LBA: //' | sed 's/\$.*//')
    NEON_TIMES+=($TIME)
    echo "${TIME}ms"
done
echo ""

echo "=== Benchmarking Original Version ==="
for i in $(seq 1 $NUM_RUNS); do
    echo -n "Run $i/$NUM_RUNS: "
    ./Examples/Monocular/mono_tum $VOCAB $SETTINGS $DATASET > /tmp/mono_tum_original_run${i}.log 2>&1
    
    # Wait for all threads and resources to be fully released
    sleep 2
    pkill -9 -f mono_tum 2>/dev/null  # Force kill any hanging processes
    sleep 1
    
    TIME=$(grep "^LBA:" /tmp/mono_tum_original_run${i}.log | sed 's/LBA: //' | sed 's/\$.*//')
    ORIGINAL_TIMES+=($TIME)
    echo "${TIME}ms"
done
echo ""



# Calculate statistics
ORIGINAL_STATS=($(calculate_stats "${ORIGINAL_TIMES[@]}"))
NEON_STATS=($(calculate_stats "${NEON_TIMES[@]}"))

ORIGINAL_MEAN=${ORIGINAL_STATS[0]}
ORIGINAL_STD=${ORIGINAL_STATS[1]}
NEON_MEAN=${NEON_STATS[0]}
NEON_STD=${NEON_STATS[1]}

echo "========================================"
echo "  LBA Performance Summary ($NUM_RUNS runs)"
echo "========================================"
printf "Original version:     %.6f ± %.6f ms\n" $ORIGINAL_MEAN $ORIGINAL_STD
printf "NEON-optimized:       %.6f ± %.6f ms\n" $NEON_MEAN $NEON_STD
echo ""

if [ ! -z "$ORIGINAL_MEAN" ] && [ ! -z "$NEON_MEAN" ]; then
    SPEEDUP=$(echo "scale=4; $ORIGINAL_MEAN / $NEON_MEAN" | bc)
    IMPROVEMENT=$(echo "scale=2; ($ORIGINAL_MEAN - $NEON_MEAN) / $ORIGINAL_MEAN * 100" | bc)
    printf "Speedup:              %.4fx\n" $SPEEDUP
    printf "Improvement:          %.2f%%\n" $IMPROVEMENT
fi

echo ""
echo "Individual run times:"
echo "Original: ${ORIGINAL_TIMES[@]}"
echo "NEON:     ${NEON_TIMES[@]}"
echo ""
echo "Last run logs saved to:"
echo "  Original: /tmp/mono_tum_original_run${NUM_RUNS}.log"
echo "  NEON:     /tmp/mono_tum_neon_run${NUM_RUNS}.log"
