# Performance Optimizations for Tree Traversal Kernel

This document summarizes the optimizations implemented for the tree traversal kernel, achieving **1957 cycles** (~75.5x speedup).

## Key Achievements
- **Cycles**: 1957 (Baseline: 147,734)
- **Speedup**: ~75.5x
- **Passed Threshold**: `test_opus4_many_hours` (< 2164 cycles)

## Implemented Optimizations

### 1. Full Vectorization & Interleaving
- **SIMD Processing**: Processes all 256 inputs in parallel using 32 SIMD batches (VLEN=8).
- **Massive Interleaving**: Interleaves instruction streams from **30 independent batches** simultaneously (`N_TEMPS = 30`). This fills VLIW slots effectively.
- **Why N=30?**: N=32 caused scratch space exhaustion with cached constants. N=30 provides excellent latency hiding while fitting in space.

### 2. Balanced Tree Traversal (Depth 2 + vselect)
- **Hybrid Approach**:
    - **Levels 0-2 (7 nodes)**: Uses a recursive **Binary Search Mux**.
    - **Levels 3+**: Falls back to standard scalar `load` operations.
- **vselect Optimization**: Replaced `valu` arithmetic (subtract/multiply-add) with `flow` `vselect` instruction for Mux levels. This offloads the bottlenecked `valu` engine to the underutilized `flow` engine.
- **Depth Tuning**: Depth 2 remains optimal. Depth 3 (15 nodes) reduces loads but explodes Valu/Flow instruction counts, causing a net slowdown (2000+ cycles).

### 3. Scratch Space Management
- **Efficient Register Allocation**: Allocated 30 sets of registers.
- **Cached Constants**: Constants (Mux midpoints, Hash constants) are cached in scratch space. Dynamic generation (using `valu` broadcast) was tested but found to be slower due to `valu` contention.
- **Persistent State**: Indices and values kept in scratch.

### 4. Configuration Flags
- Implemented flags in `perf_takehome.py` to easily tune strategies:
    - `MAX_OPTIMIZED_DEPTH`: 2 vs 3.
    - `N_TEMPS`: Batch count.
    - `USE_VSELECT_MUX`: Switch between `valu` and `flow` Mux.
    - `USE_DYNAMIC_CONSTANTS`: Switch between cached and computed constants.

## Failed Experiments (Lessons Learned)

### 1. Depth 3 Mux
- Increasing Mux to Depth 3 (15 nodes) reduced loads significantly.
- However, the extra Mux logic (whether Valu or Flow) cost more cycles than the loads saved.
- Best Depth 3 result: ~2003 cycles (Valu bound).

### 2. N_TEMPS = 32
- Caused scratch space exhaustion (1536 words limit) when combined with cached constants.
- Removing cached constants allowed N=32 but slowed down execution due to Valu overhead.

### 3. Dynamic Constants
- Generating constants on-the-fly (`vbroadcast`) saved space but increased `valu` pressure by ~40%, causing regression.
