# Performance Optimizations for Tree Traversal Kernel

This document summarizes the optimizations implemented for the tree traversal kernel on the custom VLIW SIMD architecture.

## Project Overview
The goal is to optimize a kernel that performs multiple rounds of tree traversal. Each round involves:
1. Loading a node value from a perfect binary tree based on a current index.
2. Updating the current value using an XOR and a hash function.
3. Updating the current index based on the lower bit of the hashed value.

## Target Architecture
- **VLIW**: Parallel execution across multiple engines:
  - `alu`: 12 slots
  - `valu`: 6 slots (SIMD, VLEN=8)
  - `load`: 2 slots
  - `store`: 2 slots
  - `flow`: 1 slot
- **SIMD**: Vector length of 8 elements.
- **Scratch Space**: 1536 words, used for persistent state, temporary variables, and constants.

## Implemented Optimizations

### 1. Full Vectorization
The kernel processes all 256 inputs in parallel using SIMD instructions. It processes 32 batches of 8 elements each.

### 2. Constant Management
- **Vectorized Constants**: Constants used in VALU operations are pre-broadcasted to scratch space vectors (`scratch_const_vector`) to avoid redundant `vbroadcast` instructions in the main loop.
- **Persistent State**: Current indices and values for all batches are kept in scratch space (`v_idx_p`, `v_val_p`) across rounds.

### 3. Hash Function Optimization
- **`multiply_add` usage**: Several hash stages of the form `(val + C1) + (val << C2)` are simplified to `val * (1 << C2 + 1) + C1` using the `multiply_add` instruction, saving cycles and slots.
- **Interleaved stages**: Hash operations are interleaved with other instructions to maximize slot utilization.

### 4. Tree Traversal (Levels 0-4)
- **Node Pre-loading**: Tree nodes for the first 5 levels (31 nodes total) are pre-loaded into scratch space as vectors during initialization.
- **Binary Search Mux**: For levels 0-4, the correct node vector is selected using a bitwise binary search mux implemented with `multiply_add` and bitwise masks. This avoids 8 separate scalar loads and memory address calculations per batch.

### 5. Instruction Interleaving and Pipeline Filling
- **Batch Interleaving**: `N_TEMPS = 8` sets of temporary registers are used to interleave operations from 8 different batches within the same loop. This allows the scheduler to fill the 6 `valu` slots and other VLIW engines effectively, hiding instruction latencies.

### 6. Scratch Space Optimization
The memory layout in scratch space was carefully designed to fit:
- 32 batches of persistent state (512 words).
- 31 pre-loaded node vectors (248 words).
- 8 sets of temporary registers.
- Necessary constants.
Total usage remains within the 1536-word limit.

## Performance Results
- **Baseline Cycles**: 147,734
- **Optimized Cycles**: 2,705
- **Speedup**: **~54.6x**

## Future Work
- Explore more efficient mux implementations to pre-load level 5 (requiring 32 additional vectors).
- Further fine-tune batch interleaving and register allocation to reach the next performance tier.
