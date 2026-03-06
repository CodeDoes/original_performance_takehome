---
name: optimize-vliw-kernel
description: "Iteratively optimize a kernel for the custom VLIW architecture. Use when: you need to tune N_TEMPS, MAX_OPTIMIZED_DEPTH, or instruction scheduling to reduce cycles."
---

# VLIW Kernel Optimization

You are tasked with optimizing a kernel for a custom VLIW SIMD architecture. Follow this iterative process:

## 1. Analysis
- Read `perf_takehome.py` and `problem.py` to understand the current constraints (e.g., `SCRATCH_SIZE = 1536`, `VLEN = 8`, engine slot limits).
- Identify the current performance bottleneck (e.g., `load` slots vs `valu` slots).

## 2. Parameter Tuning
Adjust these key variables:
- `N_TEMPS`: Number of interleaved batches (impacts pipeline filling and scratch usage).
- `MAX_OPTIMIZED_DEPTH`: Depth of tree nodes pre-loaded into scratch space.
- `Hybrid Traversal`: Decide which levels use `valu` muxing vs `load` engine.

## 3. Implementation
- Update `KernelBuilder.build_kernel` in `perf_takehome.py`.
- Use `multiply_add` to combine addition and multiplication where possible.
- Ensure all constants are pre-broadcasted to scratch vectors if used repeatedly.

## 4. Validation
- Run `python test.py` in the terminal.
- Capture the `CYCLES` output and speedup.
- If an `AssertionError: Out of scratch space` occurs, reduce `N_TEMPS` or `MAX_OPTIMIZED_DEPTH`.

## 5. Iteration
- If performance improved, commit the changes with the cycle count in the message.
- If performance regressed, analyze the scheduler's behavior and revert or adjust.
