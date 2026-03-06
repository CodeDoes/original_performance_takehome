# Architecture & Trade-offs

This document records the architectural decisions made during optimization.

## 1. The Mux vs. Load Trade-off
We initially aimed for a deep Binary Search Mux (Depth 4 or 5) to eliminate most loads. However, empirical testing revealed:
- **Depth 0-2 (7 nodes)**: Highly effective. Fast vector ALU/Flow ops replace slow scalar loads.
- **Depth 3 (15 nodes)**: Attempted with `vselect` optimization. While it reduced loads further (112 vs 120 cycles), the extra instruction overhead (Flow or Valu) made it net slower (~2000 cycles).
- **Decision**: Cap Mux at Depth 2. Use `load` engine (2 slots) for deeper levels to balance the workload.

## 2. Interleaving Strategy
To hide the 64-cycle latency of loads and arithmetic, we need many active batches in flight.
- **Optimal**: `N_TEMPS = 30`. This aligns with `valu` width (6 slots) and fits in scratch space.
- **Space Constraint**: `N_TEMPS = 32` combined with cached constants exceeded the 1536-word scratch limit. Dynamic constant generation was tested to save space for N=32, but the extra `valu` instructions (broadcasts) caused a performance regression.

## 3. Instruction Selection
- **Mux Implementation**: Replaced `valu` arithmetic (subtract/multiply-add) with `flow` `vselect` instruction. Since `valu` is the bottleneck (Hash function uses 12 slots/batch) and `flow` is underutilized, this offloading saved ~32 cycles per round.
- **Hash Function**: Simplified to use `valu` ops. Attempts to simplify further were limited by correctness requirements.

## 4. Scratch Space Tetris
The 1536-word scratch space is the hard limit.
- **Persistent State**: 512 words.
- **Constants**: ~200 words (Hash, Mux, Nodes).
- **Temps**: 30 * 3 vectors = 720 words.
- **Total**: ~1432 words. Fits comfortably.
