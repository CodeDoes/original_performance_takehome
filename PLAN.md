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

## 5. Theoretical Limits & Bottlenecks
- **Load Bound**: The kernel is currently **Load Bound**.
  - `load` engine has 2 slots.
  - With Depth 2 (8 load levels), we execute 8 loads per batch per round.
  - Total Loads = 32 batches * 8 loads = 256 loads per round.
  - Min Cycles = 256 / 2 = 128 cycles per round?
  - Wait, N=30. 240 loads. 120 cycles.
  - Theoretical Min: 120 cycles * 16 rounds = 1920 cycles.
  - Actual: 1957 cycles. We are within 2% of the hardware limit for this strategy.
- **Valu Bound**: Depth 3 attempts shifted the bottleneck to `valu` (176 cycles) or `flow` (224 cycles), making it slower.
- **Conclusion**: To break 1900 cycles, one must reduce the total number of loads without incurring equal cost in Mux logic. Depth 3 failed this trade-off. Future efforts should focus on alternative Load reduction or bypassing loads (e.g., input pattern exploitation).
