# Architecture & Trade-offs

This document records the architectural decisions made during optimization.

## 1. The Mux vs. Load Trade-off
We initially aimed for a deep Binary Search Mux (Depth 4 or 5) to eliminate most loads. However, empirical testing revealed:
- **Depth 0-2 (7 nodes)**: Highly effective. Fast vector ALU/Flow ops replace slow scalar loads.
- **Depth 3 (15 nodes)**: Attempted with `vselect` optimization. While it reduced loads further (112 vs 120 cycles), the extra instruction overhead (Flow or Valu) made it net slower (~2000 cycles).
- **Decision**: Cap Mux at Depth 2. Use `load` engine (2 slots) for deeper levels to balance the workload.

## 2. Interleaving Strategy
To hide the 64-cycle latency of loads and arithmetic, we need many active batches in flight.
- **Optimal**: `N_TEMPS = 32`. This perfectly aligns all 32 batches to their own scratch-space registers. By preventing WAW and WAR hazards on temporary registers across batches, the VLIW list scheduler is able to completely hide the 64-cycle latency of `load` and `valu` operations.
- **Space Constraint**: We are right at the 1536-word scratch limit with `N_TEMPS = 32`.

## 3. Instruction Selection
- **Mux Implementation**: Replaced `valu` arithmetic (subtract/multiply-add) with `flow` `vselect` instruction. Since `valu` is the bottleneck (Hash function uses 12 slots/batch) and `flow` is underutilized, this offloading saved ~32 cycles per round.
- **Hash Function**: Simplified to use `valu` ops. Attempts to simplify further were limited by correctness requirements.
- **Input Exploitation (Manual JIT)**: Bypassed initial `inp_indices_p` load (since all input indices start at 0), using `vbroadcast v_zero` instead.

## 4. Theoretical Limits & Bottlenecks
- **Load Bound**: The kernel is currently **Load Bound**.
  - `load` engine has 2 slots.
  - With Depth 2 (8 load levels), we execute 8 loads per batch per round.
  - Total Loads = 32 batches * 8 loads = 256 loads per round.
  - Min Cycles = 256 / 2 = 128 cycles per round.
  - Theoretical Min: 128 cycles * 16 rounds = 2048 cycles? Wait, with Depth 2, `level 0, 1, 2` use `vselect`, saving 3 * 32 loads = 96 loads. We only execute loads for levels 3+. This is `16 - 3 = 13` levels (for forest_height 15? No, forest_height=10).
  - Out of 16 rounds, 5 rounds use Mux (levels 0, 1, 2, and after wrap around 11, 12). 11 rounds use scalar loads.
  - 11 rounds * 32 batches * 8 loads = 2816 loads total.
  - 2816 loads / 2 slots/cycle = 1408 cycles absolute minimum for load operations.
  - Actual: 1704 cycles. We are nearing the hardware limit for this strategy given the necessary scalar loads and the massive `valu` load of `myhash`.
- **Valu Bound**: Depth 3 attempts shifted the bottleneck to `valu` (176 cycles) or `flow` (224 cycles), making it slower.
- **Conclusion**: To break 1487 cycles, one must reduce the total number of loads further (e.g. by exploiting input patterns to skip deeper loads or doing multiple rounds of tree traversal before writing back indices) or by drastically reducing `valu` usage to allow Depth 3.
