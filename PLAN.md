# Optimization Plan: Simplified & High-Performance

The current implementation is approaching the theoretical limits of the architecture, but there is a path to further performance by simplifying the approach and maximizing the VLIW slot utilization.

## 1. The Bottleneck Analysis
- **Load Bottleneck**: Using the `load` engine for 256 elements takes **128 cycles** (2 slots).
- **VALU Opportunity**: The `valu` engine has **6 slots**. If we can perform "selection" (muxing) in fewer than 128 cycles, it's a win.
- **Hash Cost**: The hash function takes ~96 cycles on the `valu` engine for all 32 batches.

## 2. The Plan

### Phase 1: Pure Interleaving (The "Simple" Way)
Instead of complex manual batch management, we will leverage the scheduler:
1.  **32-Batch Interleaving**: We will process all 256 inputs every round.
2.  **Strategic Muxing**: 
    - **Levels 0-3**: Use the `valu` muxing logic. At Depth 3 (8 nodes), it takes ~24 cycles, which is much faster than 128 cycles of loads.
    - **Levels 4-10**: Use standard `load` operations. While the `load` engine is busy for 128 cycles, the `valu` engine can be used to finish the hash and index updates of the *previous* batch or other interleaved work.

### Phase 2: Refined Scratch Management
To allow for 32-batch interleaving without running out of scratch space (1536 words):
1.  **Reuse Temps**: Use only 8 sets of temporary registers. The scheduler is smart enough to pack instructions from 32 batches into these 8 "register windows" if we structure the loop correctly.
2.  **Vectorized Constants**: Keep only the most frequent constants in scratch space.

### Phase 3: Hash Micro-Optimizations
1.  **Full `multiply_add` Coverage**: Apply the `(1 << val3) + 1` multiplier trick to all 3 stages that follow the `+ << +` pattern.
2.  **Pipeline the Hash**: Structure the hash code to allow the scheduler to interleave Stage N of Batch A with Stage M of Batch B.

## 3. Why this isn't "Too Complicated"
- We stop trying to mux the entire tree. Muxing 1024 nodes (Depth 10) is impossible in scratch space.
- We rely on the architecture's strengths: 2 load slots are plenty if we keep them busy, and 6 VALU slots are powerful for the top of the tree.

## 4. Expected Outcome
By focusing on the **LOAD/VALU balance** and maximizing **VLIW density**, we expect to beat the 2,271 cycle mark and move towards the final performance targets.
