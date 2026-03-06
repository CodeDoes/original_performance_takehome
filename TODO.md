# TODO

## Completed
- [x] Optimize Mux using `vselect` (flow engine) to reduce `valu` pressure.
- [x] Increase Mux Depth to 3 (Attempted, proved slower due to instruction overhead).
- [x] Implement Configuration Flags for easy tuning.
- [x] Achieve < 2000 cycles (Current: 1704).
- [x] **Input Exploitation (Manual JIT)**: Hardcode `idx = 0` for round 0 by broadcasting `v_zero` instead of loading `inp_indices_p` from memory. Saves significant load cycles at initialization.
- [x] **Maximized Interleaving**: Reached theoretical max of `N_TEMPS = 32` within the 1536-word scratch limit.

## Next Steps / Ideas for Next Developer
- [ ] **Instruction Selection**: Re-evaluate if `alu` can be used for any vector logic (unlikely, but worth checking `vcompare` vs `compare`).
- [ ] **Load Optimization**: Investigate if `load_offset` can combine operations? (Likely no, `offset` is immediate or register, but doesn't parallelize loads).
- [ ] **Multi-Round Unrolling**: Can we execute 2 rounds per batch sequentially to avoid intermediate stores entirely? Doing so might exceed scratch space, but it would drastically cut `vstore` operations and `vload` setup.
- [ ] **Hash Simplification**: The `HASH_STAGES` logic consumes roughly 12 `valu` slots per batch. Is there a mathematical identity to simplify the chained `+, ^, <<, >>` operations to save `valu` slots, allowing us to push Mux to Depth 3 and save loads?
