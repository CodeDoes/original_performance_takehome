# TODO

## Completed
- [x] Optimize Mux using `vselect` (flow engine) to reduce `valu` pressure.
- [x] Increase Mux Depth to 3 (Attempted, proved slower due to instruction overhead).
- [x] Implement Configuration Flags for easy tuning.
- [x] Achieve < 2000 cycles (Current: 1957).

## Next Steps / Ideas for Next Developer
- [ ] **Input Exploitation**: The indices for Round 0, 1, 2 are predictable (0, 1/2, 3..6). Implement special handling for early rounds to skip Mux/Load logic entirely. This could save ~300 cycles.
- [ ] **Instruction Selection**: Re-evaluate if `alu` can be used for any vector logic (unlikely, but worth checking `vcompare` vs `compare`).
- [ ] **Load Optimization**: Investigate if `load_offset` can combine operations? (Likely no, `offset` is immediate or register, but doesn't parallelize loads).
