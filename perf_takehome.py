"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.
"""

from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.raw_slots = [] # New list to store raw (engine, slot) tuples
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def get_reads_writes(self, engine, slot):
        reads = set()
        writes = set()

        if engine == "alu":
            op, dest, a1, a2 = slot
            writes.add(dest)
            reads.add(a1)
            reads.add(a2)
        elif engine == "valu":
            if slot[0] == "vbroadcast":
                dest, src = slot[1], slot[2]
                for i in range(VLEN): writes.add(dest + i)
                reads.add(src)
            elif slot[0] == "multiply_add":
                dest, a, b, c = slot[1], slot[2], slot[3], slot[4]
                for i in range(VLEN):
                    writes.add(dest + i)
                    reads.add(a + i)
                    reads.add(b + i)
                    reads.add(c + i)
            else:
                op, dest, a1, a2 = slot
                for i in range(VLEN):
                    writes.add(dest + i)
                    reads.add(a1 + i)
                    reads.add(a2 + i)
        elif engine == "load":
            if slot[0] == "load":
                dest, addr = slot[1], slot[2]
                writes.add(dest)
                reads.add(addr)
            elif slot[0] == "load_offset":
                dest, addr, offset = slot[1], slot[2], slot[3]
                writes.add(dest + offset)
                reads.add(addr + offset)
            elif slot[0] == "vload":
                dest, addr = slot[1], slot[2]
                for i in range(VLEN): writes.add(dest + i)
                reads.add(addr)
            elif slot[0] == "const":
                dest, val = slot[1], slot[2]
                writes.add(dest)
        elif engine == "store":
            if slot[0] == "store":
                addr, src = slot[1], slot[2]
                reads.add(addr)
                reads.add(src)
            elif slot[0] == "vstore":
                addr, src = slot[1], slot[2]
                reads.add(addr)
                for i in range(VLEN): reads.add(src + i)
        elif engine == "flow":
            if slot[0] == "select":
                dest, cond, a, b = slot[1], slot[2], slot[3], slot[4]
                writes.add(dest)
                reads.add(cond)
                reads.add(a)
                reads.add(b)
            elif slot[0] == "add_imm":
                dest, a, imm = slot[1], slot[2], slot[3]
                writes.add(dest)
                reads.add(a)
            elif slot[0] == "vselect":
                dest, cond, a, b = slot[1], slot[2], slot[3], slot[4]
                for i in range(VLEN):
                    writes.add(dest + i)
                    reads.add(cond + i)
                    reads.add(a + i)
                    reads.add(b + i)
            elif slot[0] in ("cond_jump", "cond_jump_rel"):
                reads.add(slot[1])
            elif slot[0] == "jump_indirect":
                reads.add(slot[1])
            elif slot[0] == "trace_write":
                reads.add(slot[1])
            elif slot[0] == "coreid":
                writes.add(slot[1])
        elif engine == "debug":
            if slot[0] == "compare":
                reads.add(slot[1])
            elif slot[0] == "vcompare":
                loc = slot[1]
                for i in range(VLEN): reads.add(loc + i)

        return reads, writes

    def build(self, raw_slots: list[tuple[Engine, tuple]], vliw: bool = True):
        if not vliw:
            instrs = []
            for engine, slot in raw_slots:
                instrs.append({engine: [slot]})
            return instrs

        # List scheduling algorithm
        packed_instrs = [] # List of defaultdict(list)
        last_write_cycle = {} # addr -> cycle_index
        last_read_cycle = {}  # addr -> cycle_index
        barrier_cycle = -1

        def get_bundle(c):
            while len(packed_instrs) <= c:
                packed_instrs.append(defaultdict(list))
            return packed_instrs[c]

        for engine, slot in raw_slots:
            reads, writes = self.get_reads_writes(engine, slot)
            
            # 1. Earliest cycle based on RAW hazards and barriers
            start_cycle = barrier_cycle + 1
            for r in reads:
                if r in last_write_cycle:
                    start_cycle = max(start_cycle, last_write_cycle[r] + 1)
            
            # 2. Earliest cycle based on WAR hazards (write cannot precede program-order-previous read)
            for w in writes:
                if w in last_read_cycle:
                    start_cycle = max(start_cycle, last_read_cycle[w])

            is_barrier = False
            if engine == "flow" and slot[0] in ("pause", "halt", "cond_jump", "cond_jump_rel", "jump", "jump_indirect"):
                is_barrier = True
            elif engine == "debug":
                is_barrier = True
            
            if is_barrier:
                start_cycle = len(packed_instrs)

            # 3. Find first cycle >= start_cycle with a free slot and no WAW hazards
            c = start_cycle
            while True:
                bundle = get_bundle(c)
                
                # Structural hazard
                if len(bundle[engine]) >= SLOT_LIMITS.get(engine, 1):
                    c += 1
                    continue
                
                # WAW hazard in same bundle (multiple writes to same addr in same cycle)
                waw_hazard = False
                for existing_engine, existing_slots in bundle.items():
                    for existing_slot in existing_slots:
                        _, existing_writes = self.get_reads_writes(existing_engine, existing_slot)
                        if any(w in existing_writes for w in writes):
                            waw_hazard = True
                            break
                    if waw_hazard: break
                
                if waw_hazard:
                    c += 1
                    continue
                
                # If we reach here, we can place the slot in cycle c
                bundle[engine].append(slot)
                for w in writes:
                    last_write_cycle[w] = c
                for r in reads:
                    last_read_cycle[r] = c
                
                if is_barrier:
                    barrier_cycle = c
                
                break

        return [dict(b) for b in packed_instrs]

    def add(self, engine: Engine, slot: tuple):
        # Add raw (engine, slot) tuples to a list for later packing
        self.raw_slots.append((engine, slot))

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))

        return slots

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Optimized vectorized implementation.
        Processes VLEN elements at a time.
        """
        # Scratch space addresses for parameters
        init_vars = [
            "rounds",
            "n_nodes",
            "batch_size",
            "forest_height",
            "forest_values_p",
            "inp_indices_p",
            "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        
        tmp_scalar = self.alloc_scratch("tmp_scalar")
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp_scalar, i))
            self.add("load", ("load", self.scratch[v], tmp_scalar))

        # Constants
        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)

        # Pre-broadcast constants to vectors
        v_zero = self.alloc_scratch("v_zero", VLEN)
        self.add("valu", ("vbroadcast", v_zero, zero_const))
        v_one = self.alloc_scratch("v_one", VLEN)
        self.add("valu", ("vbroadcast", v_one, one_const))
        v_two = self.alloc_scratch("v_two", VLEN)
        self.add("valu", ("vbroadcast", v_two, two_const))

        v_c1_map = {}
        v_c3_map = {}
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            c1 = self.scratch_const(val1)
            c3 = self.scratch_const(val3)
            v_c1 = self.alloc_scratch(f"v_c1_{hi}", VLEN)
            v_c3 = self.alloc_scratch(f"v_c3_{hi}", VLEN)
            self.add("valu", ("vbroadcast", v_c1, c1))
            self.add("valu", ("vbroadcast", v_c3, c3))
            v_c1_map[hi] = v_c1
            v_c3_map[hi] = v_c3

        v_n_nodes = self.alloc_scratch("v_n_nodes", VLEN)
        self.add("valu", ("vbroadcast", v_n_nodes, self.scratch["n_nodes"]))
        
        v_forest_p = self.alloc_scratch("v_forest_p", VLEN)
        self.add("valu", ("vbroadcast", v_forest_p, self.scratch["forest_values_p"]))

        # Pre-calculate batch addresses
        batch_addr_idx_map = {}
        batch_addr_val_map = {}
        for i in range(0, batch_size, VLEN):
            i_const = self.scratch_const(i)
            addr_idx = self.alloc_scratch(f"batch_addr_idx_{i}")
            addr_val = self.alloc_scratch(f"batch_addr_val_{i}")
            self.add("alu", ("+", addr_idx, self.scratch["inp_indices_p"], i_const))
            self.add("alu", ("+", addr_val, self.scratch["inp_values_p"], i_const))
            batch_addr_idx_map[i] = addr_idx
            batch_addr_val_map[i] = addr_val

        # Preamble pause
        self.add("flow", ("pause",))

        # Process all sub-batches together
        N_BATCHES = batch_size // VLEN
        
        v_indices = [self.alloc_scratch(f"v_idx_{b}", VLEN) for b in range(N_BATCHES)]
        v_values = [self.alloc_scratch(f"v_val_{b}", VLEN) for b in range(N_BATCHES)]
        v_node_vals = [self.alloc_scratch(f"v_node_val_{b}", VLEN) for b in range(N_BATCHES)]
        
        # Temporary registers (one set per batch, reuse node_vals for tmp3)
        v_tmp1 = [self.alloc_scratch(f"v_tmp1_{b}", VLEN) for b in range(N_BATCHES)]
        v_tmp2 = [self.alloc_scratch(f"v_tmp2_{b}", VLEN) for b in range(N_BATCHES)]
        v_tmp3 = v_node_vals # Reuse node_vals scratch for tmp3 after it's used
        
        # 0. Initial load from memory
        for b in range(N_BATCHES):
            i = b * VLEN
            self.add("load", ("vload", v_indices[b], batch_addr_idx_map[i]))
            self.add("load", ("vload", v_values[b], batch_addr_val_map[i]))

        for round in range(rounds):
            # 1. Address calculations and Loads for ALL batches
            # Use ALU for address calculation to free up VALU slots
            for b in range(N_BATCHES):
                for vi in range(VLEN):
                    self.add("alu", ("+", v_tmp1[b] + vi, v_indices[b] + vi, self.scratch["forest_values_p"]))
                for vi in range(VLEN):
                    self.add("load", ("load", v_node_vals[b] + vi, v_tmp1[b] + vi))

            # 2. Hash calculation for ALL batches
            for b in range(N_BATCHES):
                self.add("valu", ("^", v_values[b], v_values[b], v_node_vals[b]))

            for hi in range(len(HASH_STAGES)):
                op1, val1, op2, op3, val3 = HASH_STAGES[hi]
                v_c1 = v_c1_map[hi]
                v_c3 = v_c3_map[hi]
                
                # Part 1: First two ops for all batches
                for b in range(N_BATCHES):
                    self.add("valu", (op1, v_tmp1[b], v_values[b], v_c1))
                    self.add("valu", (op3, v_tmp2[b], v_values[b], v_c3))
                
                # Part 2: Third op for all batches
                for b in range(N_BATCHES):
                    self.add("valu", (op2, v_values[b], v_tmp1[b], v_tmp2[b]))

            # 3. Update indices and Wrap for ALL batches
            for b in range(N_BATCHES):
                self.add("valu", ("%", v_tmp1[b], v_values[b], v_two))
                self.add("valu", ("==", v_tmp1[b], v_tmp1[b], v_zero))
                # Now we can reuse v_node_vals[b] as v_tmp3[b]
                self.add("flow", ("vselect", v_tmp3[b], v_tmp1[b], v_one, v_two))
                self.add("valu", ("*", v_indices[b], v_indices[b], v_two))
                self.add("valu", ("+", v_indices[b], v_indices[b], v_tmp3[b]))
                self.add("valu", ("<", v_tmp1[b], v_indices[b], v_n_nodes))
                self.add("flow", ("vselect", v_indices[b], v_tmp1[b], v_indices[b], v_zero))

        # 3. Final store to memory
        for b in range(N_BATCHES):
            i = b * VLEN
            self.add("store", ("vstore", batch_addr_idx_map[i], v_indices[b]))
            self.add("store", ("vstore", batch_addr_val_map[i], v_values[b]))

        self.add("flow", ("pause",))
        self.instrs = self.build(self.raw_slots, vliw=True)
        self.raw_slots = []


BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
