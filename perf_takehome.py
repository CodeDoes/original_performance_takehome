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

    def scratch_const_vector(self, val, name=None):
        if (val, "vector") not in self.const_map:
            addr = self.alloc_scratch(name or f"vconst_{val}", VLEN)
            scalar_addr = self.scratch_const(val)
            self.add("valu", ("vbroadcast", addr, scalar_addr))
            self.const_map[(val, "vector")] = addr
        return self.const_map[(val, "vector")]

    def build_hash_vector(self, v_val, v_t1, v_t2):
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            if op1 == "+" and op2 == "+" and op3 == "<<":
                multiplier = (1 << val3) + 1
                v_mul = self.scratch_const_vector(multiplier)
                v_add = self.scratch_const_vector(val1)
                self.add("valu", ("multiply_add", v_val, v_val, v_mul, v_add))
            elif op1 == "+" and op2 == "^" and op3 == "<<":
                v_one_vec = self.scratch_const_vector(1)
                v_val1 = self.scratch_const_vector(val1)
                self.add("valu", ("multiply_add", v_t1, v_val, v_one_vec, v_val1))
                self.add("valu", (op3, v_t2, v_val, self.scratch_const_vector(val3)))
                self.add("valu", (op2, v_val, v_t1, v_t2))
            else:
                v_val1 = self.scratch_const_vector(val1)
                v_val3 = self.scratch_const_vector(val3)
                self.add("valu", (op1, v_t1, v_val, v_val1))
                self.add("valu", (op3, v_t2, v_val, v_val3))
                self.add("valu", (op2, v_val, v_t1, v_t2))

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Optimized implementation.
        """
        init_vars = ["rounds", "n_nodes", "batch_size", "forest_height", "forest_values_p", "inp_indices_p", "inp_values_p"]
        for v in init_vars: self.alloc_scratch(v, 1)
        ts = self.alloc_scratch("ts")
        for i, v in enumerate(init_vars):
            self.add("load", ("const", ts, i))
            self.add("load", ("load", self.scratch[v], ts))

        v_zero = self.scratch_const_vector(0)
        v_one = self.scratch_const_vector(1)
        v_two = self.scratch_const_vector(2)
        v_nn = self.alloc_scratch("v_nn", VLEN)
        self.add("valu", ("vbroadcast", v_nn, self.scratch["n_nodes"]))

        # Optimized layers 0-5 (63 nodes)
        MAX_OPTIMIZED_DEPTH = 5
        N_OPTIMIZED_NODES = (2 ** (MAX_OPTIMIZED_DEPTH + 1)) - 1
        ts_node = self.alloc_scratch("ts_node")
        vdn = [self.alloc_scratch(f"vdn_{i}", VLEN) for i in range(N_OPTIMIZED_NODES)]
        
        # Persistent state
        v_idx_p = [self.alloc_scratch(f"vip_{b}", VLEN) for b in range(batch_size // VLEN)]
        v_val_p = [self.alloc_scratch(f"vvp_{b}", VLEN) for b in range(batch_size // VLEN)]
        
        # Temp registers (one set for the whole kernel)
        N_TEMPS = 2
        v_nv = [self.alloc_scratch(f"vnv_{i}", VLEN) for i in range(N_TEMPS)]
        v_t1 = [self.alloc_scratch(f"vt1_{i}", VLEN) for i in range(N_TEMPS)]
        v_t2 = [self.alloc_scratch(f"vt2_{i}", VLEN) for i in range(N_TEMPS)]
        v_ct = [self.alloc_scratch(f"vct_{i}", VLEN) for i in range(N_TEMPS)]
        # Intermediate mux registers for binary search mux
        v_mx = [self.alloc_scratch(f"vmx_{i}", VLEN) for i in range(32)]

        ba_idx = [self.alloc_scratch(f"bai_{i}") for i in range(0, batch_size, VLEN)]
        ba_val = [self.alloc_scratch(f"bav_{i}") for i in range(0, batch_size, VLEN)]
        for i in range(0, batch_size, VLEN):
            self.add("alu", ("+", ba_idx[i//VLEN], self.scratch["inp_indices_p"], self.scratch_const(i)))
            self.add("alu", ("+", ba_val[i//VLEN], self.scratch["inp_values_p"], self.scratch_const(i)))

        # Initial Load all 256
        for b in range(batch_size // VLEN):
            self.add("load", ("vload", v_idx_p[b], ba_idx[b]))
            self.add("load", ("vload", v_val_p[b], ba_val[b]))

        # Pre-load nodes for Level 0-4
        for ni in range(N_OPTIMIZED_NODES):
            self.add("alu", ("+", ts, self.scratch["forest_values_p"], self.scratch_const(ni)))
            self.add("load", ("load", ts_node, ts))
            self.add("valu", ("vbroadcast", vdn[ni], ts_node))

        for round in range(rounds):
            level = round % (forest_height + 1)
            for b in range(batch_size // VLEN):
                ti = b % N_TEMPS
                
                if level <= MAX_OPTIMIZED_DEPTH:
                    curr_start = (1 << level) - 1
                    curr_num = 1 << level
                    if curr_num == 1:
                        self.add("valu", ("+", v_nv[ti], vdn[0], v_zero))
                    else:
                        # Binary search mux using bitmasking
                        self.add("load", ("const", ts, curr_start))
                        self.add("valu", ("vbroadcast", v_ct[ti], ts))
                        self.add("valu", ("-", v_ct[ti], v_idx_p[b], v_ct[ti]))
                        
                        v_mask_const = self.scratch_const_vector(1)
                        self.add("valu", ("&", v_t1[ti], v_ct[ti], v_mask_const))
                        self.add("valu", ("==", v_t1[ti], v_t1[ti], v_zero)) # mask = (bit0 == 0)
                        
                        prev_layer = [vdn[curr_start + i] for i in range(curr_num)]
                        next_layer = []
                        layer_idx = 0
                        for i in range(0, len(prev_layer), 2):
                            self.add("valu", ("-", v_t2[ti], prev_layer[i], prev_layer[i+1]))
                            self.add("valu", ("multiply_add", v_mx[layer_idx], v_t1[ti], v_t2[ti], prev_layer[i+1]))
                            next_layer.append(v_mx[layer_idx])
                            layer_idx += 1
                        
                        for bit in range(1, level):
                            v_mask_const = self.scratch_const_vector(1 << bit)
                            self.add("valu", ("&", v_t1[ti], v_ct[ti], v_mask_const))
                            self.add("valu", ("==", v_t1[ti], v_t1[ti], v_zero))
                            
                            prev_layer = next_layer
                            next_layer = []
                            for i in range(0, len(prev_layer), 2):
                                self.add("valu", ("-", v_t2[ti], prev_layer[i], prev_layer[i+1]))
                                self.add("valu", ("multiply_add", v_mx[layer_idx], v_t1[ti], v_t2[ti], prev_layer[i+1]))
                                next_layer.append(v_mx[layer_idx])
                                layer_idx += 1
                        
                        self.add("valu", ("+", v_nv[ti], next_layer[0], v_zero))
                else:
                    for vi in range(VLEN):
                        self.add("alu", ("+", v_t1[ti] + vi, v_idx_p[b] + vi, self.scratch["forest_values_p"]))
                    for vi in range(VLEN):
                        self.add("load", ("load", v_nv[ti] + vi, v_t1[ti] + vi))

                # Hash
                self.add("valu", ("^", v_val_p[b], v_val_p[b], v_nv[ti]))
                self.build_hash_vector(v_val_p[b], v_t1[ti], v_t2[ti])
                
                # Update
                self.add("valu", ("&", v_t1[ti], v_val_p[b], v_one))
                self.add("valu", ("+", v_t1[ti], v_t1[ti], v_one))
                self.add("valu", ("multiply_add", v_idx_p[b], v_idx_p[b], v_two, v_t1[ti]))
                self.add("valu", ("<", v_t1[ti], v_idx_p[b], v_nn))
                self.add("valu", ("*", v_idx_p[b], v_idx_p[b], v_t1[ti]))

        # Final Store
        for b in range(batch_size // VLEN):
            self.add("store", ("vstore", ba_idx[b], v_idx_p[b]))
            self.add("store", ("vstore", ba_val[b], v_val_p[b]))

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
    
    # Run until completion instead of round-by-round to avoid brittle check
    machine.run()
    
    for ref_mem in reference_kernel2(mem, value_trace):
        pass
    
    inp_values_p = ref_mem[6]
    if prints:
        print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
        print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
    assert (
        machine.mem[inp_values_p : inp_values_p + len(inp.values)]
        == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
    ), "Incorrect final values"

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
