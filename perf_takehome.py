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

# Configuration Flags
MAX_OPTIMIZED_DEPTH = 2      # Depth 2 (3 levels Mux) is optimal due to Mux vs Load trade-off
N_TEMPS = 30                 # Maximize batches to hide latency (fits in space with Depth 2)
USE_VSELECT_MUX = True       # Use flow engine for Mux to offload Valu
USE_DYNAMIC_CONSTANTS = False # Cached constants are faster (Valu bound)

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
                if USE_DYNAMIC_CONSTANTS:
                    self.add("valu", ("vbroadcast", v_t1, self.scratch_const(multiplier)))
                    self.add("valu", ("vbroadcast", v_t2, self.scratch_const(val1)))
                    v_mul, v_add = v_t1, v_t2
                else:
                    v_mul = self.scratch_const_vector(multiplier)
                    v_add = self.scratch_const_vector(val1)
                
                self.add("valu", ("multiply_add", v_val, v_val, v_mul, v_add))
            elif hi == 1:
                # Offload Stage 1 entirely to ALU (8 scalar ops * 3)
                s_val1 = self.scratch_const(val1)
                s_val3 = self.scratch_const(val3)
                for i in range(VLEN):
                    self.add("alu", (op1, v_t1 + i, v_val + i, s_val1))
                    self.add("alu", (op3, v_t2 + i, v_val + i, s_val3))
                    self.add("alu", (op2, v_val + i, v_t1 + i, v_t2 + i))
            elif hi == 3:
                # Offload 1/3 of Stage 3 to ALU
                s_val1 = self.scratch_const(val1)
                for i in range(VLEN):
                     self.add("alu", (op1, v_t1 + i, v_val + i, s_val1))
                
                if USE_DYNAMIC_CONSTANTS:
                     self.add("valu", ("vbroadcast", v_t2, self.scratch_const(val3)))
                     v_val3 = v_t2
                else:
                     v_val3 = self.scratch_const_vector(val3)

                self.add("valu", (op3, v_t2, v_val, v_val3))
                self.add("valu", (op2, v_val, v_t1, v_t2))
            else:
                if USE_DYNAMIC_CONSTANTS:
                    self.add("valu", ("vbroadcast", v_t1, self.scratch_const(val1)))
                    self.add("valu", ("vbroadcast", v_t2, self.scratch_const(val3)))
                    v_val1, v_val3 = v_t1, v_t2
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
        
        if not USE_DYNAMIC_CONSTANTS:
            v_nn = self.alloc_scratch("v_nn", VLEN)
            self.add("valu", ("vbroadcast", v_nn, self.scratch["n_nodes"]))
            v_forest_p = self.alloc_scratch("v_forest_p", VLEN)
            self.add("valu", ("vbroadcast", v_forest_p, self.scratch["forest_values_p"]))

        # Optimized layers
        N_OPTIMIZED_NODES = (2 ** (MAX_OPTIMIZED_DEPTH + 1)) - 1
        ts_node = self.alloc_scratch("ts_node")
        vdn = [self.alloc_scratch(f"vdn_{i}", VLEN) for i in range(N_OPTIMIZED_NODES)]
        
        # Persistent state
        v_idx_p = [self.alloc_scratch(f"vip_{b}", VLEN) for b in range(batch_size // VLEN)]
        v_val_p = [self.alloc_scratch(f"vvp_{b}", VLEN) for b in range(batch_size // VLEN)]
        
        # Temp registers
        # v_regs[ti][0] will be the result/accumulator (v_nv)
        v_regs = [[self.alloc_scratch(f"vr_{i}_{j}", VLEN) for j in range(MAX_OPTIMIZED_DEPTH + 1)] for i in range(N_TEMPS)]

        # Initial Load
        ts_addr = self.alloc_scratch("ts_addr")
        for b in range(batch_size // VLEN):
            self.add("alu", ("+", ts_addr, self.scratch["inp_indices_p"], self.scratch_const(b * VLEN)))
            self.add("load", ("vload", v_idx_p[b], ts_addr))
            self.add("alu", ("+", ts_addr, self.scratch["inp_values_p"], self.scratch_const(b * VLEN)))
            self.add("load", ("vload", v_val_p[b], ts_addr))

        # Pre-load nodes
        for ni in range(N_OPTIMIZED_NODES):
            self.add("alu", ("+", ts, self.scratch["forest_values_p"], self.scratch_const(ni)))
            self.add("load", ("load", ts_node, ts))
            self.add("valu", ("vbroadcast", vdn[ni], ts_node))

        def build_mux(start_idx, count, regs_indices):
            """
            Recursive binary mux.
            """
            if count == 1:
                return ("vdn", start_idx)
            
            mid = start_idx + count // 2
            
            left_res = build_mux(start_idx, count // 2, regs_indices)
            
            remaining_regs = list(regs_indices)
            if isinstance(left_res, int):
                remaining_regs.remove(left_res)
            
            right_res = build_mux(mid, count // 2, remaining_regs)
            
            # Allocate result register
            if isinstance(left_res, int):
                res_reg = v_regs[ti][left_res]
                res_idx = left_res
            else:
                res_idx = remaining_regs[0]
                res_reg = v_regs[ti][res_idx]
            
            # Find a free register for mask
            used_regs = set()
            if isinstance(left_res, int): used_regs.add(left_res)
            if isinstance(right_res, int): used_regs.add(right_res)
            
            mask_idx = -1
            for r in regs_indices:
                if r not in used_regs:
                    mask_idx = r
                    break
            
            if mask_idx == -1: raise Exception("No register for mask")
            mask_reg = v_regs[ti][mask_idx]
            
            # Resolve operands
            left_op = vdn[left_res[1]] if isinstance(left_res, tuple) else v_regs[ti][left_res]
            right_op = vdn[right_res[1]] if isinstance(right_res, tuple) else v_regs[ti][right_res]
            
            # Mux logic: res = (idx < mid) ? left : right
            if USE_DYNAMIC_CONSTANTS:
                self.add("valu", ("vbroadcast", mask_reg, self.scratch_const(mid)))
                self.add("valu", ("<", mask_reg, v_idx_p[b], mask_reg))
            else:
                self.add("valu", ("<", mask_reg, v_idx_p[b], self.scratch_const_vector(mid)))
            
            if USE_VSELECT_MUX and count == 2:
                self.add("flow", ("vselect", res_reg, mask_reg, left_op, right_op))
            else:
                self.add("valu", ("-", res_reg, left_op, right_op))
                self.add("valu", ("multiply_add", res_reg, mask_reg, res_reg, right_op))
            
            return res_idx

        for round in range(rounds):
            level = round % (forest_height + 1)
            is_last_level = (level == forest_height)
            
            for b in range(batch_size // VLEN):
                ti = b % N_TEMPS
                
                # Node Value Selection
                if level == 0 and MAX_OPTIMIZED_DEPTH >= 0:
                    # Level 0: Index is always 0. Use vdn[0] directly.
                    v_node_val = vdn[0]
                    
                elif level == 1 and MAX_OPTIMIZED_DEPTH >= 1:
                    # Level 1: Index is 1 or 2. Select vdn[1] or vdn[2].
                    mask_reg = v_regs[ti][2]
                    v_node_val = v_regs[ti][0]
                    
                    if USE_DYNAMIC_CONSTANTS:
                        self.add("valu", ("vbroadcast", mask_reg, self.scratch_const(2)))
                        self.add("valu", ("<", mask_reg, v_idx_p[b], mask_reg))
                    else:
                        self.add("valu", ("<", mask_reg, v_idx_p[b], v_two))
                    
                    if USE_VSELECT_MUX:
                        self.add("flow", ("vselect", v_node_val, mask_reg, vdn[1], vdn[2]))
                    else:
                        self.add("valu", ("-", v_node_val, vdn[1], vdn[2]))
                        self.add("valu", ("multiply_add", v_node_val, mask_reg, v_node_val, vdn[2]))
                        
                elif level <= MAX_OPTIMIZED_DEPTH:
                    # Generic Mux for other levels
                    curr_start = (1 << level) - 1
                    curr_num = 1 << level
                    res = build_mux(curr_start, curr_num, list(range(MAX_OPTIMIZED_DEPTH + 1)))
                    if isinstance(res, tuple):
                         self.add("valu", ("+", v_regs[ti][0], vdn[res[1]], v_zero))
                         v_node_val = v_regs[ti][0]
                    else:
                         v_node_val = v_regs[ti][res]
                else:
                    # Fallback to scalar loads
                    v_node_val = v_regs[ti][0]
                    v_addr = v_regs[ti][1]
                    if USE_DYNAMIC_CONSTANTS:
                        self.add("valu", ("vbroadcast", v_addr, self.scratch["forest_values_p"]))
                        self.add("valu", ("+", v_addr, v_idx_p[b], v_addr))
                    else:
                        self.add("valu", ("+", v_addr, v_idx_p[b], v_forest_p))
                    
                    for vi in range(VLEN):
                        self.add("load", ("load", v_node_val + vi, v_addr + vi))

                # Hash
                self.add("valu", ("^", v_val_p[b], v_val_p[b], v_node_val))
                self.build_hash_vector(v_val_p[b], v_regs[ti][1], v_regs[ti][2])
                
                # Update index
                v_t1 = v_regs[ti][1]
                self.add("valu", ("&", v_t1, v_val_p[b], v_one))
                self.add("valu", ("+", v_t1, v_t1, v_one))
                self.add("valu", ("multiply_add", v_idx_p[b], v_idx_p[b], v_two, v_t1))
                
                if is_last_level:
                    if USE_DYNAMIC_CONSTANTS:
                        self.add("valu", ("vbroadcast", v_regs[ti][2], self.scratch["n_nodes"]))
                        v_n = v_regs[ti][2]
                    else:
                        v_n = v_nn
                    
                    self.add("valu", ("<", v_t1, v_idx_p[b], v_n))
                    self.add("valu", ("*", v_idx_p[b], v_idx_p[b], v_t1))

        # Final Store

        for b in range(batch_size // VLEN):
            self.add("alu", ("+", ts_addr, self.scratch["inp_indices_p"], self.scratch_const(b * VLEN)))
            self.add("store", ("vstore", ts_addr, v_idx_p[b]))
            self.add("alu", ("+", ts_addr, self.scratch["inp_values_p"], self.scratch_const(b * VLEN)))
            self.add("store", ("vstore", ts_addr, v_val_p[b]))

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
        trace=trace,
        value_trace=value_trace,
    )
    machine.enable_pause = False
    machine.enable_debug = False
    machine.prints = prints
    machine.run()

    for ref_mem in reference_kernel2(mem):
        pass

    inp_values_p = ref_mem[6]
    assert (
        machine.mem[inp_values_p : inp_values_p + len(inp.values)]
        == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
    ), "Incorrect output values"
    print("CYCLES: ", machine.cycle)
    return machine.cycle
