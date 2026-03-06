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
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        if not vliw:
            instrs = []
            for engine, slot in slots:
                instrs.append({engine: [slot]})
            return instrs

        # Simple greedy VLIW packer
        instrs = []
        current_instr = defaultdict(list)
        current_outputs = set()
        current_inputs = set()

        def get_regs(engine, slot):
            inputs = set()
            outputs = set()
            if engine == "alu":
                op, dest, a1, a2 = slot
                outputs.add(dest)
                inputs.update([a1, a2])
            elif engine == "valu":
                if slot[0] == "vbroadcast":
                    _, dest, src = slot
                    outputs.update(range(dest, dest + VLEN))
                    inputs.add(src)
                elif slot[0] == "multiply_add":
                    _, dest, a, b, c = slot
                    outputs.update(range(dest, dest + VLEN))
                    inputs.update(range(a, a + VLEN))
                    inputs.update(range(b, b + VLEN))
                    inputs.update(range(c, c + VLEN))
                else:
                    op, dest, a1, a2 = slot
                    outputs.update(range(dest, dest + VLEN))
                    inputs.update(range(a1, a1 + VLEN))
                    inputs.update(range(a2, a2 + VLEN))
            elif engine == "load":
                if slot[0] == "load":
                    _, dest, addr = slot
                    outputs.add(dest)
                    inputs.add(addr)
                elif slot[0] == "load_offset":
                    _, dest, addr, offset = slot
                    outputs.add(dest + offset)
                    inputs.add(addr + offset)
                elif slot[0] == "vload":
                    _, dest, addr = slot
                    outputs.update(range(dest, dest + VLEN))
                    inputs.add(addr)
                elif slot[0] == "const":
                    _, dest, val = slot
                    outputs.add(dest)
            elif engine == "store":
                if slot[0] == "store":
                    _, addr, src = slot
                    inputs.update([addr, src])
                elif slot[0] == "vstore":
                    _, addr, src = slot
                    inputs.add(addr)
                    inputs.update(range(src, src + VLEN))
            elif engine == "flow":
                if slot[0] == "select":
                    _, dest, cond, a, b = slot
                    outputs.add(dest)
                    inputs.update([cond, a, b])
                elif slot[0] == "vselect":
                    _, dest, cond, a, b = slot
                    outputs.update(range(dest, dest + VLEN))
                    inputs.update(range(cond, cond + VLEN))
                    inputs.update(range(a, a + VLEN))
                    inputs.update(range(b, b + VLEN))
                elif slot[0] == "add_imm":
                    _, dest, a, imm = slot
                    outputs.add(dest)
                    inputs.add(a)
            return inputs, outputs

        for engine, slot in slots:
            if engine == "debug" or engine == "pause": # debug and pause are special
                if current_instr:
                    instrs.append(dict(current_instr))
                    current_instr = defaultdict(list)
                    current_outputs = set()
                    current_inputs = set()
                instrs.append({engine: [slot]})
                continue

            inputs, outputs = get_regs(engine, slot)
            
            # Dependency check
            can_pack = (
                len(current_instr[engine]) < SLOT_LIMITS.get(engine, 1) and
                not (inputs & current_outputs) and
                not (outputs & current_inputs) and
                not (outputs & current_outputs)
            )

            if can_pack:
                current_instr[engine].append(slot)
                current_outputs.update(outputs)
                current_inputs.update(inputs)
            else:
                instrs.append(dict(current_instr))
                current_instr = defaultdict(list)
                current_instr[engine].append(slot)
                current_outputs = outputs
                current_inputs = inputs
        
        if current_instr:
            instrs.append(dict(current_instr))
            
        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

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

    def vector_const(self, val):
        if val not in self.const_map:
            scalar_addr = self.alloc_scratch(f"const_{val}")
            self.add("load", ("const", scalar_addr, val))
            self.const_map[val] = scalar_addr
        
        v_key = f"v_const_{val}"
        if v_key not in self.scratch:
            v_addr = self.alloc_scratch(v_key, VLEN)
            self.add("valu", ("vbroadcast", v_addr, self.const_map[val]))
        return self.scratch[v_key]

    def build_hash_vec(self, v_val_hash_addr, v_tmp1, v_tmp2, round, i_start):
        slots = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            v_const1 = self.vector_const(val1)
            v_const3 = self.vector_const(val3)
            slots.append(("valu", (op1, v_tmp1, v_val_hash_addr, v_const1)))
            slots.append(("valu", (op3, v_tmp2, v_val_hash_addr, v_const3)))
            slots.append(("valu", (op2, v_val_hash_addr, v_tmp1, v_tmp2)))
        return slots

    def alloc_vector_regs(self, suffix: str):
        """Allocates a set of vector registers for one unrolled iteration."""
        return {
            "v_idx": self.alloc_scratch(f"v_idx{suffix}", VLEN),
            "v_val": self.alloc_scratch(f"v_val{suffix}", VLEN),
            "v_node_val": self.alloc_scratch(f"v_node_val{suffix}", VLEN),
            "v_addr": self.alloc_scratch(f"v_addr{suffix}", VLEN),
            "v_tmp1": self.alloc_scratch(f"v_tmp1{suffix}", VLEN),
            "v_tmp2": self.alloc_scratch(f"v_tmp2{suffix}", VLEN),
        }

    def emit_gather(self, regs_list: list[dict]):
        """Interleaves gathering node values for multiple vector groups using scalar loads."""
        slots = []
        forest_values_p_scalar = self.scratch["forest_values_p"]
        tmp_scalar_addr = self.alloc_scratch("tmp_scalar_addr") # New scratch for scalar address

        for vi in range(VLEN):
            for r in regs_list:
                # Calculate individual address: forest_values_p + v_idx[vi]
                slots.append(("alu", ("+", tmp_scalar_addr, forest_values_p_scalar, r["v_idx"] + vi)))
                # Load individual element into v_node_val[vi]
                slots.append(("load", ("load", r["v_node_val"] + vi, tmp_scalar_addr)))
        return slots

    def emit_interleaved_hash(self, regs_list: list[dict]):
        """Interleaves the hash stages across all unrolled vector groups."""
        slots = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            v_const1 = self.vector_const(val1)
            v_const3 = self.vector_const(val3)
            for r in regs_list:
                slots.append(("valu", (op1, r["v_tmp1"], r["v_val"], v_const1)))
            for r in regs_list:
                slots.append(("valu", (op3, r["v_tmp2"], r["v_val"], v_const3)))
            for r in regs_list:
                slots.append(("valu", (op2, r["v_val"], r["v_tmp1"], r["v_tmp2"])))
        return slots
    def emit_state_update(self, r: dict, v_n_nodes, v_zero, v_one, v_two):
        """Logic for updating the index and wrapping it."""
        slots = []
        slots.append(("valu", ("%", r["v_tmp1"], r["v_val"], v_two)))
        slots.append(("valu", ("==", r["v_tmp1"], r["v_tmp1"], v_zero)))
        slots.append(("flow", ("vselect", r["v_tmp2"], r["v_tmp1"], v_one, v_two)))
        slots.append(("valu", ("*", r["v_idx"], r["v_idx"], v_two)))
        slots.append(("valu", ("+", r["v_idx"], r["v_idx"], r["v_tmp2"])))
        slots.append(("valu", ("<", r["v_tmp1"], r["v_idx"], v_n_nodes)))
        slots.append(("flow", ("vselect", r["v_idx"], r["v_tmp1"], r["v_idx"], v_zero)))
        return slots

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Highly optimized vectorized VLIW implementation with 4x unrolling.
        """
        init_vars = ["rounds", "n_nodes", "batch_size", "forest_height", "forest_values_p", "inp_indices_p", "inp_values_p"]
        for v in init_vars: self.alloc_scratch(v, 1)
        tmp_setup = self.alloc_scratch("tmp_setup")
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp_setup, i))
            self.add("load", ("load", self.scratch[v], tmp_setup))

        v_zero, v_one, v_two = self.vector_const(0), self.vector_const(1), self.vector_const(2)
        v_n_nodes = self.alloc_scratch("v_n_nodes", VLEN)
        self.add("valu", ("vbroadcast", v_n_nodes, self.scratch["n_nodes"]))


        UNROLL = 4
        regs_list = [self.alloc_vector_regs(f"_{j}") for j in range(UNROLL)]
        ptr_indices = [self.alloc_scratch(f"ptr_idx_{j}") for j in range(UNROLL)]
        ptr_values = [self.alloc_scratch(f"ptr_val_{j}") for j in range(UNROLL)]

        self.add("flow", ("pause",))
        curr_idx_base, curr_val_base = self.alloc_scratch("curr_idx_base"), self.alloc_scratch("curr_val_base")

        for round in range(rounds):
            self.add("alu", ("+", curr_idx_base, self.scratch["inp_indices_p"], self.scratch_const(0)))
            self.add("alu", ("+", curr_val_base, self.scratch["inp_values_p"], self.scratch_const(0)))

            for i in range(0, batch_size, VLEN * UNROLL):
                for j in range(UNROLL):
                    self.add("flow", ("add_imm", ptr_indices[j], curr_idx_base, j * VLEN))
                    self.add("flow", ("add_imm", ptr_values[j], curr_val_base, j * VLEN))

                for j in range(UNROLL):
                    self.add("load", ("vload", regs_list[j]["v_idx"], ptr_indices[j]))
                    self.add("load", ("vload", regs_list[j]["v_val"], ptr_values[j]))

                # Gather
                for engine, slot in self.emit_gather(regs_list):
                    self.add(engine, slot)

                # XOR and Hash
                for r in regs_list:
                    # v_val = v_val ^ v_node_val
                    self.add("valu", ("^", r["v_val"], r["v_val"], r["v_node_val"]))
                for engine, slot in self.emit_interleaved_hash(regs_list):
                    self.add(engine, slot)

                # State Update and Store
                for j in range(UNROLL):
                    r = regs_list[j]
                    for engine, slot in self.emit_state_update(r, v_n_nodes, v_zero, v_one, v_two):
                        self.add(engine, slot)
                    self.add("store", ("vstore", ptr_indices[j], r["v_idx"]))
                    self.add("store", ("vstore", ptr_values[j], r["v_val"]))