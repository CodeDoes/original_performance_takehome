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

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Vectorized VLIW implementation.
        """
        init_vars = ["rounds", "n_nodes", "batch_size", "forest_height", "forest_values_p", "inp_indices_p", "inp_values_p"]
        for v in init_vars: self.alloc_scratch(v, 1)
        tmp1 = self.alloc_scratch("tmp1")
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        v_idx = self.alloc_scratch("v_idx", VLEN)
        v_val = self.alloc_scratch("v_val", VLEN)
        v_node_val = self.alloc_scratch("v_node_val", VLEN)
        v_addr = self.alloc_scratch("v_addr", VLEN)
        v_tmp1 = self.alloc_scratch("v_tmp1", VLEN)
        v_tmp2 = self.alloc_scratch("v_tmp2", VLEN)
        
        v_zero = self.vector_const(0)
        v_one = self.vector_const(1)
        v_two = self.vector_const(2)
        v_n_nodes = self.alloc_scratch("v_n_nodes", VLEN)
        self.add("valu", ("vbroadcast", v_n_nodes, self.scratch["n_nodes"]))
        v_forest_p = self.alloc_scratch("v_forest_p", VLEN)
        self.add("valu", ("vbroadcast", v_forest_p, self.scratch["forest_values_p"]))

        current_inp_indices_ptr = self.alloc_scratch("current_inp_indices_p_ptr")
        current_inp_values_ptr = self.alloc_scratch("current_inp_values_p_ptr")

        self.add("flow", ("pause",))
        body = []

        for round in range(rounds):
            body.append(("alu", ("+", current_inp_indices_ptr, self.scratch["inp_indices_p"], self.scratch_const(0))))
            body.append(("alu", ("+", current_inp_values_ptr, self.scratch["inp_values_p"], self.scratch_const(0))))

            for i in range(0, batch_size, VLEN):
                body.append(("load", ("vload", v_idx, current_inp_indices_ptr)))
                body.append(("load", ("vload", v_val, current_inp_values_ptr)))

                body.append(("valu", ("+", v_addr, v_forest_p, v_idx)))
                for vi in range(VLEN):
                    body.append(("load", ("load_offset", v_node_val, v_addr, vi)))

                body.append(("valu", ("^", v_val, v_val, v_node_val)))
                body.extend(self.build_hash_vec(v_val, v_tmp1, v_tmp2, round, i))

                body.append(("valu", ("%", v_tmp1, v_val, v_two)))
                body.append(("valu", ("==", v_tmp1, v_tmp1, v_zero)))
                body.append(("flow", ("vselect", v_tmp2, v_tmp1, v_one, v_two)))
                body.append(("valu", ("*", v_idx, v_idx, v_two)))
                body.append(("valu", ("+", v_idx, v_idx, v_tmp2)))

                body.append(("valu", ("<", v_tmp1, v_idx, v_n_nodes)))
                body.append(("flow", ("vselect", v_idx, v_tmp1, v_idx, v_zero)))

                body.append(("store", ("vstore", current_inp_indices_ptr, v_idx)))
                body.append(("store", ("vstore", current_inp_values_ptr, v_val)))

                body.append(("alu", ("+", current_inp_indices_ptr, current_inp_indices_ptr, self.scratch_const(VLEN))))
                body.append(("alu", ("+", current_inp_values_ptr, current_inp_values_ptr, self.scratch_const(VLEN))))
        body_instrs = self.build(body, vliw=True)
        self.instrs.extend(body_instrs)
        self.instrs.append({"flow": [("pause",)]})

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))

        return slots

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
