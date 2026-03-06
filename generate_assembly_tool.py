
from perf_takehome import KernelBuilder
from problem import Machine, Tree, Input, build_mem_image, N_CORES, CoreState
import sys

def generate_assembly(forest_height=10, rounds=2, batch_size=16):
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    
    # Monkey-patch kb.build to return non-VLIW instructions for this tool.
    original_build = kb.build
    kb.build = lambda slots, vliw=True: original_build(slots, vliw=False)
    
    # Build with debug_markers=True
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds, debug_markers=True)

    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
    )
    
    # Detect rounds using the debug markers
    core = machine.cores[0]
    while core.pc < len(machine.program) and core.state != CoreState.STOPPED:
        instr = machine.program[core.pc]
        
        if "debug" in instr:
            for slot in instr["debug"]:
                if slot[0] == "compare" and isinstance(slot[2], tuple) and slot[2][2] == "start_round":
                    round_num = slot[2][0]
                    print(f"--- Start of Round {round_num} ---")
        
        rewritten = machine.rewrite_instr(instr)
        # Skip printing the debug marker itself if you want it exactly like the example
        if "debug" in rewritten and any(isinstance(s[2], tuple) and s[2][2] == "start_round" for s in rewritten["debug"]):
            pass 
        else:
            print(f"PC {core.pc}: {rewritten}")
        
        # Step the machine
        machine.step(instr, core)
        core.pc += 1
        
        if "flow" in instr:
            for slot in instr["flow"]:
                if slot[0] == "pause":
                    # Check if this is the preamble or end pause
                    if core.pc < 100: # heuristic for preamble
                        pass
                    else:
                        # In the example, end of round is marked
                        # Since we don't have pauses at end of every round, 
                        # this tool just prints them when they occur.
                        print(f"--- Pause at PC {core.pc-1} ---")

if __name__ == "__main__":
    generate_assembly(forest_height=2, rounds=2, batch_size=8)
