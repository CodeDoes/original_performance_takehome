import subprocess
import sys
import re

def run_command(command):
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running {command}:")
        print(result.stdout)
        print(result.stderr)
        return None
    return result.stdout

def main():
    # 1. Run the submission tests
    # Capture output even if it fails (exit code != 0)
    result = subprocess.run("python tests/submission_tests.py", shell=True, capture_output=True, text=True)
    output = result.stdout
    error_output = result.stderr
    full_output = output + "\n" + error_output
    print(output)
    if error_output:
        print(error_output)

    # 2. Extract cycle count for the title
    match = re.search(r"CYCLES:\s+(\d+)", output)
    if match:
        cycles = match.group(1)
        speedup_match = re.search(r"Speedup over baseline:\s+([\d.]+)", output)
        speedup = speedup_match.group(1) if speedup_match else "N/A"
        commit_title = f"Cycles: {cycles} ({speedup}x)"
    else:
        commit_title = "Incremental optimization (Tests failed or CYCLES not found)"

    # 3. Prepare the full commit message
    commit_msg = f"{commit_title}\n\nTest Output:\n{full_output}"

    # 4. Commit (always)
    run_command("git add perf_takehome.py")
    
    # Use a temporary file for the commit message to handle multi-line content safely
    with open(".commit_msg.tmp", "w", encoding='utf-8') as f:
        f.write(commit_msg)
    
    # Commit always, even if tests failed (but only if there are changes to commit)
    commit_result = subprocess.run('git commit -F .commit_msg.tmp', shell=True, capture_output=True, text=True)
    if commit_result.returncode == 0:
        print(f"Successfully committed: {commit_title}")
    else:
        if "nothing to commit" in commit_result.stdout or "nothing to commit" in commit_result.stderr:
            print("Nothing to commit (no changes in perf_takehome.py).")
        else:
            print("Failed to commit:")
            print(commit_result.stdout)
            print(commit_result.stderr)

if __name__ == "__main__":
    main()
