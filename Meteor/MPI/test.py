import subprocess
import time
import os
import glob
import matplotlib.pyplot as plt
import statistics

# --- CONFIGURATION ---
CPP_FILE = "sol_mpi.cpp"   # Your MPI C++ Source
EXE_FILE = "mpi.exe"       # The compiled executable
TEST_FOLDER = "test_cases" # Must match the folder created previously

# VARY NUMBER OF PROCESSES (RANKS)
PROCESSES_TO_TEST = [1, 2, 4, 8, 16]

def compile_cpp():
    print("Compiling MPI C++ code...")
    # Using mpic++ wrapper
    cmd = ["wsl","mpic++", "-O3", CPP_FILE, "-o", EXE_FILE]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("Compilation Failed!")
        print(result.stderr)
        exit(1)
    print("Compilation Successful.\n")

def run_benchmark():
    # Check if test cases exist
    if not os.path.exists(TEST_FOLDER):
        print(f"Error: Folder '{TEST_FOLDER}' not found.")
        print("Please run the previous 'benchmark_suite.py' first to generate inputs.")
        exit(1)

    test_files = sorted(glob.glob(os.path.join(TEST_FOLDER, "*.txt")))
    if not test_files:
        print("Error: No .txt files found in test_cases folder.")
        exit(1)

    avg_times = []
    speedups = []
    base_time = 0

    print(f"{'Procs':<10} | {'Avg Time (s)':<15} | {'Speedup':<10}")
    print("-" * 45)

    for p in PROCESSES_TO_TEST:
        total_duration = 0.0
        
        # Run against ALL test cases
        for test_file in test_files:
            with open(test_file, "r") as f:
                start_time = time.perf_counter()
                
                # --- MPI EXECUTION COMMAND ---
                # --oversubscribe: Allows running 16 processes even if you only have 4 cores
                cmd = ["wsl","mpirun", "--oversubscribe", "-np", str(p), f"./{EXE_FILE}"]
                
                subprocess.run(
                    cmd,
                    stdin=f,           # Pipe file content to Rank 0's cin
                    stdout=subprocess.DEVNULL, # Hide output
                    stderr=subprocess.DEVNULL
                )
                
                end_time = time.perf_counter()
                total_duration += (end_time - start_time)
        
        # Calculate Statistics
        avg_time = total_duration / len(test_files)
        
        if p == PROCESSES_TO_TEST[0]:
            base_time = avg_time
            speedup = 1.0
        else:
            speedup = base_time / avg_time

        avg_times.append(avg_time)
        speedups.append(speedup)
        
        print(f"{p:<10} | {avg_time:<15.4f} | {speedup:<10.2f}x")

    return avg_times, speedups

def plot_results(times, speedups):
    plt.figure(figsize=(12, 5))
    
    # Graph 1: Execution Time vs Processes
    plt.subplot(1, 2, 1)
    plt.plot(PROCESSES_TO_TEST, times, marker='o', linestyle='-', color='b', linewidth=2)
    plt.title(f"MPI Execution Time (Avg of {len(glob.glob(os.path.join(TEST_FOLDER, '*.txt')))} tests)")
    plt.xlabel("Number of Processes (Nodes)")
    plt.ylabel("Time (seconds)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(PROCESSES_TO_TEST)

    # Graph 2: Speedup vs Processes
    plt.subplot(1, 2, 2)
    plt.plot(PROCESSES_TO_TEST, speedups, marker='o', linestyle='-', color='r', linewidth=2, label='Actual')
    plt.plot(PROCESSES_TO_TEST, PROCESSES_TO_TEST, linestyle='--', color='gray', label='Ideal Linear')
    plt.title("Speedup vs Processes")
    plt.xlabel("Number of Processes (Nodes)")
    plt.ylabel("Speedup Factor")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(PROCESSES_TO_TEST)
    plt.legend()

    plt.tight_layout()
    plt.savefig("mpi_benchmark_results.png")
    print("\nGraphs saved as 'mpi_benchmark_results.png'")
    plt.show()

if __name__ == "__main__":
    compile_cpp()
    times, speedups = run_benchmark()
    plot_results(times, speedups)