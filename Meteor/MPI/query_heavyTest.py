import subprocess
import time
import os
import glob
import matplotlib.pyplot as plt
import random

# --- CONFIGURATION ---
CPP_FILE = "sol_mpi.cpp"
EXE_FILE = "mpi.exe"
TEST_FOLDER = "test_cases_queryheavy"
PROCESSES_TO_TEST = [1, 2, 4, 8] # 16 might overshoot on a laptop

# --- HEAVY QUERY PARAMETERS ---
NUM_TESTS = 10         # 10 heavy tests is enough
N = 50000              # Fewer Nations (50k)
M = 50000            # MASSIVE SECTORS (3 Million) -> Heavy Parallel Work
K = 5000000              # Fewer Updates (50k) -> Low Serial Overhead
P = 1000000000
def generate_heavy_test_cases():
    print(f"Creating folder '{TEST_FOLDER}'...")
    if not os.path.exists(TEST_FOLDER):
        os.makedirs(TEST_FOLDER)
    
    print(f"Generating {NUM_TESTS} HEAVY test cases (N={N}, M={M})...")
    
    for i in range(NUM_TESTS):
        filename = os.path.join(TEST_FOLDER, f"heavy_{i}.txt")
        if os.path.exists(filename): continue # Skip if already exists
        
        with open(filename, "w") as f:
            f.write(f"{N} {M}\n")
            
            # 3 Million Sectors distributed among 50k nations
            # This makes the "member_sectors" vectors very large
            for _ in range(M):
                f.write(f"{random.randint(1, N)} ")
            f.write("\n")
            
            # Targets
            for _ in range(N):
                f.write(f"{random.randint(1, P)} ")
            f.write(f"\n{K}\n")
            
            # Updates
            for _ in range(K):
                l = random.randint(1, M)
                r = random.randint(1, M)
                v = random.randint(1, P)
                f.write(f"{l} {r} {v}\n")
    
    print("Generation complete.\n")

def compile_cpp():
    print("Compiling MPI C++ code...")
    cmd = ["wsl","mpic++", "-O3", CPP_FILE, "-o", EXE_FILE]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("Compilation Failed!")
        print(result.stderr)
        exit(1)
    print("Compilation Successful.\n")

def run_benchmark():
    test_files = sorted(glob.glob(os.path.join(TEST_FOLDER, "*.txt")))
    avg_times = []
    speedups = []
    base_time = 0

    print(f"{'Procs':<10} | {'Avg Time (s)':<15} | {'Speedup':<10}")
    print("-" * 45)

    for p in PROCESSES_TO_TEST:
        total_duration = 0.0
        
        for test_file in test_files:
            with open(test_file, "r") as f:
                start_time = time.perf_counter()
                
                # Using --oversubscribe to force execution if you lack cores
                cmd = ["wsl","mpirun", "--oversubscribe", "-np", str(p), f"./{EXE_FILE}"]
                
                subprocess.run(
                    cmd,
                    stdin=f,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                
                end_time = time.perf_counter()
                total_duration += (end_time - start_time)
        
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
    
    plt.subplot(1, 2, 1)
    plt.plot(PROCESSES_TO_TEST, times, marker='o', color='b', linewidth=2)
    plt.title(f"MPI Execution Time (M={M/1000000}M Sectors)")
    plt.xlabel("Processes")
    plt.ylabel("Time (s)")
    plt.grid(True)
    plt.xticks(PROCESSES_TO_TEST)

    plt.subplot(1, 2, 2)
    plt.plot(PROCESSES_TO_TEST, speedups, marker='o', color='r', linewidth=2, label='Actual')
    plt.plot(PROCESSES_TO_TEST, PROCESSES_TO_TEST, linestyle='--', color='gray', label='Ideal')
    plt.title("MPI Speedup (High Query Load)")
    plt.xlabel("Processes")
    plt.ylabel("Speedup Factor")
    plt.grid(True)
    plt.xticks(PROCESSES_TO_TEST)
    plt.legend()

    plt.tight_layout()
    plt.savefig("mpi_heavy_results.png")
    print("\nGraphs saved as 'mpi_heavy_results.png'")
    plt.show()

if __name__ == "__main__":
    generate_heavy_test_cases()
    compile_cpp()
    times, speedups = run_benchmark()
    plot_results(times, speedups)