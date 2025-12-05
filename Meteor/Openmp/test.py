import subprocess
import time
import os
import glob
import matplotlib.pyplot as plt
import random
import statistics

# --- CONFIGURATION ---
CPP_FILE = "sol_openmp.cpp"
EXE_FILE = "openmp"
TEST_FOLDER = "test_cases"
NUM_TEST_CASES = 50       # Total files to generate
THREADS_TO_TEST = [1, 2, 4, 8, 16]

# Large sizes to ensure CPU is busy (OpenMP needs high workload)
MIN_N, MAX_N = 400000, 800000
P = 1000000000
def generate_bulk_test_cases():
    print(f"Creating folder '{TEST_FOLDER}'...")
    if not os.path.exists(TEST_FOLDER):
        os.makedirs(TEST_FOLDER)
    
    print(f"Generating {NUM_TEST_CASES} different test cases...")
    
    for i in range(NUM_TEST_CASES):
        filename = os.path.join(TEST_FOLDER, f"test_{i}.txt")
        
        # Randomize size slightly for realistic variance
        N = random.randint(MIN_N, MAX_N)
        M = N
        K = N
        
        with open(filename, "w") as f:
            f.write(f"{N} {M}\n")
            # Sectors
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
    print("Compiling C++ code...")
    # -O3 is crucial for optimized performance
    cmd = ["g++", "-O3", "-fopenmp", CPP_FILE, "-o", EXE_FILE]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("Compilation Failed!")
        print(result.stderr)
        exit(1)
    print("Compilation Successful.\n")

def run_benchmark():
    # Get list of all test files
    test_files = sorted(glob.glob(os.path.join(TEST_FOLDER, "*.txt")))
    
    avg_times = []
    speedups = []
    base_time = 0

    print(f"{'Threads':<10} | {'Avg Time (s)':<15} | {'Speedup':<10} | {'Total Time (s)':<15}")
    print("-" * 60)

    for t in THREADS_TO_TEST:
        os.environ["OMP_NUM_THREADS"] = str(t)
        
        total_duration_for_thread = 0.0
        
        # Run against ALL 50 test cases
        for test_file in test_files:
            with open(test_file, "r") as f:
                start_time = time.perf_counter()
                
                subprocess.run(
                    [f"./{EXE_FILE}"], 
                    stdin=f,           # Stream file directly to C++ cin
                    stdout=subprocess.DEVNULL, # Hide output to keep console clean
                    stderr=subprocess.DEVNULL
                )
                
                end_time = time.perf_counter()
                total_duration_for_thread += (end_time - start_time)
        
        # Calculate Average
        avg_time = total_duration_for_thread / NUM_TEST_CASES
        
        # Calculate Speedup
        if t == THREADS_TO_TEST[0]:
            base_time = avg_time
            speedup = 1.0
        else:
            speedup = base_time / avg_time

        avg_times.append(avg_time)
        speedups.append(speedup)
        
        print(f"{t:<10} | {avg_time:<15.4f} | {speedup:<10.2f}x | {total_duration_for_thread:<15.2f}")

    return avg_times, speedups

def plot_results(times, speedups):
    plt.figure(figsize=(12, 5))
    
    # Graph 1: Time vs Threads
    plt.subplot(1, 2, 1)
    plt.plot(THREADS_TO_TEST, times, marker='o', linestyle='-', color='b', linewidth=2)
    plt.title(f"Avg Execution Time (over {NUM_TEST_CASES} inputs)")
    plt.xlabel("Number of Threads")
    plt.ylabel("Time (seconds)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(THREADS_TO_TEST)

    # Graph 2: Speedup vs Threads
    plt.subplot(1, 2, 2)
    plt.plot(THREADS_TO_TEST, speedups, marker='o', linestyle='-', color='r', linewidth=2, label='Actual')
    plt.plot(THREADS_TO_TEST, THREADS_TO_TEST, linestyle='--', color='gray', label='Ideal Linear')
    plt.title("Speedup vs Threads")
    plt.xlabel("Number of Threads")
    plt.ylabel("Speedup Factor")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(THREADS_TO_TEST)
    plt.legend()

    plt.tight_layout()
    plt.savefig("final_benchmark_50_tests.png")
    print("\nGraphs saved as 'final_benchmark_50_tests.png'")
    plt.show()

if __name__ == "__main__":
    # Check if C++ file exists
    if not os.path.exists(CPP_FILE):
        print(f"Error: {CPP_FILE} not found. Please save your C++ code first.")
    else:
        generate_bulk_test_cases()
        compile_cpp()
        times, speedups = run_benchmark()
        plot_results(times, speedups)