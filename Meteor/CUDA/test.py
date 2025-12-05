import subprocess
import time
import os
import glob
import matplotlib.pyplot as plt
import random

# --- CONFIGURATION ---
EXE_FILE = "cuda2.exe"       # Executable path (use "meteors.exe" on Windows)
TEST_FOLDER = "test_cases_heavy"   # The folder you created earlier with 50 tests
BLOCK_SIZES = [16,32, 64, 128, 256, 512, 1024]

# --- HEAVY QUERY PARAMETERS ---
NUM_TESTS = 10         # 10 heavy tests is enough
N = 5000000              # Fewer Nations (50k)
M = 50000            # MASSIVE SECTORS (3 Million) -> Heavy Parallel Work
K = 50000              # Fewer Updates (50k) -> Low Serial Overhead
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
def run_benchmark():
    if not os.path.exists(TEST_FOLDER):
        print(f"Error: Folder '{TEST_FOLDER}' not found!")
        return [], []

    test_files = sorted(glob.glob(os.path.join(TEST_FOLDER, "*.txt")))
    if not test_files:
        print(f"Error: No .txt files found in '{TEST_FOLDER}'")
        return [], []

    print(f"Benchmarking on {len(test_files)} existing test cases...")
    print(f"{'Block Size':<15} | {'Avg Time (s)':<15} | {'Speedup':<15}")
    print("-" * 50)

    avg_times = []
    speedups = []
    base_time = 0

    for bs in BLOCK_SIZES:
        total_time = 0.0
        
        # Run against ALL files in the folder
        for test_file in test_files:
            with open(test_file, "r") as f:
                start = time.perf_counter()
                
                # Pass Block Size as argument to C++ program
                subprocess.run(
                    [EXE_FILE, str(bs)], 
                    stdin=f,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                
                end = time.perf_counter()
                total_time += (end - start)
        
        # Calculate Average for this Block Size
        avg_time = total_time / len(test_files)
        
        # Calculate Speedup relative to 32 threads
        if bs == BLOCK_SIZES[0]:
            base_time = avg_time
            speedup = 1.0
        else:
            speedup = base_time / avg_time

        avg_times.append(avg_time)
        speedups.append(speedup)
        
        print(f"{bs:<15} | {avg_time:<15.4f} | {speedup:<15.2f}x")

    return avg_times, speedups

def plot_results(times, speedups):
    plt.figure(figsize=(14, 6))

    # Plot 1: Time vs Block Size
    plt.subplot(1, 2, 1)
    plt.plot(BLOCK_SIZES, times, marker='o', linestyle='-', color='b', linewidth=2)
    plt.xscale('log', base=2)
    plt.xticks(BLOCK_SIZES, BLOCK_SIZES)
    plt.title(f"Block Size vs Time (Avg of {len(times)} tests)")
    plt.xlabel("Threads per Block")
    plt.ylabel("Avg Execution Time (s)")
    plt.grid(True, which="both", ls="--", alpha=0.5)

    # Plot 2: Speedup vs Block Size
    plt.subplot(1, 2, 2)
    plt.plot(BLOCK_SIZES, speedups, marker='s', linestyle='-', color='g', linewidth=2)
    plt.xscale('log', base=2)
    plt.xticks(BLOCK_SIZES, BLOCK_SIZES)
    plt.title("Block Size vs Speedup (Relative to 32)")
    plt.xlabel("Threads per Block")
    plt.ylabel("Speedup Factor")
    plt.grid(True, which="both", ls="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("cuda_block_benchmark.png")
    print("\nGraphs saved as 'cuda_block_benchmark.png'")
    plt.show()

if __name__ == "__main__":
    generate_heavy_test_cases()
    t, s = run_benchmark()
    if t:
        plot_results(t, s)