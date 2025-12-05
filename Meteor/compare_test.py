import subprocess
import time
import os
import glob
import matplotlib.pyplot as plt
import statistics

# --- CONFIGURATION ---
# Ensure these match your compiled executable names
EXE_OPENMP = "openmp.exe"  # or solution_openmp.exe
EXE_MPI    = "./mpi.exe"     # or solution_mpi.exe
EXE_CUDA   = "cuda2.exe"    # or meteors.exe / meteors

# Folder containing the 50 test cases
TEST_FOLDER = "test_cases"

def run_suite(command, name, test_files):
    print(f"Benchmarking {name} on {len(test_files)} files...")
    times = []
    
    try:
        for filepath in test_files:
            with open(filepath, "r") as f:
                start = time.perf_counter()
                # Run the executable
                subprocess.run(
                    command, 
                    stdin=f, 
                    stdout=subprocess.DEVNULL, 
                    stderr=subprocess.DEVNULL, 
                    check=True
                )
                end = time.perf_counter()
                times.append(end - start)
        
        avg_time = statistics.mean(times)
        print(f"  -> Average Time: {avg_time:.4f} s")
        return avg_time

    except Exception as e:
        print(f"  -> Error running {name}: {e}")
        return None

def main():
    # 1. Find Test Cases
    if not os.path.exists(TEST_FOLDER):
        print(f"Error: Folder '{TEST_FOLDER}' not found. Run the test generation script first.")
        return

    test_files = sorted(glob.glob(os.path.join(TEST_FOLDER, "*.txt")))
    if not test_files:
        print(f"Error: No .txt files found in '{TEST_FOLDER}'.")
        return

    print(f"Found {len(test_files)} test cases. Starting comparison...")
    print("-" * 50)

    # 2. Define Configurations
    # Format: (Label, Command List, Color)
    configs = [
        ("Serial (1 Thread)", [EXE_OPENMP], "gray"),
        ("OpenMP (4 Threads)", [EXE_OPENMP], "blue"),
        ("MPI (4 Ranks)", ["wsl","mpirun", "--oversubscribe", "-np", "4", EXE_MPI], "orange"),
        ("CUDA (GPU)", [EXE_CUDA], "green")
    ]

    results = {} # Name -> Avg Time

    # 3. Run Benchmarks
    # Special handling for Environment Variables (OpenMP)
    
    # A. Serial Run
    os.environ["OMP_NUM_THREADS"] = "1"
    results["Serial"] = run_suite(configs[0][1], configs[0][0], test_files)

    # B. OpenMP Run
    os.environ["OMP_NUM_THREADS"] = "4" # Adjust to your core count
    results["OpenMP"] = run_suite(configs[1][1], configs[1][0], test_files)

    # C. MPI Run
    results["MPI"] = run_suite(configs[2][1], configs[2][0], test_files)

    # D. CUDA Run
    results["CUDA"] = run_suite(configs[3][1], configs[3][0], test_files)

    # 4. Process Data
    names = []
    times = []
    colors = []
    
    # Filter out failed runs
    valid_configs = []
    for label, cmd, color in configs:
        key = label.split(" (")[0] # Simple key like "Serial", "OpenMP"
        if key == "Serial": val = results["Serial"]
        elif key == "OpenMP": val = results["OpenMP"]
        elif key == "MPI": val = results["MPI"]
        elif key == "CUDA": val = results["CUDA"]
        
        if val is not None:
            names.append(label)
            times.append(val)
            colors.append(color)

    if not times:
        print("No results to plot.")
        return

    # Calculate Speedups
    baseline = results["Serial"]
    speedups = [baseline / t for t in times]

    # 5. Plot Graphs
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Graph 1: Average Execution Time
    ax1.bar(names, times, color=colors)
    ax1.set_title(f"Avg Execution Time (over {len(test_files)} tests)")
    ax1.set_ylabel("Time (seconds)")
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    for i, v in enumerate(times):
        ax1.text(i, v + 0.01, f"{v:.3f}s", ha='center', fontweight='bold')

    # Graph 2: Speedup Factor
    ax2.bar(names, speedups, color=colors)
    ax2.set_title("Speedup vs Serial Baseline")
    ax2.set_ylabel("Speedup Factor (x)")
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    for i, v in enumerate(speedups):
        ax2.text(i, v + 0.1, f"{v:.1f}x", ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig("final_comparative_analysis.png")
    print("-" * 50)
    print("Comparison complete! Graph saved as 'final_comparative_analysis.png'")
    plt.show()

if __name__ == "__main__":
    main()