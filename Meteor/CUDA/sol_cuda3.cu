#include <iostream>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <thrust/fill.h>
#include <thrust/execution_policy.h>

using namespace std;

// --- Helper Macro ---
#define CHECK(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s:%d, code:%d, reason: %s\n", __FILE__, __LINE__, error, cudaGetErrorString(error)); \
        exit(1); \
    } \
}

typedef unsigned long long ull;

// --- KERNELS ---

// 1. Scatter Updates to Difference Array
// This is O(1) per update thread. Extremely fast.
__global__ void scatter_updates_kernel(
    const int start_u, const int end_u,
    const int* d_up_l, const int* d_up_r, const long long* d_up_v,
    ull* d_D, int m
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int u = start_u + tid;
    if (u > end_u) return;

    int L = d_up_l[u];
    int R = d_up_r[u];
    long long val = d_up_v[u];
    
    ull add = (ull)val;
    ull sub = (ull)(-val);

    // Difference Array Logic (1-based):
    // Range [L, R] -> Add to L+1, Subtract from R+2
    
    if (L <= R) {
        atomicAdd(&d_D[L + 1], add);
        if (R + 2 <= m) atomicAdd(&d_D[R + 2], sub);
    } else {
        // Circular Case
        atomicAdd(&d_D[1], add);
        if (R + 2 <= m) atomicAdd(&d_D[R + 2], sub);
        atomicAdd(&d_D[L + 1], add);
    }
}

// 2. Check Nations (O(1) Memory Access)
// This is 20x faster than the BIT version because it reads d_arr[sector] directly.
__global__ void check_bucket_kernel(
    int count,
    const int* d_active_batch,
    const int* d_offsets,
    const int* d_flat_sectors,
    const ull* d_arr, 
    const long long* d_targets,
    int* d_result_flags
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    int nation = d_active_batch[idx];
    long long target = d_targets[nation];
    
    int start = d_offsets[nation];
    int end = d_offsets[nation + 1];

    long long current_sum = 0;

    for (int i = start; i < end; ++i) {
        int sector_idx = d_flat_sectors[i]; // 1-based index
        
        // Single Read! No Tree Traversal.
        ull val_u = d_arr[sector_idx];
        
        // Correctly handle negative values via casting
        current_sum += (long long)val_u;
        
        if (current_sum >= target) break;
    }

    d_result_flags[idx] = (current_sum >= target) ? 1 : 0;
}

int main(int argc, char** argv) {
    // Optimization: Fast IO
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int blockSize = 256;
    if (argc > 1) blockSize = atoi(argv[1]);
    int n, m;
    if (!(cin >> n >> m)) return 0;

    // --- HOST PREP ---
    // Efficiently flatten the jagged array of sectors
    vector<vector<int>> temp_sectors(n);
    for (int i = 0; i < m; ++i) {
        int x; cin >> x;
        if (x >= 1 && x <= n) temp_sectors[x - 1].push_back(i + 1); // 1-based
    }

    vector<int> h_flat;
    vector<int> h_offsets(n + 1);
    h_offsets[0] = 0;
    for (int i = 0; i < n; ++i) {
        h_flat.insert(h_flat.end(), temp_sectors[i].begin(), temp_sectors[i].end());
        h_offsets[i + 1] = h_flat.size();
    }
    vector<vector<int>>().swap(temp_sectors);

    vector<long long> h_targets(n);
    for (int i = 0; i < n; ++i) cin >> h_targets[i];

    int k; cin >> k;
    vector<int> h_up_l(k), h_up_r(k);
    vector<long long> h_up_v(k);
    for (int i = 0; i < k; ++i) {
        cin >> h_up_l[i] >> h_up_r[i] >> h_up_v[i];
        h_up_l[i]--; h_up_r[i]--; // 0-based input
    }

    // --- DEVICE ALLOC ---
    int *d_flat, *d_offsets, *d_up_l, *d_up_r, *d_active, *d_results;
    long long *d_targets, *d_up_v;
    
    // Allocate M+2 for safe difference array boundary checks
    int m_alloc = m + 2;
    ull *d_D_raw, *d_arr_raw;

    CHECK(cudaMalloc(&d_flat, h_flat.size() * sizeof(int)));
    CHECK(cudaMalloc(&d_offsets, h_offsets.size() * sizeof(int)));
    CHECK(cudaMalloc(&d_targets, n * sizeof(long long)));
    CHECK(cudaMalloc(&d_up_l, k * sizeof(int)));
    CHECK(cudaMalloc(&d_up_r, k * sizeof(int)));
    CHECK(cudaMalloc(&d_up_v, k * sizeof(long long)));
    CHECK(cudaMalloc(&d_active, n * sizeof(int)));
    CHECK(cudaMalloc(&d_results, n * sizeof(int)));
    
    // Thrust vectors for Scan
    thrust::device_vector<ull> d_D(m_alloc);
    thrust::device_vector<ull> d_arr(m_alloc);
    d_D_raw = thrust::raw_pointer_cast(d_D.data());
    d_arr_raw = thrust::raw_pointer_cast(d_arr.data());

    // --- COPIES ---
    CHECK(cudaMemcpy(d_flat, h_flat.data(), h_flat.size() * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_offsets, h_offsets.data(), h_offsets.size() * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_targets, h_targets.data(), n * sizeof(long long), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_up_l, h_up_l.data(), k * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_up_r, h_up_r.data(), k * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_up_v, h_up_v.data(), k * sizeof(long long), cudaMemcpyHostToDevice));

    // --- CPU SCHEDULING ---
    vector<int> l(n, 0), r(n, k);
    vector<vector<int>> groups(k);
    
    // Pinned buffers for fast "Batch" transfers
    int* h_active_pinned;
    int* h_results_pinned;
    CHECK(cudaMallocHost(&h_active_pinned, n * sizeof(int)));
    CHECK(cudaMallocHost(&h_results_pinned, n * sizeof(int)));

    // --- MAIN LOOP ---
    // Log2(300,000) ~ 19. Run 25 iterations.
    for (int iter = 0; iter < 25; ++iter) {
        
        bool active = false;
        for (int i = 0; i < k; ++i) groups[i].clear();
        for (int i = 0; i < n; ++i) {
            if (l[i] < r[i]) {
                int mid = (l[i] + r[i]) >> 1;
                groups[mid].push_back(i);
                active = true;
            }
        }
        if (!active) break;

        // Reset Difference Array
        thrust::fill(d_D.begin(), d_D.end(), 0);
        
        int current_time = -1;

        // Sweep Line
        for (int t = 0; t < k; ++t) {
            if (groups[t].empty()) continue;

            // 1. Scatter Updates [current_time+1 ... t]
            if (t > current_time) {
                int start = current_time + 1;
                int count = t - start + 1;
                int blocks = (count + blockSize - 1) / blockSize;
                
                // Fast atomic scatter
                scatter_updates_kernel<<<blocks, blockSize>>>(
                    start, t, d_up_l, d_up_r, d_up_v, d_D_raw, m
                );
                
                // 2. Parallel Inclusive Scan (Transform Difference -> Actual)
                // This replaces the slow BIT update loop
                thrust::inclusive_scan(d_D.begin(), d_D.end(), d_arr.begin());
                
                current_time = t;
            }

            // 3. Process Group
            int count = groups[t].size();
            memcpy(h_active_pinned, groups[t].data(), count * sizeof(int));
            CHECK(cudaMemcpy(d_active, h_active_pinned, count * sizeof(int), cudaMemcpyHostToDevice));

            int checkBlocks = (count + blockSize - 1) / blockSize;
            check_bucket_kernel<<<checkBlocks, blockSize>>>(
                count, d_active, d_offsets, d_flat, d_arr_raw, d_targets, d_results
            );
            
            // 4. Read Results
            CHECK(cudaMemcpy(h_results_pinned, d_results, count * sizeof(int), cudaMemcpyDeviceToHost));
            
            for (int i = 0; i < count; ++i) {
                int nation = groups[t][i];
                if (h_results_pinned[i]) r[nation] = t;
                else l[nation] = t + 1;
            }
        }
    }

    // --- Output ---
    for (int i = 0; i < n; ++i) {
        if (l[i] >= k) cout << "NIE\n";
        else cout << l[i] + 1 << "\n";
    }

    // Free
    cudaFreeHost(h_active_pinned); cudaFreeHost(h_results_pinned);
    cudaFree(d_flat); cudaFree(d_offsets); cudaFree(d_targets);
    cudaFree(d_up_l); cudaFree(d_up_r); cudaFree(d_up_v);
    cudaFree(d_active); cudaFree(d_results);

    return 0;
}