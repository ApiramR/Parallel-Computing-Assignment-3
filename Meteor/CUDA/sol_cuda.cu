#include <iostream>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

using namespace std;

#define CHECK(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s:%d, code:%d, reason: %s\n", __FILE__, __LINE__, error, cudaGetErrorString(error)); \
        exit(1); \
    } \
}

typedef unsigned long long ull;

// --- KERNELS ---

__global__ void clear_tree_kernel(ull* tree, int m) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx <= m) tree[idx] = 0;
}

// BATCH UPDATE (Safe Signed Math)
__global__ void batch_update_kernel(
    ull* tree, 
    int m, 
    const int* up_l, const int* up_r, const long long* up_v,
    int start, int end
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int u_idx = start + idx;
    if (u_idx > end) return;

    int l = up_l[u_idx];
    int r = up_r[u_idx];
    ull val = (ull)up_v[u_idx];
    ull nval = (ull)(-up_v[u_idx]);

    if (l <= r) {
        for (int i = l + 1; i <= m; i += i & -i) atomicAdd(&tree[i], val);
        for (int i = r + 2; i <= m; i += i & -i) atomicAdd(&tree[i], nval);
    } else {
        for (int i = 1; i <= m; i += i & -i) atomicAdd(&tree[i], val);
        for (int i = r + 2; i <= m; i += i & -i) atomicAdd(&tree[i], nval);
        for (int i = l + 1; i <= m; i += i & -i) atomicAdd(&tree[i], val);
    }
}

// QUERY KERNEL
__global__ void check_nations_kernel(
    int count,
    const int* active_batch, 
    const int* offsets,
    const int* flat_sectors,
    const ull* tree,
    const long long* targets,
    int* results
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    int nation_id = active_batch[idx];
    long long target = targets[nation_id];
    long long current_sum = 0;
    
    int start = offsets[nation_id];
    int end = offsets[nation_id + 1];

    for (int i = start; i < end; ++i) {
        int sector = flat_sectors[i];
        ull s = 0;
        int x = sector;
        while (x > 0) {
            s += tree[x];
            x -= (x & -x);
        }
        current_sum += (long long)s;
        if (current_sum >= target) break;
    }

    results[nation_id] = (current_sum >= target) ? 1 : 0;
}

// Helper struct for sorting on CPU
struct NationMeta {
    int id;
    int mid;
    // Overload < for sorting
    bool operator<(const NationMeta& other) const {
        return mid < other.mid;
    }
};

int main(int argc, char** argv) {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int n, m;
    if (!(cin >> n >> m)) return 0;

    // 1. Host Prep
    vector<vector<int>> temp_sectors(n);
    for (int i = 0; i < m; ++i) {
        int x; cin >> x;
        if (x >= 1 && x <= n) temp_sectors[x - 1].push_back(i + 1);
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
        h_up_l[i]--; h_up_r[i]--;
    }

    // 2. GPU Alloc
    int *d_flat, *d_offsets, *d_up_l, *d_up_r, *d_active_batch, *d_results;
    long long *d_targets, *d_up_v;
    ull *d_tree;

    CHECK(cudaMalloc(&d_flat, h_flat.size() * sizeof(int)));
    CHECK(cudaMalloc(&d_offsets, h_offsets.size() * sizeof(int)));
    CHECK(cudaMalloc(&d_targets, n * sizeof(long long)));
    CHECK(cudaMalloc(&d_tree, (m + 5) * sizeof(ull)));
    CHECK(cudaMalloc(&d_up_l, k * sizeof(int)));
    CHECK(cudaMalloc(&d_up_r, k * sizeof(int)));
    CHECK(cudaMalloc(&d_up_v, k * sizeof(long long)));
    CHECK(cudaMalloc(&d_results, n * sizeof(int)));

    CHECK(cudaMemcpy(d_flat, h_flat.data(), h_flat.size() * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_offsets, h_offsets.data(), h_offsets.size() * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_targets, h_targets.data(), n * sizeof(long long), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_up_l, h_up_l.data(), k * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_up_r, h_up_r.data(), k * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_up_v, h_up_v.data(), k * sizeof(long long), cudaMemcpyHostToDevice));

    // Pinned Memory for fast transfers
    int* h_active_batch;
    CHECK(cudaMallocHost(&h_active_batch, n * sizeof(int))); // Pinned
    CHECK(cudaMalloc(&d_active_batch, n * sizeof(int)));

    // 3. Binary Search State
    vector<int> l(n, 0), r(n, k);
    vector<int> h_results(n); 
    
    // Sort Buffer
    vector<NationMeta> schedule;
    schedule.reserve(n);

    // --- MAIN LOOP ---
    // Read Block Size from Command Line
    int blockSize = 256; // Default
    if (argc > 1) {
        blockSize = atoi(argv[1]);
        if (blockSize <= 0) blockSize = 256;
    }
    for (int iter = 0; iter < 25; ++iter) {
        
        // A. CPU: Build Schedule
        schedule.clear();
        for (int i = 0; i < n; ++i) {
            if (l[i] < r[i]) {
                int mid = (l[i] + r[i]) >> 1;
                schedule.push_back({i, mid});
            }
        }
        
        if (schedule.empty()) break;

        // B. Sort Schedule by 'mid' (Time)
        // This ensures we process time steps linearly
        std::sort(schedule.begin(), schedule.end());

        // C. Clear Tree
        int treeBlocks = (m + blockSize - 1) / blockSize;
        clear_tree_kernel<<<treeBlocks, blockSize>>>(d_tree, m);
        cudaDeviceSynchronize();

        int current_time = -1;
        int processed_count = 0;

        // D. Sweep Line (Jumping over empty gaps)
        while (processed_count < schedule.size()) {
            int target_time = schedule[processed_count].mid;
            
            // 1. Catch up updates
            int updates_needed = target_time - current_time;
            if (updates_needed > 0) {
                int start = current_time + 1;
                int end = target_time; // inclusive
                int count = end - start + 1;
                int blocks = (count + blockSize - 1) / blockSize;
                
                batch_update_kernel<<<blocks, blockSize>>>(d_tree, m, d_up_l, d_up_r, d_up_v, start, end);
                current_time = target_time;
            }

            // 2. Identify Batch (All nations with same mid)
            int batch_start_idx = processed_count;
            while (processed_count < schedule.size() && schedule[processed_count].mid == target_time) {
                // Collect IDs into Pinned Buffer
                h_active_batch[processed_count - batch_start_idx] = schedule[processed_count].id;
                processed_count++;
            }
            int batch_size = processed_count - batch_start_idx;

            // 3. Upload & Query
            // Copy is fast because h_active_batch is Pinned Memory
            CHECK(cudaMemcpy(d_active_batch, h_active_batch, batch_size * sizeof(int), cudaMemcpyHostToDevice));
            
            int checkBlocks = (batch_size + blockSize - 1) / blockSize;
            check_nations_kernel<<<checkBlocks, blockSize>>>(
                batch_size, d_active_batch, d_offsets, d_flat, d_tree, d_targets, d_results
            );
        }

        // E. Read Results & Update State
        CHECK(cudaMemcpy(h_results.data(), d_results, n * sizeof(int), cudaMemcpyDeviceToHost));

        // Update L/R based on the schedule we just ran
        for (const auto& item : schedule) {
            int id = item.id;
            int mid = item.mid;
            
            if (h_results[id] == 1) {
                r[id] = mid;
            } else {
                l[id] = mid + 1;
            }
        }
    }

    // --- Output ---
    for (int i = 0; i < n; ++i) {
        if (l[i] >= k) cout << "NIE\n";
        else cout << l[i] + 1 << "\n";
    }

    // Free
    cudaFreeHost(h_active_batch);
    cudaFree(d_flat); cudaFree(d_offsets); cudaFree(d_targets);
    cudaFree(d_tree); cudaFree(d_up_l); cudaFree(d_up_r); cudaFree(d_up_v);
    cudaFree(d_active_batch); cudaFree(d_results);

    return 0;
}