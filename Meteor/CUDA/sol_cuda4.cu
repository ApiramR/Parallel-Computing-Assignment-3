// pbs_cuda_fastest.cu
// Fastest Parallel Binary Search using difference-array + parallel inclusive scan (Thrust).
// - Builds D with 2 atomics per update on device
// - Uses thrust::inclusive_scan to compute arr (prefix sums)
// - Evaluates each bucket in parallel with a lightweight kernel

#include <iostream>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>

using namespace std;
using ull = unsigned long long;
using i64 = long long;

#define CHECK(call) {                                   \
  cudaError_t e = (call);                               \
  if (e != cudaSuccess) {                               \
    fprintf(stderr, "CUDA Error %s:%d: %s\n",           \
      __FILE__, __LINE__, cudaGetErrorString(e));       \
    exit(1);                                             \
  }                                                     \
}

// Host-side update representation
struct Update { int l, r; long long v; };

// Scatter updates into D (device) using atomicAdd on ull (2 or 3 writes per update)
// start_u..end_u inclusive are indices in the updates array (0-based)
__global__ void scatter_updates_kernel(
    int start_u, int end_u,
    const int* d_up_l, const int* d_up_r, const long long* d_up_v,
    ull* d_D, int m_plus1
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int u = start_u + tid;
    if (u > end_u) return;

    int L = d_up_l[u];
    int R = d_up_r[u];
    long long vv = d_up_v[u];
    ull add = (ull) vv;
    ull sub = (ull) (-vv);

    // Use indices 1..m_plus1 in D/arr
    if (L <= R) {
        int a = L + 1;       // add at a
        int b = R + 2;       // subtract at b
        if (a <= m_plus1) atomicAdd(&d_D[a], add);
        if (b <= m_plus1) atomicAdd(&d_D[b], sub);
    } else {
        // wrap: add at 1, sub at R+2, add at L+1
        if (1 <= m_plus1) atomicAdd(&d_D[1], add);
        int b = R + 2;
        if (b <= m_plus1) atomicAdd(&d_D[b], sub);
        int c = L + 1;
        if (c <= m_plus1) atomicAdd(&d_D[c], add);
    }
}

// For each nation in this batch (d_active[0..count-1]), compute sum over its sectors using arr (d_arr).
// d_offsets gives offsets into d_flat (size n+1), and d_flat contains 1-based sector positions.
__global__ void check_bucket_kernel(
    int count,
    const int* d_active,
    const int* d_offsets,
    const int* d_flat,
    const ull* d_arr,           // arr[0..m_plus1], we use indices 1..m_plus1
    const long long* d_targets,
    int* d_flags                // output 0/1 per thread (count)
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;

    int nation = d_active[tid];
    int s = d_offsets[nation];
    int e = d_offsets[nation + 1];

    long long sum = 0;
    for (int i = s; i < e; ++i) {
        int pos = d_flat[i]; // 1-based
        long long val = (long long) d_arr[pos]; // reinterpret ull -> signed
        sum += val;
        if (sum >= d_targets[nation]) break;
    }
    d_flags[tid] = (sum >= d_targets[nation]) ? 1 : 0;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m;
    if (!(cin >> n >> m)) return 0;

    vector<vector<int>> owners(n);
    for (int i = 0; i < m; ++i) {
        int x; cin >> x;
        if (x < 1 || x > n) { /* ignore invalid */ }
        else owners[x - 1].push_back(i + 1); // store 1-based sector index
    }

    vector<long long> targets(n);
    for (int i = 0; i < n; ++i) cin >> targets[i];

    int k; cin >> k;
    vector<Update> updates(k);
    vector<int> h_up_l(k), h_up_r(k);
    vector<long long> h_up_v(k);
    for (int i = 0; i < k; ++i) {
        int L, R; long long V; cin >> L >> R >> V;
        updates[i].l = L - 1;
        updates[i].r = R - 1;
        updates[i].v = V;
        h_up_l[i] = updates[i].l;
        h_up_r[i] = updates[i].r;
        h_up_v[i] = updates[i].v;
    }

    // Flatten sectors & offsets
    vector<int> flat; flat.reserve(m);
    vector<int> offsets(n + 1); offsets[0] = 0;
    for (int i = 0; i < n; ++i) {
        flat.insert(flat.end(), owners[i].begin(), owners[i].end());
        offsets[i + 1] = (int)flat.size();
    }

    // Device allocations
    int *d_flat = nullptr, *d_offsets = nullptr;
    int *d_up_l = nullptr, *d_up_r = nullptr;
    long long *d_up_v = nullptr, *d_targets = nullptr;

    CHECK(cudaMalloc(&d_flat, max(1, (int)flat.size()) * sizeof(int)));
    CHECK(cudaMalloc(&d_offsets, (n + 1) * sizeof(int)));
    if (k > 0) {
        CHECK(cudaMalloc(&d_up_l, k * sizeof(int)));
        CHECK(cudaMalloc(&d_up_r, k * sizeof(int)));
        CHECK(cudaMalloc(&d_up_v, k * sizeof(long long)));
    } else {
        // allocate tiny to avoid nullptr in kernels (won't be used)
        CHECK(cudaMalloc(&d_up_l, 1 * sizeof(int)));
        CHECK(cudaMalloc(&d_up_r, 1 * sizeof(int)));
        CHECK(cudaMalloc(&d_up_v, 1 * sizeof(long long)));
    }
    CHECK(cudaMalloc(&d_targets, n * sizeof(long long)));

    CHECK(cudaMemcpy(d_flat, flat.data(), flat.size() * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_offsets, offsets.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    if (k > 0) {
        CHECK(cudaMemcpy(d_up_l, h_up_l.data(), k * sizeof(int), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_up_r, h_up_r.data(), k * sizeof(int), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_up_v, h_up_v.data(), k * sizeof(long long), cudaMemcpyHostToDevice));
    }
    CHECK(cudaMemcpy(d_targets, targets.data(), n * sizeof(long long), cudaMemcpyHostToDevice));

    // Device difference array D and arr (use thrust DeviceVector)
    int m_plus1 = m + 1; // we'll use indices 1..m_plus1
    thrust::device_vector<ull> d_D(m_plus1 + 1);   // size m_plus1+1, indices 0..m_plus1
    thrust::device_vector<ull> d_arr(m_plus1 + 1);

    // Device space for active list and flags (max n)
    int *d_active = nullptr;
    int *d_flags = nullptr;
    CHECK(cudaMalloc(&d_active, max(1, n) * sizeof(int)));
    CHECK(cudaMalloc(&d_flags, max(1, n) * sizeof(int)));

    // Host pinned buffer for active batch (faster H2D)
    int *h_active = nullptr;
    CHECK(cudaMallocHost(&h_active, max(1, n) * sizeof(int)));

    // Binary search state
    vector<int> l(n, 0), rvec(n, k), ans(n, -1);
    for (int i = 0; i < n; ++i) {
        if (l[i] >= rvec[i]) ans[i] = l[i];
    }

    // Main iterative bucketed PBS loop
    bool changed = true;
    const int BLOCK = 256;
    while (changed) {
        changed = false;

        // Build schedule vector (mid, id)
        vector<pair<int,int>> schedule;
        schedule.reserve(n);
        for (int i = 0; i < n; ++i) {
            if (l[i] < rvec[i]) {
                int mid = (l[i] + rvec[i]) >> 1;
                schedule.emplace_back(mid, i);
            }
        }
        if (schedule.empty()) break;

        sort(schedule.begin(), schedule.end()); // host std::sort

        // Reset D (device) to zero
        thrust::fill(d_D.begin(), d_D.end(), (ull)0);

        // Sweep schedule grouped by mid (time)
        int idx = 0;
        int SZ = (int)schedule.size();
        int current_time = -1;

        while (idx < SZ) {
            int target_time = schedule[idx].first;

            // Apply updates from (current_time+1) .. target_time (inclusive)
            if (target_time > current_time) {
                int start_u = current_time + 1;
                int end_u = target_time;
                if (start_u < 0) start_u = 0;
                if (start_u < k) {
                    if (end_u >= k) end_u = k - 1;
                    if (start_u <= end_u) {
                        int count_u = end_u - start_u + 1;
                        int blocks = (count_u + BLOCK - 1) / BLOCK;
                        // scatter updates into D
                        scatter_updates_kernel<<<blocks, BLOCK>>>(
                            start_u, end_u, d_up_l, d_up_r, d_up_v,
                            thrust::raw_pointer_cast(d_D.data()), m_plus1
                        );
                        CHECK(cudaGetLastError());
                        CHECK(cudaDeviceSynchronize());
                    }
                }
                current_time = target_time;
            }

            // Build batch of nations with this same mid
            int batch_start = idx;
            while (idx < SZ && schedule[idx].first == target_time) ++idx;
            int batch_size = idx - batch_start;

            // Compute arr = inclusive_scan(D) -> d_arr
            // We need indices 0..m_plus1 inclusive; arr[pos] will be prefix(D)[pos]
            thrust::inclusive_scan(d_D.begin(), d_D.begin() + (m_plus1 + 1), d_arr.begin());

            // Prepare active list on host (IDs)
            for (int t = 0; t < batch_size; ++t) {
                h_active[t] = schedule[batch_start + t].second;
            }
            // Copy active list to device (fast since pinned)
            CHECK(cudaMemcpy(d_active, h_active, batch_size * sizeof(int), cudaMemcpyHostToDevice));

            // Launch kernel to check this batch
            int g = (batch_size + BLOCK - 1) / BLOCK;
            check_bucket_kernel<<<g, BLOCK>>>(
                batch_size, d_active, d_offsets, d_flat,
                thrust::raw_pointer_cast(d_arr.data()),
                d_targets, d_flags
            );
            CHECK(cudaGetLastError());
            CHECK(cudaDeviceSynchronize());

            // Copy flags back
            vector<int> h_flags(batch_size);
            CHECK(cudaMemcpy(h_flags.data(), d_flags, batch_size * sizeof(int), cudaMemcpyDeviceToHost));

            // Update binary search bounds for these nations
            for (int t = 0; t < batch_size; ++t) {
                int nation = h_active[t];
                int mid = target_time;
                if (h_flags[t]) {
                    rvec[nation] = mid;
                } else {
                    l[nation] = mid + 1;
                }
                if (l[nation] >= rvec[nation]) ans[nation] = l[nation];
                else changed = true;
            }
        } // end sweep
    } // end while changed

    // Output answers (1-based) matching original format
    for (int i = 0; i < n; ++i) {
        int out = ans[i];
        out++;
        if (out > k) cout << "NIE\n";
        else cout << out << '\n';
    }

    // Cleanup
    cudaFree(d_flat);
    cudaFree(d_offsets);
    cudaFree(d_up_l);
    cudaFree(d_up_r);
    cudaFree(d_up_v);
    cudaFree(d_targets);
    cudaFree(d_active);
    cudaFree(d_flags);
    cudaFreeHost(h_active);

    return 0;
}
