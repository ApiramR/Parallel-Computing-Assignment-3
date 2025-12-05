// pbs_cuda_optimized.cu
// Optimized GPU pipeline for iterative PBS + range updates.
// - For each bucket/time t we:
//   1) apply pending updates to D (2 atomics per update, O(#updates))
//   2) inclusive-scan D -> arr (thrust device scan; O(m))
//   3) kernel: for each nation in bucket, sum arr[sectors] (parallel over nations)
// This avoids heavy Fenwick atomic contention.

#include <vector>
#include <iostream>
#include <algorithm>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>

using namespace std;
using ull = unsigned long long;
using i64 = long long;

#define CHECK(call) { \
  cudaError_t e = call; if(e != cudaSuccess) { \
    fprintf(stderr,"CUDA err %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} }

struct Update { int l, r; long long v; };

__global__ void scatter_updates_kernel(
    const int start_u, const int end_u,
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

    // Note: D is sized m_plus1+1 (we will use indices 1..m_plus1)
    if (L <= R) {
        // add at L+1, subtract at R+2
        int a = L + 1;
        int b = R + 2;
        if (a <= m_plus1) atomicAdd(&d_D[a], add);
        if (b <= m_plus1) atomicAdd(&d_D[b], sub);
    } else {
        // wrap-around: [1..R] and [L..m]
        // add at 1, subtract at R+2, add at L+1
        if (1 <= m_plus1) atomicAdd(&d_D[1], add);
        int b = R + 2;
        if (b <= m_plus1) atomicAdd(&d_D[b], sub);
        int c = L + 1;
        if (c <= m_plus1) atomicAdd(&d_D[c], add);
    }
}

// For each nation in 'active' (length count) compute sum of arr at its sectors.
// d_active is an array of nation ids of length count.
__global__ void check_bucket_kernel(
    int count,
    const int* d_active,         // list of nation ids in this bucket
    const int* d_offsets,        // offsets into d_flat (size n+1)
    const int* d_flat,           // flattened sectors (1-based positions)
    const ull* d_arr,            // arr[1..m_plus1] - unsigned sums per position
    const long long* d_targets,  // global targets
    int* d_result_flags          // result flags, length = count (0/1)
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;

    int nation = d_active[tid];
    int s = d_offsets[nation];
    int e = d_offsets[nation + 1];

    long long sum = 0;
    for (int i = s; i < e; ++i) {
        int pos = d_flat[i]; // 1-based
        // read d_arr[pos] as unsigned and cast to signed
        long long val = (long long) d_arr[pos];
        sum += val;
        if (sum >= d_targets[nation]) break;
    }
    d_result_flags[tid] = (sum >= d_targets[nation]) ? 1 : 0;
}

int main(int argc, char** argv) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m;
    if (!(cin >> n >> m)) return 0;

    vector<vector<int>> arr(n);
    for (int i = 0; i < m; ++i) {
        int x; cin >> x;
        arr[x - 1].push_back(i + 1); // store 1-based indices for easier use
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

    // Flatten sectors
    vector<int> flat; flat.reserve(m);
    vector<int> offsets(n+1); offsets[0] = 0;
    for (int i = 0; i < n; ++i) {
        flat.insert(flat.end(), arr[i].begin(), arr[i].end());
        offsets[i+1] = (int)flat.size();
    }

    // Device buffers (static)
    int *d_flat = nullptr, *d_offsets = nullptr;
    int *d_up_l = nullptr, *d_up_r = nullptr;
    long long *d_up_v = nullptr;
    long long *d_targets = nullptr;

    // D and arr arrays on device: size m_plus1+1 (use indices 0..m_plus1)
    int m_plus1 = m + 1;
    thrust::device_vector<ull> d_D(m_plus1 + 1);   // D[0..m_plus1], we'll use 1..m_plus1
    thrust::device_vector<ull> d_arr(m_plus1 + 1); // arr after inclusive scan

    CHECK(cudaMalloc(&d_flat, flat.size() * sizeof(int)));
    CHECK(cudaMalloc(&d_offsets, (n+1) * sizeof(int)));
    CHECK(cudaMalloc(&d_up_l, k * sizeof(int)));
    CHECK(cudaMalloc(&d_up_r, k * sizeof(int)));
    CHECK(cudaMalloc(&d_up_v, k * sizeof(long long)));
    CHECK(cudaMalloc(&d_targets, n * sizeof(long long)));

    CHECK(cudaMemcpy(d_flat, flat.data(), flat.size() * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_offsets, offsets.data(), (n+1) * sizeof(int), cudaMemcpyHostToDevice));
    if (k) {
        CHECK(cudaMemcpy(d_up_l, h_up_l.data(), k * sizeof(int), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_up_r, h_up_r.data(), k * sizeof(int), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_up_v, h_up_v.data(), k * sizeof(long long), cudaMemcpyHostToDevice));
    }
    CHECK(cudaMemcpy(d_targets, targets.data(), n * sizeof(long long), cudaMemcpyHostToDevice));

    // Buffers for active batch & result flags
    int *d_active = nullptr;
    int *d_result_flags = nullptr;
    CHECK(cudaMalloc(&d_active, n * sizeof(int)));
    CHECK(cudaMalloc(&d_result_flags, n * sizeof(int)));

    // Host schedule arrays
    vector<int> l(n,0), r(n,k), ans(n,-1);
    vector<vector<int>> need(k+1);
    for (int i = 0; i < n; ++i) {
        int mid = (l[i] + r[i]) >> 1;
        if (l[i] >= r[i]) ans[i] = l[i];
        else need[mid].push_back(i);
    }

    // Prepare schedule vector for sorting times
    vector<pair<int,int>> schedule; // (mid, id)

    int block = atoi(argv[1]);
    cout<<block<<'\n';
    // main loop (repeat until no moves) - similar to serial
    bool changed = true;
    while (changed) {
        changed = false;

        // Build schedule: list of (mid, id)
        schedule.clear();
        schedule.reserve(n);
        for (int i = 0; i < n; ++i) {
            if (l[i] < r[i]) {
                int mid = (l[i] + r[i]) >> 1;
                schedule.emplace_back(mid, i);
            }
        }
        if (schedule.empty()) break;

        sort(schedule.begin(), schedule.end()); // sort by mid/time

        // Reset D on device to zeros
        thrust::fill(d_D.begin(), d_D.end(), (ull)0);

        int idx = 0;
        int sched_sz = (int)schedule.size();
        int current_time = -1;

        // Sweep through schedule by target_time groups
        while (idx < sched_sz) {
            int target_time = schedule[idx].first;

            // Apply updates from (current_time+1)..target_time inclusive
            if (target_time > current_time) {
                int start_u = current_time + 1;
                int end_u = target_time;
                if (start_u <= end_u && start_u < k) {
                    if (end_u >= k) end_u = k - 1;
                    int count_u = end_u - start_u + 1;
                    int blocks = (count_u + block - 1) / block;
                    scatter_updates_kernel<<<blocks, block>>>(
                        start_u, end_u, d_up_l, d_up_r, d_up_v,
                        thrust::raw_pointer_cast(d_D.data()), m_plus1
                    );
                    CHECK(cudaGetLastError());
                    CHECK(cudaDeviceSynchronize());
                }
                current_time = target_time;
            }

            // Build batch of nations that have this same mid
            int batch_start = idx;
            while (idx < sched_sz && schedule[idx].first == target_time) ++idx;
            int batch_size = idx - batch_start;

            // If no updates were applied since last scan (unlikely), we still need arr; but we can reuse arr.
            // Compute arr = inclusive_scan(D) into d_arr
            // We only need indices 1..m_plus1
            // using thrust inclusive_scan on device
            thrust::inclusive_scan(d_D.begin(), d_D.begin() + (m_plus1 + 1), d_arr.begin());

            // Prepare active list on host then copy to d_active
            // (we could stream-copy from schedule if large)
            vector<int> h_active(batch_size);
            for (int t = 0; t < batch_size; ++t) h_active[t] = schedule[batch_start + t].second;
            CHECK(cudaMemcpy(d_active, h_active.data(), batch_size * sizeof(int), cudaMemcpyHostToDevice));

            // Launch kernel to evaluate batch
            int grid = (batch_size + block - 1) / block;
            check_bucket_kernel<<<grid, block>>>(
                batch_size, d_active, d_offsets, d_flat,
                thrust::raw_pointer_cast(d_arr.data()),
                d_targets, d_result_flags
            );
            CHECK(cudaGetLastError());
            CHECK(cudaDeviceSynchronize());

            // Copy back result flags for this batch only
            vector<int> h_flags(batch_size);
            CHECK(cudaMemcpy(h_flags.data(), d_result_flags, batch_size * sizeof(int), cudaMemcpyDeviceToHost));

            // Update l/r for each nation in the batch
            for (int t = 0; t < batch_size; ++t) {
                int nation = h_active[t];
                int mid = target_time;
                if (h_flags[t]) {
                    r[nation] = mid;
                } else {
                    l[nation] = mid + 1;
                }
                if (l[nation] >= r[nation]) ans[nation] = l[nation];
                else changed = true;
            }
        } // end sweep
    } // end while changed

    // output
    for (int i = 0; i < n; ++i) {
        int out = ans[i];
        out++;
        if (out > k) cout << "NIE\n";
        else cout << out << '\n';
    }

    // free
    cudaFree(d_flat); cudaFree(d_offsets); cudaFree(d_up_l); cudaFree(d_up_r);
    cudaFree(d_up_v); cudaFree(d_targets); cudaFree(d_active); cudaFree(d_result_flags);

    return 0;
}
