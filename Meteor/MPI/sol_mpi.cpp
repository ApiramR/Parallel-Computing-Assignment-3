#include <mpi.h>
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

// --- Data Structures ---
struct Update {
    int l, r;
    long long v;
};

// --- Global Data (Replicated) ---
const int MAXN = 3000005;
long long tree[MAXN];
int m;
int local_ans[MAXN];

// Input Data
vector<vector<int>> member_sectors;
vector<long long> targets;
vector<Update> updates;

// --- Inline BIT ---
inline void add(int x, long long val) {
    for (; x <= m; x += x & -x) tree[x] += val;
}

inline long long query(int x) {
    long long s = 0;
    while (x > 0) {
        s += tree[x];
        x -= (x & -x);
    }
    return s;
}

inline void apply_update(int i, int type) {
    long long v = updates[i].v * type;
    if (updates[i].l <= updates[i].r) {
        add(updates[i].l + 1, v);
        add(updates[i].r + 2, -v);
    } else {
        add(1, v);
        add(updates[i].r + 2, -v);
        add(updates[i].l + 1, v);
    }
}

// --- Optimized Solver (No Vector Allocations) ---
// We pass iterators (begin/end) instead of vectors
void solve(int L, int R, vector<int>::iterator begin, vector<int>::iterator end) {
    if (begin == end) return;

    if (L == R) {
        for (auto it = begin; it != end; ++it) local_ans[*it] = L;
        return;
    }

    int mid = (L + R) >> 1;

    // 1. Apply Updates (Serial Logic, Replicated on all Ranks)
    for (int i = L; i <= mid; ++i) apply_update(i, 1);

    // 2. Check & Partition In-Place (Like Quicksort)
    // We move "Left-going" nations to the front of the range
    auto split_point = std::stable_partition(begin, end, [&](int id) {
        long long sum = 0;
        for (int sector : member_sectors[id]) {
            sum += query(sector);
            if (sum >= targets[id]) break;
        }

        if (sum >= targets[id]) {
            return true; // Go Left (Front)
        } else {
            targets[id] -= sum; // Adjust target for Right branch
            return false; // Go Right (Back)
        }
    });

    // 3. Rollback Updates
    for (int i = L; i <= mid; ++i) apply_update(i, -1);

    // 4. Recurse using the split point
    solve(L, mid, begin, split_point);
    solve(mid + 1, R, split_point, end);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n, k;

    // --- INPUT (Rank 0) ---
    if (rank == 0) {
        ios_base::sync_with_stdio(false);
        cin.tie(NULL);
        if (cin >> n >> m) {
            member_sectors.resize(n);
            for (int i = 0; i < m; ++i) {
                int x; cin >> x;
                member_sectors[x - 1].push_back(i + 1);
            }
            targets.resize(n);
            for (int i = 0; i < n; ++i) cin >> targets[i];
            cin >> k;
            updates.resize(k);
            for (int i = 0; i < k; ++i) {
                cin >> updates[i].l >> updates[i].r >> updates[i].v;
                updates[i].l--; updates[i].r--;
            }
        }
    }

    // --- BROADCAST METADATA ---
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (n == 0) { MPI_Finalize(); return 0; }
    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&k, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        member_sectors.resize(n);
        targets.resize(n);
        updates.resize(k);
    }

    // --- BROADCAST DATA (Flattened) ---
    vector<int> flat_sectors;
    vector<int> offsets(n + 1);

    if (rank == 0) {
        offsets[0] = 0;
        for (int i = 0; i < n; ++i) {
            flat_sectors.insert(flat_sectors.end(), member_sectors[i].begin(), member_sectors[i].end());
            offsets[i + 1] = flat_sectors.size();
        }
    }

    MPI_Bcast(offsets.data(), n + 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    int total_sectors = offsets[n];
    MPI_Bcast(&total_sectors, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (rank != 0) flat_sectors.resize(total_sectors);
    MPI_Bcast(flat_sectors.data(), total_sectors, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        for (int i = 0; i < n; ++i) {
            // Reconstruct logic: Insert range from flat to vector
            member_sectors[i].insert(member_sectors[i].end(), 
                flat_sectors.begin() + offsets[i], 
                flat_sectors.begin() + offsets[i+1]);
        }
    }

    MPI_Bcast(targets.data(), n, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(updates.data(), k * sizeof(Update), MPI_BYTE, 0, MPI_COMM_WORLD);

    // --- DISTRIBUTE WORK ---
    fill(local_ans, local_ans + n, -1);
    
    vector<int> my_nations;
    // Simple block distribution
    int chunk = (n + size - 1) / size;
    int start = rank * chunk;
    int end = min(n, (rank + 1) * chunk);
    
    my_nations.reserve(chunk);
    for(int i = start; i < end; ++i) my_nations.push_back(i);

    // --- SOLVE ---
    // Pass iterators to avoid copying vectors
    solve(0, k, my_nations.begin(), my_nations.end());

    // --- GATHER ---
    vector<int> final_ans;
    if (rank == 0) final_ans.resize(n);

    // Max reduction combines the local answers (-1 vs Valid Answer)
    MPI_Reduce(local_ans, final_ans.data(), n, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        for (int i = 0; i < n; ++i) {
            if (final_ans[i] >= k) cout << "NIE\n";
            else cout << final_ans[i] + 1 << "\n";
        }
    }

    MPI_Finalize();
    return 0;
}