#include <iostream>
#include <vector>
#include <algorithm>
#include <omp.h>

using namespace std;

struct Update {
    int l, r;
    long long v;
};

const int MAXN = 900005;
long long tree[MAXN];
int m;
int ans[MAXN];

vector<vector<int>> member_sectors;
vector<long long> targets;
vector<Update> updates;

// --- Inline Fenwick Tree ---
inline void add(int x, long long val) {
    for (; x <= m; x += x & -x) tree[x] += val;
}

inline long long query(int x) {
    long long sum = 0;
    for (; x > 0; x -= x & -x) sum += tree[x];
    return sum;
}

// Helper to apply/remove range updates
inline void apply_update(int i, int type) {
    // type 1 = add, type -1 = remove
    long long v = updates[i].v * type;
    if (updates[i].l <= updates[i].r) {
        add(updates[i].l, v);
        add(updates[i].r + 1, -v);
    } else {
        add(1, v);
        add(updates[i].r + 1, -v);
        add(updates[i].l, v);
    }
}

//  Recursive Parallel Solver 
void solve(int L, int R, vector<int>& nations) {
    if (nations.empty()) return;

    if (L == R) {
        for (int id : nations) ans[id] = L;
        return;
    }

    int mid = (L + R) >> 1;

    // 1. Apply Updates [L, mid] (Serial)
    // We do this serially because the BIT is shared. 
    // This is fast because we batch many updates.
    for (int i = L; i <= mid; ++i) {
        apply_update(i, 1);
    }

    // Temporary storage for parallel results
    // We cannot push_back to vectors in parallel efficiently, so we use arrays
    // type[i] stores: 0 = go left (satisfied), 1 = go right (not satisfied)
    vector<int> type(nations.size());
    vector<long long> collected_sums(nations.size());

    // 2. Check Nations (Parallel - The Heavy Lifting)
    #pragma omp parallel for schedule(dynamic, 64)
    for (int i = 0; i < nations.size(); ++i) {
        int id = nations[i];
        long long sum = 0;
        
        // Sum up sectors
        for (int sector : member_sectors[id]) {
            sum += query(sector);
            if (sum >= targets[id]) break; // Optimization
        }
        
        collected_sums[i] = sum;
        
        if (sum >= targets[id]) {
            type[i] = 0; // Met target -> Go Left (Try earlier time)
        } else {
            type[i] = 1; // Failed -> Go Right (Need more time)
        }
    }

    // 3. Rollback Updates (Serial)
    // We must restore the BIT to 0 state before recursing right
    for (int i = L; i <= mid; ++i) {
        apply_update(i, -1);
    }

    // 4. Partition Nations (Serial)
    vector<int> left_nations, right_nations;
    left_nations.reserve(nations.size() / 2);
    right_nations.reserve(nations.size() / 2);

    for (int i = 0; i < nations.size(); ++i) {
        if (type[i] == 0) {
            left_nations.push_back(nations[i]);
        } else {
            // CRITICAL: If going right, we subtract the current sum from target!
            // This effectively "saves" the progress from [L, mid] so we don't recalculate it.
            targets[nations[i]] -= collected_sums[i];
            right_nations.push_back(nations[i]);
        }
    }

    // Free memory
    vector<int>().swap(nations);

    // 5. Recurse
    solve(L, mid, left_nations);
    solve(mid + 1, R, right_nations);
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);


    int n;
    if (!(cin >> n >> m)) return 0;

    member_sectors.resize(n);
    for (int i = 0; i < m; ++i) {
        int x; cin >> x;
        member_sectors[x - 1].push_back(i + 1);
    }

    targets.resize(n);
    for (int i = 0; i < n; ++i) cin >> targets[i];

    int k; cin >> k;
    updates.resize(k);
    for (int i = 0; i < k; ++i) {
        cin >> updates[i].l >> updates[i].r >> updates[i].v;
        // Keep 1-based logic for BIT consistency inside apply_update
    }

    // Initialize list of all nations
    vector<int> all_nations(n);
    for (int i = 0; i < n; ++i) all_nations[i] = i;

    // Start Recursion
    // Range is [0, k]. (k means "NIE")
    // We treat 'k' as a valid index for the recursion base case logic
    // but updates only exist 0 to k-1.
    solve(0, k, all_nations);

    // Output
    for (int i = 0; i < n; ++i) {
        if (ans[i] >= k) cout << "NIE\n";
        else cout << ans[i] + 1 << "\n";
    }

    return 0;
}