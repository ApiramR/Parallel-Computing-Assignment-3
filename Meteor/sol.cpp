#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

struct Update {
    int l, r;
    long long v;
};


const int MAXN = 300005;
long long tree[MAXN];
int m;
int ans[MAXN];


vector<vector<int>> member_sectors;
vector<long long> targets;
vector<Update> updates;

inline void add(int x, long long val) {
    for (; x <= m; x += x & -x) tree[x] += val;
}

inline long long query(int x) {
    long long sum = 0;
    for (; x > 0; x -= x & -x) sum += tree[x];
    return sum;
}

inline void apply_update(int i, int type) {

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

void solve(int L, int R, vector<int>& nations) {
    if (nations.empty()) return;

    if (L == R) {
        for (int id : nations) ans[id] = L;
        return;
    }

    int mid = (L + R) >> 1;


    for (int i = L; i <= mid; ++i) {
        apply_update(i, 1);
    }

    vector<int> type(nations.size());
    vector<long long> collected_sums(nations.size());

    for (int i = 0; i < nations.size(); ++i) {
        int id = nations[i];
        long long sum = 0;
        

        for (int sector : member_sectors[id]) {
            sum += query(sector);
            if (sum >= targets[id]) break; 
        }
        
        collected_sums[i] = sum;
        
        if (sum >= targets[id]) {
            type[i] = 0; 
        } else {
            type[i] = 1;
        }
    }

    for (int i = L; i <= mid; ++i) {
        apply_update(i, -1);
    }

    vector<int> left_nations, right_nations;
    left_nations.reserve(nations.size() / 2);
    right_nations.reserve(nations.size() / 2);

    for (int i = 0; i < nations.size(); ++i) {
        if (type[i] == 0) {
            left_nations.push_back(nations[i]);
        } else {
            targets[nations[i]] -= collected_sums[i];
            right_nations.push_back(nations[i]);
        }
    }

    vector<int>().swap(nations);


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
    }

    vector<int> all_nations(n);
    for (int i = 0; i < n; ++i) all_nations[i] = i;


    solve(0, k, all_nations);

    for (int i = 0; i < n; ++i) {
        if (ans[i] >= k) cout << "NIE\n";
        else cout << ans[i] + 1 << "\n";
    }

    return 0;
}