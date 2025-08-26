# 数据结构

## 并查集

```cpp
struct DSU {
    vector<int> f, siz;
    // vector<int> dis; // 带权并查集

    DSU() {}
    DSU(int n) {
        init(n);
    }

    void init(int n) {
        f.resize(n);
        std::iota(f.begin(), f.end(), 0);
        siz.assign(n, 1);
        // dis.assign(n, 0);
    }

    int find(int x) {
        while (x != f[x]) {
            x = f[x] = f[f[x]];
        }
        return x;
    }
    // int find(int x) {
    //     if (x == f[x])
    //         return x;
    //     int r = find(f[x]);
    //     dis[x] += dis[f[x]];
    //     return f[x] = r;
    // }

    bool same(int x, int y) {
        return find(x) == find(y);
    }

    bool merge(int x, int y) {
        x = find(x);
        y = find(y);
        if (x == y) return false;
        f[y] = x;
        // dis[y] = siz[x];
        siz[x] += siz[y];
        return true;
    }

    int size(int x) {
        return siz[find(x)];
    }
};
```

## 线段树

### SegmentTree

- 数组 `0-based`；查询区间**左闭右开** $[l,r)$
- `Info` 类维护信息，重载 `operator +`

```cpp
template <class Info>
struct SegmentTree {
    int n;
    vector<Info> info;
    SegmentTree() : n(0) {}
    SegmentTree(int _n, Info _v = Info()) {
        init(_n, _v);
    }
    template <class T>
    SegmentTree(vector<T> _init) {
        init(_init);
    }
    void init(int _n, Info _v = Info()) {
        init(vector(_n, _v));
    }
    template <class T>
    void init(vector<T> _init) {
        n = _init.size();
        info.assign(4 << std::__lg(n), Info());
        auto build = [&](auto &&self, int p, int l, int r) -> void {
            if (r - l == 1) {
                info[p] = _init[l];
                return;
            }
            int m = (l + r) >> 1;
            self(self, p << 1, l, m);
            self(self, p << 1 | 1, m, r);
            pull(p);
        };
        build(build, 1, 0, n);
    }

    void pull(int p) {
        info[p] = info[p << 1] + info[p << 1 | 1];
    }

    void update(int p, int l, int r, int x, const Info &v) {
        if (r - l == 1) {
            info[p] = v;
            return;
        }
        int m = (l + r) >> 1;
        if (x < m) {
            update(p << 1, l, m, x, v);
        } else {
            update(p << 1 | 1, m, r, x, v);
        }
        pull(p);
    }
    void update(int x, const Info &v) {
        update(1, 0, n, x, v);
    }

    Info rangeQuery(int p, int l, int r, int x, int y) {
        if (x >= r || y <= l) return Info();
        if (x <= l && r <= y) return info[p];
        int m = (l + r) >> 1;
        return rangeQuery(p << 1, l, m, x, y) + rangeQuery(p << 1 | 1, m, r, x, y);
    }
    Info rangeQuery(int x, int y) {
        return rangeQuery(1, 0, n, x, y);
    }

    template <class F>
    int findFirst(int p, int l, int r, int x, int y, F &&pred) {
        if (x >= r || y <= l) return -1;
        if (x <= l && r <= y && !pred(info[p])) return -1;
        if (r - l == 1) return l;
        int m = (l + r) >> 1;
        int res = findFirst(p << 1, l, m, x, y, pred);
        if (res != -1) return res;
        return findFirst(p << 1 | 1, m, r, x, y, pred);
    }
    template <class F>
    int findFirst(int x, int y, F &&pred) {
        return findFirst(1, 0, n, x, y, pred);
    }

    template <class F>
    int findLast(int p, int l, int r, int x, int y, F &&pred) {
        if (x >= r || y <= l) return -1;
        if (x <= l && r <= y && !pred(info[p])) return -1;
        if (r - l == 1) return l;
        int m = (l + r) >> 1;
        int res = findLast(p << 1 | 1, m, r, x, y, pred);
        if (res != -1) return res;
        return findLast(p << 1, l, m, x, y, pred);
    }
    template <class F>
    int findLast(int x, int y, F &&pred) {
        return findLast(1, 0, n, x, y, pred);
    }
};

struct Info {
    int x = 0;
    Info() = default;
    Info(int x_) : x(x_) {}
    Info operator+(const Info &other) const {
        return {x + other.x};
    }
};

// input: vector<T> a
// SegmentTree<Info> seg(a)
```

### LazySegmentTree

- `Info` 类维护信息，`Tag` 类维护标记，都需要实现初始化和 `apply` 函数；

```cpp
template <class Info, class Tag>
struct LazySegmentTree {
    int n;
    vector<Info> info;
    vector<Tag> tag;
    LazySegmentTree() : n(0) {}
    LazySegmentTree(int _n, Info _v = Info()) {
        init(_n, _v);
    }
    template <class T>
    LazySegmentTree(vector<T> _init) {
        init(_init);
    }
    void init(int _n, Info _v = Info()) {
        init(vector(_n, _v));
    }
    template <class T>
    void init(vector<T> _init) {
        n = _init.size();
        info.assign(4 << std::__lg(n), Info());
        tag.assign(4 << std::__lg(n), Tag());
        auto build = [&](auto &&self, int p, int l, int r) -> void {
            if (r - l == 1) {
                info[p] = _init[l];
                return;
            }
            int m = (l + r) >> 1;
            self(self, p << 1, l, m);
            self(self, p << 1 | 1, m, r);
            pull(p);
        };
        build(build, 1, 0, n);
    }

    void pull(int p) {
        info[p] = info[p << 1] + info[p << 1 | 1];
    }

    void apply(int p, const Tag &v, int len) {
        info[p].apply(v, len);
        tag[p].apply(v);
    }

    void push(int p, int len) {
        apply(p << 1, tag[p], len >> 1);
        apply(p << 1 | 1, tag[p], len - len / 2);
        tag[p] = Tag();  // clear the tag after pushing
    }

    void update(int p, int l, int r, int x, const Info &v) {
        if (r - l == 1) {
            info[p] = v;
            return;
        }
        int m = (l + r) >> 1;
        push(p, r - l);
        if (x < m) {
            update(p << 1, l, m, x, v);
        } else {
            update(p << 1 | 1, m, r, x, v);
        }
        pull(p);
    }
    void update(int x, const Info &v) {
        update(1, 0, n, x, v);
    }

    Info rangeQuery(int p, int l, int r, int x, int y) {
        if (x >= r || y <= l) return Info();
        if (x <= l && r <= y) return info[p];
        int m = (l + r) >> 1;
        push(p, r - l);
        return rangeQuery(p << 1, l, m, x, y) + rangeQuery(p << 1 | 1, m, r, x, y);
    }
    Info rangeQuery(int x, int y) {
        return rangeQuery(1, 0, n, x, y);
    }

    void rangeApply(int p, int l, int r, int x, int y, const Tag &v) {
        if (x >= r || y <= l) return;
        if (x <= l && r <= y) {
            apply(p, v, r - l);
            return;
        }
        int m = (l + r) >> 1;
        push(p, r - l);
        rangeApply(p << 1, l, m, x, y, v);
        rangeApply(p << 1 | 1, m, r, x, y, v);
        pull(p);
    }
    void rangeApply(int x, int y, const Tag &v) {
        rangeApply(1, 0, n, x, y, v);
    }

    template <class F>
    int findFirst(int p, int l, int r, int x, int y, F &&pred) {
        if (x >= r || y <= l) return -1;
        if (x <= l && r <= y && !pred(info[p])) return -1;
        if (r - l == 1) return l;
        int m = (l + r) >> 1;
        push(p, r - l);
        int res = findFirst(p << 1, l, m, x, y, pred);
        if (res != -1) return res;
        return findFirst(p << 1 | 1, m, r, x, y, pred);
    }
    template <class F>
    int findFirst(int x, int y, F &&pred) {
        return findFirst(1, 0, n, x, y, pred);
    }

    template <class F>
    int findLast(int p, int l, int r, int x, int y, F &&pred) {
        if (x >= r || y <= l) return -1;
        if (x <= l && r <= y && !pred(info[p])) return -1;
        if (r - l == 1) return l;
        int m = (l + r) >> 1;
        push(p, r - l);
        int res = findLast(p << 1 | 1, m, r, x, y, pred);
        if (res != -1) return res;
        return findLast(p << 1, l, m, x, y, pred);
    }
    template <class F>
    int findLast(int x, int y, F &&pred) {
        return findLast(1, 0, n, x, y, pred);
    }
};

struct Tag {
    i64 add;
    Tag(i64 add_ = 0) : add(add_) {}
    void apply(const Tag &tag) {
        add += tag.add;
    }
};

struct Info {
    i64 sum;
    Info(i64 sum_ = 0) : sum(sum_) {}
    void apply(const Tag &tag, int len) {
        sum += tag.add * len;
    }
    Info operator+(const Info &other) const {
        return Info{sum + other.sum};
    }
};
```



## 树状数组

```cpp
template <typename T>
struct Fenwick {
    int n;
    vector<T> a;

    Fenwick(int n_ = 0) {
        init(n_);
    }

    void init(int n_) {
        n = n_;
        a.assign(n, T{});
    }

    void add(int x, const T &v) {
        for (int i = x + 1; i <= n; i += i & -i) {
            a[i - 1] = a[i - 1] + v;
        }
    }

    // 前缀和: [0, x)
    T sum(int x) {
        T ans{};
        for (int i = x; i > 0; i -= i & -i) {
            ans = ans + a[i - 1];
        }
        return ans;
    }

    // 区间和: [l, r)
    T rangeSum(int l, int r) {
        return sum(r) - sum(l);
    }

    int select(const T &k) {
        // 二分查找满足前缀和不超过 k 的最大下标
        int x = 0;
        T cur{};
        for (int i = 1 << std::__lg(n); i; i /= 2) {
            if (x + i <= n && cur + a[x + i - 1] <= k) {
                x += i;
                cur = cur + a[x - 1];
            }
        }
        return x;
    }
};
```



# 图论

## 树上问题

### 树链剖分

```cpp
struct HLD {
    int n;
    vector<int> siz, top, dep, parent, in, out, ord;
    vector<vector<int>> adj;
    int cur;

    HLD() {}
    HLD(int n) {
        init(n);
    }
    void init(int n) {
        this->n = n;
        siz.resize(n);
        top.resize(n);
        dep.resize(n);
        parent.resize(n);
        in.resize(n);
        out.resize(n);
        ord.resize(n);
        cur = 0;
        adj.assign(n, {});
    }
    void addEdge(int u, int v) {
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    void work(int root = 0) {
        top[root] = root;
        dep[root] = 0;
        parent[root] = -1;
        dfs1(root);
        dfs2(root);
    }
    void dfs1(int u) {
        if (parent[u] != -1) {
            adj[u].erase(std::find(adj[u].begin(), adj[u].end(), parent[u]));
        }

        siz[u] = 1;
        for (auto &v : adj[u]) {
            parent[v] = u;
            dep[v] = dep[u] + 1;
            dfs1(v);
            siz[u] += siz[v];
            if (siz[v] > siz[adj[u][0]]) {
                std::swap(v, adj[u][0]);
            }
        }
    }
    void dfs2(int u) {
        in[u] = cur++;
        ord[in[u]] = u;
        for (auto v : adj[u]) {
            top[v] = v == adj[u][0] ? top[u] : v;
            dfs2(v);
        }
        out[u] = cur;
    }
    int lca(int u, int v) {
        while (top[u] != top[v]) {
            if (dep[top[u]] > dep[top[v]]) {
                u = parent[top[u]];
            } else {
                v = parent[top[v]];
            }
        }
        return dep[u] < dep[v] ? u : v;
    }

    int dist(int u, int v) {
        return dep[u] + dep[v] - 2 * dep[lca(u, v)];
    }

    int jump(int u, int k) {
        if (dep[u] < k) return -1;
        int d = dep[u] - k;
        while (dep[top[u]] > d) {
            u = parent[top[u]];
        }
        return ord[in[u] - dep[u] + d];
    }

    bool isAncester(int u, int v) {
        return in[u] <= in[v] && in[v] < out[u];
    }

    int rootedParent(int u, int v) {
        std::swap(u, v);
        if (u == v) return u;
        if (!isAncester(u, v)) return parent[u];
        auto it = upper_bound(adj[u].begin(), adj[u].end(), v, [&](int x, int y) { return in[x] < in[y]; }) - 1;
        return *it;
    }

    int rootedSize(int u, int v) {
        if (u == v) return n;
        if (!isAncester(v, u)) return siz[v];
        return n - siz[rootedParent(u, v)];
    }

    int rootedLca(int a, int b, int c) {
        return lca(a, b) ^ lca(b, c) ^ lca(c, a);
    }
};
```



## 图的匹配

### 一般图最大匹配

- 带花树算法（Blossom Algorithm）$O(|V||E|^2)$
- 顶点索引 `0-indexed`；`g.add(u,v)` 双向边；
- 调用 `g.findMatching` 返回匹配数组，`match[u] = v` : 表示顶点 `u` 和顶点 `v` 相互匹配；`match[u] = -1`: 表示顶点 `u` 是未匹配点。可用 `std::count_if` 易得最大匹配对数。
- `greedy()` 通过贪心快速找到一个较大匹配，以减少后续带花树增广的次数，从而优化常数；**但有可能会破坏具体题目建图后的特殊结构！**

```cpp
struct Graph {
    int n;
    vector<vector<int>> g;
    Graph(int n) : n(n), g(n) {}
    void add(int u, int v) {
        g[u].push_back(v);
        g[v].push_back(u);
    }

    vector<int> findMatching() {
        vector<int> match(n, -1), vis(n), link(n), f(n), dep(n);
        // disjoint set union
        auto find = [&](int u) {
            while (f[u] != u)
                u = f[u] = f[f[u]];
            return u;
        };
        auto lca = [&](int u, int v) {
            u = find(u);
            v = find(v);
            while (u != v) {
                if (dep[u] < dep[v])
                    std::swap(u, v);
                u = find(link[match[u]]);
            }
            return u;
        };
        queue<int> que;
        auto blossom = [&](int u, int v, int p) {
            while (find(u) != p) {
                link[u] = v;
                v = match[u];
                if (vis[v] == 0) {
                    vis[v] = 1;
                    que.push(v);
                }
                f[u] = f[v] = p;
                u = link[v];
            }
        };
        // find an augmenting path starting from u and augment (if exist)
        auto augment = [&](int u) {
            while (!que.empty()) que.pop();
            std::iota(f.begin(), f.end(), 0);
            // vis = 0 corresponds to inner vertices
            // vis = 1 corresponds to outer vertices
            std::fill(vis.begin(), vis.end(), -1);
            que.push(u);
            vis[u] = 1;
            dep[u] = 0;
            while (!que.empty()) {
                int u = que.front();
                que.pop();
                for (auto v : g[u]) {
                    if (vis[v] == -1) {
                        vis[v] = 0;
                        link[v] = u;
                        dep[v] = dep[u] + 1;
                        // found an augmenting path
                        if (match[v] == -1) {
                            for (int x = v, y = u, temp; y != -1; x = temp, y = x == -1 ? -1 : link[x]) {
                                temp = match[y];
                                match[x] = y;
                                match[y] = x;
                            }
                            return;
                        }
                        vis[match[v]] = 1;
                        dep[match[v]] = dep[u] + 2;
                        que.push(match[v]);
                    } else if (vis[v] == 1 && find(v) != find(u)) {
                        // found a blossom
                        int p = lca(u, v);
                        blossom(u, v, p);
                        blossom(v, u, p);
                    }
                }
            }
        };
        // find a maximal matching greedily (decrease constant)
        auto greedy = [&]() {
            for (int u = 0; u < n; ++u) {
                if (match[u] != -1) continue;
                for (auto v : g[u]) {
                    if (match[v] == -1) {
                        match[u] = v;
                        match[v] = u;
                        break;
                    }
                }
            }
        };
        greedy();
        for (int u = 0; u < n; ++u) {
            if (match[u] == -1) augment(u);
        }
        return match;
    }
};

void solve() {
    int n, m;
    cin >> n >> m;
    Graph g(n);
    for (int i = 0; i < m; i++) {
        int u, v;
        cin >> u >> v;
        u--, v--;
        g.add(u, v);
    }

    auto match = g.findMatching();
    int cnt = count_if(match.begin(), match.end(), [](int x) { return x != -1; }) / 2;
    cout << cnt << "\n";
    for (int i = 0; i < match.size(); i++) {
        cout << match[i] + 1 << " \n"[i == match.size() - 1];
    }
}
```



# 字符串

## 后缀数组

- 时间复杂度：后缀数组构建 $O(n\log n)$；Kasai 算法求高度数组 `lcp[i]` $O(n)$
- 数组均为 `0-based`，其中 `lcp[0] = 0`

```cpp
struct SuffixArray {
    int n;
    vector<int> sa;   // 排名为 i 的后缀起始位置
    vector<int> rk;   // 第 i 个后缀的排名
    vector<int> lcp;  // LCP(suffix(sa[i]), suffix(sa[i-1])) 最长公共前缀长度

    SuffixArray(const string &s) {
        n = s.length();
        sa.resize(n);
        rk.resize(n);
        lcp.resize(n, 0);
        std::iota(sa.begin(), sa.end(), 0);
        sort(sa.begin(), sa.end(), [&](int i, int j) {
            return s[i] < s[j];
        });
        rk[sa[0]] = 0;
        for (int i = 1; i < n; i++) {
            rk[sa[i]] = rk[sa[i - 1]] + (s[sa[i]] != s[sa[i - 1]]);
        }
        // 倍增循环
        int k = 1;
        vector<int> tmp, cnt(n);
        tmp.reserve(n);
        while (rk[sa[n - 1]] < n - 1) {
            tmp.clear();
            for (int i = 0; i < k; i++) {
                tmp.push_back(n - k + i);
            }
            for (auto i : sa) {
                if (i >= k) {
                    tmp.push_back(i - k);
                }
            }
            std::fill(cnt.begin(), cnt.end(), 0);
            for (int i = 0; i < n; i++) {
                cnt[rk[i]]++;
            }
            for (int i = 1; i < n; i++) {
                cnt[i] += cnt[i - 1];
            }
            for (int i = n - 1; i >= 0; i--) {
                sa[--cnt[rk[tmp[i]]]] = tmp[i];
            }
            std::swap(rk, tmp);
            rk[sa[0]] = 0;
            for (int i = 1; i < n; i++) {
                rk[sa[i]] = rk[sa[i - 1]] + (tmp[sa[i - 1]] < tmp[sa[i]] || sa[i - 1] + k == n || tmp[sa[i - 1] + k] < tmp[sa[i] + k]);
            }
            k *= 2;
        }
        // Kasai 算法求 LCP
        for (int i = 0, j = 0; i < n; i++) {
            if (rk[i] == 0) {
                j = 0;
            } else {
                for (j -= (j > 0); i + j < n && sa[rk[i] - 1] + j < n && s[i + j] == s[sa[rk[i] - 1] + j]; j++);
                lcp[rk[i]] = j;
            }
        }
    }
};
```

## 后缀自动机

```cpp
struct SAM {
    static constexpr int ALPHABET_SIZE = 52;
    struct Node {
        int len;
        int link;
        array<int, ALPHABET_SIZE> next;
        Node() : len{}, link{}, next{} {}
    };
    vector<Node> t;
    SAM() {
        t.assign(2, Node());
        t[0].next.fill(1);
        t[0].len = -1;
    }
    int newNode() {
        t.emplace_back();
        return t.size() - 1;
    }
    int extend(int p, int c) {
        if (t[p].next[c]) {
            int q = t[p].next[c];
            if (t[q].len == t[p].len + 1) {
                return q;
            }
            int r = newNode();
            t[r].len = t[p].len + 1;
            t[r].link = t[q].link;
            t[r].next = t[q].next;
            t[q].link = r;
            while (t[p].next[c] == q) {
                t[p].next[c] = r;
                p = t[p].link;
            }
            return r;
        }
        int cur = newNode();
        t[cur].len = t[p].len + 1;
        while (!t[p].next[c]) {
            t[p].next[c] = cur;
            p = t[p].link;
        }
        t[cur].link = extend(p, c);
        return cur;
    }

    int extend(int p, char c, char offset = 'a') {
        return extend(p, c - offset);
    }

    int next(int p, int x) { return t[p].next[x]; }
    int next(int p, char c, char offset = 'a') {
        return next(p, c - 'a');
    }
    int link(int p) { return t[p].link; }
    int len(int p) { return t[p].len; }
    int size() { return t.size(); }
};
```



# 数学

## ModInt

- 快速幂、高精度模乘、安全取模（考虑负数）、扩展欧几里得求逆元
- `ModIntBase`：静态模数类（模数在编译时确定）
- `Barrett` 和 `DynModint`：动态模数类（模数在运行时输入）`DynModint<Id>::setMod(p)`

```cpp
template <typename T>
constexpr T power(T a, u64 b, T res = 1) {
    for (; b != 0; b /= 2, a *= a) {
        if (b & 1) res *= a;
    }
    return res;
}

// 高精度模乘 a * b % P
template <u32 P>
constexpr u32 mulMod(u32 a, u32 b) {
    return (u64)a * b % P;
}

template <u64 P>
constexpr u64 mulMod(u64 a, u64 b) {
    u64 res = a * b - u64(1.L * a * b / P - 0.5L) * P;
    res %= P;
    return res;
}

constexpr i64 safeMod(i64 x, i64 m) {
    x %= m;
    if (x < 0) x += m;
    return x;
}

// exgcd求逆元
// ax + by = gcd(a, b)
// return: (gcd(a, b), x)
// if gcd(a, b) == 1, x 是 a 的模 b 逆元
constexpr pair<i64, i64> invGcd(i64 a, i64 b) {
    a = safeMod(a, b);
    if (a == 0) {
        return {b, 0};
    }

    i64 s = b, t = a;
    i64 m0 = 0, m1 = 1;

    while (t) {
        i64 u = s / t;
        s -= t * u;
        m0 -= m1 * u;

        std::swap(s, t);
        std::swap(m0, m1);
    }

    if (m0 < 0) m0 += b / s;
    return {s, m0};
}

template <std::unsigned_integral U, U P>
struct ModIntBase {
public:
    constexpr ModIntBase() : x(0) {}
    template <std::unsigned_integral T>
    constexpr ModIntBase(T x_) : x(x_ % mod()) {}
    template <std::signed_integral T>
    constexpr ModIntBase(T x_) {
        using S = std::make_signed_t<U>;
        S v = x_ % S(mod());
        if (v < 0) {
            v += mod();
        }
        x = v;
    }

    constexpr static U mod() { return P; }

    constexpr U val() const { return x; }

    constexpr ModIntBase operator-() const {
        ModIntBase res;
        res.x = (x == 0 ? 0 : mod() - x);
        return res;
    }

    constexpr ModIntBase inv() const {
        return power(*this, mod() - 2);
    }

    constexpr ModIntBase &operator*=(const ModIntBase &rhs) & {
        x = mulMod<mod()>(x, rhs.val());
        return *this;
    }
    constexpr ModIntBase &operator+=(const ModIntBase &rhs) & {
        x += rhs.val();
        if (x >= mod()) {
            x -= mod();
        }
        return *this;
    }
    constexpr ModIntBase &operator-=(const ModIntBase &rhs) & {
        x -= rhs.val();
        if (x >= mod()) {
            x += mod();
        }
        return *this;
    }
    constexpr ModIntBase &operator/=(const ModIntBase &rhs) & {
        return *this *= rhs.inv();
    }

    friend constexpr ModIntBase operator*(ModIntBase lhs, const ModIntBase &rhs) {
        lhs *= rhs;
        return lhs;
    }
    friend constexpr ModIntBase operator+(ModIntBase lhs, const ModIntBase &rhs) {
        lhs += rhs;
        return lhs;
    }
    friend constexpr ModIntBase operator-(ModIntBase lhs, const ModIntBase &rhs) {
        lhs -= rhs;
        return lhs;
    }
    friend constexpr ModIntBase operator/(ModIntBase lhs, const ModIntBase &rhs) {
        lhs /= rhs;
        return lhs;
    }

    friend constexpr std::istream &operator>>(std::istream &is, ModIntBase &a) {
        i64 i;
        is >> i;
        a = i;
        return is;
    }
    friend constexpr std::ostream &operator<<(std::ostream &os, const ModIntBase &a) {
        return os << a.val();
    }

    friend constexpr bool operator==(const ModIntBase &lhs, const ModIntBase &rhs) {
        return lhs.val() == rhs.val();
    }
    friend constexpr std::strong_ordering operator<=>(const ModIntBase &lhs, const ModIntBase &rhs) {
        return lhs.val() <=> rhs.val();
    }

private:
    U x;
};

template <u32 P>
using ModInt = ModIntBase<u32, P>;
template <u64 P>
using ModInt64 = ModIntBase<u64, P>;

struct Barrett {
public:
    Barrett(u32 m_) : m(m_), im((u64)(-1) / m_ + 1) {}

    constexpr u32 mod() const { return m; }

    constexpr u32 mul(u32 a, u32 b) const {
        u64 z = a;
        z *= b;

        u64 x = u64((u128(z) * im) >> 64);

        u32 v = u32(z - x * m);
        if (m <= v) {
            v += m;
        }
        return v;
    }

private:
    u32 m;
    u64 im;
};

template <u32 Id>
struct DynModInt {
public:
    constexpr DynModInt() : x(0) {}
    template <std::unsigned_integral T>
    constexpr DynModInt(T x_) : x(x_ % mod()) {}
    template <std::signed_integral T>
    constexpr DynModInt(T x_) {
        int v = x_ % int(mod());
        if (v < 0) {
            v += mod();
        }
        x = v;
    }

    constexpr static void setMod(u32 m) { bt = m; }

    static u32 mod() { return bt.mod(); }

    constexpr u32 val() const { return x; }

    constexpr DynModInt operator-() const {
        DynModInt res;
        res.x = (x == 0 ? 0 : mod() - x);
        return res;
    }

    constexpr DynModInt inv() const {
        auto v = invGcd(x, mod());
        assert(v.first == 1);
        return v.second;
    }

    constexpr DynModInt &operator*=(const DynModInt &rhs) & {
        x = bt.mul(x, rhs.val());
        return *this;
    }
    constexpr DynModInt &operator+=(const DynModInt &rhs) & {
        x += rhs.val();
        if (x >= mod()) {
            x -= mod();
        }
        return *this;
    }
    constexpr DynModInt &operator-=(const DynModInt &rhs) & {
        x -= rhs.val();
        if (x >= mod()) {
            x += mod();
        }
        return *this;
    }
    constexpr DynModInt &operator/=(const DynModInt &rhs) & {
        return *this *= rhs.inv();
    }

    friend constexpr DynModInt operator*(DynModInt lhs, const DynModInt &rhs) {
        lhs *= rhs;
        return lhs;
    }
    friend constexpr DynModInt operator+(DynModInt lhs, const DynModInt &rhs) {
        lhs += rhs;
        return lhs;
    }
    friend constexpr DynModInt operator-(DynModInt lhs, const DynModInt &rhs) {
        lhs -= rhs;
        return lhs;
    }
    friend constexpr DynModInt operator/(DynModInt lhs, const DynModInt &rhs) {
        lhs /= rhs;
        return lhs;
    }

    friend constexpr std::istream &operator>>(std::istream &is, DynModInt &a) {
        i64 i;
        is >> i;
        a = i;
        return is;
    }
    friend constexpr std::ostream &operator<<(std::ostream &os, const DynModInt &a) {
        return os << a.val();
    }

    friend constexpr bool operator==(const DynModInt &lhs, const DynModInt &rhs) {
        return lhs.val() == rhs.val();
    }
    friend constexpr std::strong_ordering operator<=>(const DynModInt &lhs, const DynModInt &rhs) {
        return lhs.val() <=> rhs.val();
    }

private:
    u32 x;
    static Barrett bt;
};

template <u32 Id>
Barrett DynModInt<Id>::bt = 998244353;

using Z = ModInt<1000000007>;
// using Z = ModInt<998244353>;
// using D = DynModInt<0>;
// DMI::setMod(p);
```

## 数论

### 扩展欧几里得

- 求 $ax+by=\gcd (a,b)$ 的一组可行解；方程的解有无数个，有的可能爆 `long long`，但扩欧算法可以保证求出的可行解 $|x| \leq b, |y| \leq a$

```cpp
// ax + by = gcd(a,b)
// return: gcd(a,b) and x,y (reference)
i64 exgcd(i64 a, i64 b, i64 &x, i64 &y) {
    if (b == 0) {
        x = 1;
        y = 0;
        return a;
    }
    i64 g = exgcd(b, a % b, y, x);
    y -= a / b * x;
    return g;
}
```

### 线性同余方程

形如 $ax\equiv b \pmod m$ 的方程，其中 $a,b,m$ 为整数，$m>0$。方程等价于 $ax+my=b$，方程有解当且仅当 $g=\gcd(a,m)\mid b$，可使用扩展欧几里得算法，通解为 $x\equiv (x_0+k\cdot m/g),\ k\in Z$。

```cpp
// return: 最小非负解 x0 or 无解 -1
int linear_congruence(i64 a, i64 b, i64 m) {
    i64 x, y;
    i64 g = exgcd(a, m, x, y);
    if (b % g != 0) return -1;
    m /= g;
    x *= (b / g);
    return (x % m + m) % m;
}
```

### 中国剩余定理

#### CRT

求解如下形式的一元线性同余方程组（其中 $m_1,m_2,\cdots,m_k$ 两两互质）
$$
\left\{
\begin{array}{l}
x \equiv a_{1} \pmod{m_{1}} \\
x \equiv a_{2} \pmod{m_{2}} \\
\vdots \\
x \equiv a_{k} \pmod{m_{k}}
\end{array}
\right.
$$

```cpp
i64 CRT(const vector<i64> &a, const vector<i64> &m) {
    int n = a.size();
    i64 M = 1;
    for (int i = 0; i < n; i++) M *= m[i];
    i64 res = 0;
    for (int i = 0; i < n; i++) {
        i64 Mi = M / m[i];
        i64 x, y;
        // Mi * x ≡ 1 (mod m[i]) 求逆元 <=>
        // Mi * x + m[i] * y = gcd(Mi, m[i]) = 1
        exgcd(Mi, m[i], x, y);
        // use i128 to avoid overflow !!!
        // i64 tmp = (i128)a[i] * Mi % M * x % M;
        i64 tmp = a[i] * Mi * x % M;
        res = (res + tmp) % M;
    }
    return (res + M) % M;
}
```

#### 扩展CRT

不保证模数 $m_i$ 两两互质的情况。通解：$x \equiv x_0 \pmod M$

```cpp
// return: x (mod M) 最小非负解, -1无解
i64 exCRT(const vector<i64> &a, const vector<i64> &m) {
    int n = a.size();
    i64 M = m[0], R = a[0];
    for (int i = 1; i < n; i++) {
        i64 x, y;
        i64 g = exgcd(M, m[i], x, y);
        if ((a[i] - R) % g != 0) return -1;  // 无解
        // use i128 to avoid overflow !!!
        i64 t = ((i128)(a[i] - R) / g * x) % (m[i] / g);

        i128 tmp = (i128)t * M + R;
        M = (i128)M / g * m[i];  // lcm(M, m[i])
        R = (i64)((tmp % M + M) % M);
    }
    return (R % M + M) % M;
}
```

### 欧拉函数

欧拉函数：$\varphi(n)$ 表示 $\leq n$ 和 $n$ 互质的数的个数。

- 若 $n$ 为质数，显然有 $\varphi(n)=n-1$
- 设 $n = \prod_{i=1}^{s} p_i^{k_i}$（唯一分解定理），则有：

$$
\varphi(n)=n\times \prod_{i=1}^{s}\frac{p_i-1}{p_i}
$$

```cpp
i64 phi(i64 n) {
    i64 res = n;
    for (i64 i = 2; i * i <= n; i++) {
        if (n % i == 0) {
            res = res / i * (i - 1);
            while (n % i == 0) n /= i;
        }
    }
    if (n > 1) res = res / n * (n - 1);
    return res;
}
```

### 扩展欧拉定理

欧拉降幂：任意模数下、任意底数的幂次计算，将指数转化为 $\leq 2\varphi(m)$，从而可以通过快速幂在 $O(\log \varphi(m))$ 内计算 $a^b \mathrm{~mod~}m$。

- 对于任意正整数 $m$、整数 $a$ 和非负整数 $b$，有：
  $$
  a^b\equiv\begin{cases}a^{b\mathrm{~mod~}\varphi(m)},&\gcd(a,m)=1,\\a^b,&\gcd(a,m)\neq1,b<\varphi(m),&\mathrm{(mod~}m)\\a^{(b\mathrm{~mod~}\varphi(m))+\varphi(m)},&\gcd(a,m)\neq1,b\geq\varphi(m).&\end{cases}
  $$

- 当 $m$ 是质数时，使用第一条简化降幂要注意特判 $\gcd(a,m)\neq1$ 即 $a\%m=0$，此时 $a^b \equiv 0 \pmod m$ 
```cpp
// 扩展欧拉定理 a^b mod m
i64 exEuler(i64 a, i64 b, i64 m) {
    i64 ph = phi(m);
    i64 d = gcd(a, m);
    if (b == 0) return 1 % m;
    if (gcd(a, m) == 1) return qpow(a, b % ph, m);
    if (b < ph) return qpow(a, b, m);
    return qpow(a, b % ph + ph, m);
}
i64 exEuler(i64 a, string bb, i64 m) {
    i64 ph = phi(m);
    i64 b = 0;
    bool flag = false;
    for (int i = 0; i < bb.length(); i++) {
        b = b * 10 + bb[i] - '0';
        if(b >= phi) {
            b %= phi;
            flag = true;
        }
    }
    if(flag) b += phi; // b >= φ(m)
    return qpow(a, b, m);
}
```

### 卢卡斯定理

大组合数 $n$ 取模的求解，模数不太大 $\mathrm{mod} \sim 10^6$；更准确的说，只要模数的唯一分解 $\mathrm{mod}=\prod p_i^{k_i}$ 中所有素数幂的和 $\sum p_i^{k_i} \sim 10^6$ 规模时即可使用。

#### Lucas定理

- 模数为**素数** $p$，$n$ 可以很大远超常规预处理的范围，只需预处理到 $p$：
  $$
  \binom nm\equiv \binom{\lfloor n/p\rfloor}{\lfloor m/p\rfloor}\binom{n\mathrm{~mod~}p}{m\mathrm{~mod~}p}\pmod p
  $$

- 时间复杂度：组合数预处理 $O(p)$，单次递归查询 $O(\log_p n)$

```cpp
// template <int P>
struct Lucas {
    int P;  // prime mod
    vector<i64> fac, ifac;

    Lucas(int p) : P(p) {
        fac.resize(p);
        ifac.resize(p);
        fac[0] = ifac[0] = 1;
        for (int i = 1; i < p; i++) {
            fac[i] = fac[i - 1] * i % P;
        }
        ifac[p - 1] = qpow(fac[p - 1], P - 2, P);
        for (int i = p - 1; i >= 1; i--) {
            ifac[i - 1] = ifac[i] * i % P;
        }
    }

    // C(n,m) % P: only for 0 <= n,m < P
    i64 C(int n, int m) {
        if (n < m || m < 0) return 0;
        return fac[n] * ifac[m] % P * ifac[n - m] % P;
    }

    // C(n,m) % P: Lucas theorem
    i64 lucas(i64 n, i64 m) {
        if (m == 0) return 1;
        return C(n % P, m % P) * lucas(n / P, m / P) % P;
    }
};
// Lucas comb(p); comb.lucas(n,m) => C(n,m) % p
```

#### 扩展Lucas定理

对于 $\mathrm{mod}$ 不是素数的情况（素数幂 or 一般合数），质因数分解得 $\mathrm{mod}=p_1^{\alpha_1}p_2^{\alpha_2}\cdots p_k^{\alpha_k}$，分别求出模 $p_i ^{\alpha_i}$ 意义下组合数 $C(n,m)$ 的余数，对同余方程组用 $\text{CRT}$ 合并答案。

- 时间复杂度：$O(\sqrt{\mathrm{mod}}+\sum{p_i^{\alpha_i}})$
- 如果 $\mathrm{mod}$ 可以很容易分解为**幂次为1的质数**，建议用Lucas对每个 $p_i$ 计算后CRT合并，而不是使用exLucas

```cpp
// C(n,m) % mod, mod can be composite
class ExLucas {
public:
    static i64 exlucas(i64 n, i64 m, i64 mod) {
        if (mod == 1 || n < m || m < 0) return 0;
        vector<i64> mods, a;
        i64 x = mod;
        for (i64 i = 2; i * i <= x; i++) {
            if (x % i == 0) {
                i64 pk = 1;
                while (x % i == 0) {
                    x /= i;
                    pk *= i;
                }
                mods.push_back(pk);
                a.push_back(C(n, m, i, pk));
            }
        }
        if (x > 1) {
            mods.push_back(x);
            a.push_back(C(n, m, x, x));
        }
        return CRT(a, mods);
    }

private:
    static i64 fac(i64 n, i64 p, i64 pk, const vector<i64> &pre) {
        if (n == 0) return 1;
        i64 res = pre[pk];
        res = qpow(res, n / pk, pk);
        res = res * pre[n % pk] % pk;
        return res * fac(n / p, p, pk, pre) % pk;
    }
    static i64 inv(i64 a, i64 m) {
        i64 x, y;
        exgcd(a, m, x, y);
        return (x % m + m) % m;
    }
    // C(n,m) % pk, p is prime factor of pk
    static i64 C(i64 n, i64 m, i64 p, i64 pk) {
        if (n < m || m < 0) return 0;
        // O(pk) 预处理前缀积
        vector<i64> pre(pk + 1);
        pre[0] = 1;
        for (i64 i = 1; i <= pk; i++) {
            if (i % p == 0) {
                pre[i] = pre[i - 1];
            } else {
                pre[i] = pre[i - 1] * i % pk;
            }
        }
        i64 f1 = fac(n, p, pk, pre);
        i64 f2 = fac(m, p, pk, pre);
        i64 f3 = fac(n - m, p, pk, pre);
        i64 res = 0;
        for (i64 i = n; i; i /= p) res += i / p;
        for (i64 i = m; i; i /= p) res -= i / p;
        for (i64 i = n - m; i; i /= p) res -= i / p;
        return f1 * inv(f2, pk) % pk * inv(f3, pk) % pk * qpow(p, res, pk) % pk;
    }
};
// Exlucas::exlucas(n, m, mod) => C(n,m) % mod
```



### 齐肯多夫定理

任何正整数 $n$ 都可以唯一表示成若干个**不连续**的斐波那契数之和。对于任何正整数，其齐肯多夫表述法都可以贪心选出每次不大于 $n$ 的最大斐波那契数 $f_i$

时间复杂度：预处理 $P(\log N)$；每次分解 $O(\log n)$

```cpp
struct Zeckendorf {
    vector<i64> fib;  // 1,2,3,5,8...为便于分解只存储一个1

    Zeckendorf(i64 maxn = 1e18) {
        fib = {1, 2};
        while (true) {
            i64 nxt = fib[fib.size() - 1] + fib[fib.size() - 2];
            if (nxt > maxn) break;
            fib.push_back(nxt);
        }
    }

    bool isFib(i64 n) {
        return binary_search(fib.begin(), fib.end(), n);
    }

    // Zeckendorf decomposition -> 从大到小
    vector<i64> decompose(i64 n) {
        vector<i64> res;
        int i = fib.size() - 1;
        while (n > 0 && i >= 0) {
            if (fib[i] <= n) {
                n -= fib[i];
                res.push_back(fib[i]);
                i -= 2;  // avoid consecutive fib numbers
            } else {
                i--;
            }
        }
        return res;
    }
};
```



## 组合数学

- `using Z = ModInt<1000000007>;`
- `Comb`：组合数、阶乘、逆元、逆元阶乘预处理

```cpp
struct Comb {
    int n;  // current precomputed up to n
    vector<Z> _fac;
    vector<Z> _invfac;
    vector<Z> _inv;

    Comb() : n{0}, _fac{1}, _invfac{1}, _inv{0} {}
    Comb(int n) : Comb() {
        init(n);
    }

    void init(int m) {
        if (m <= n) return;
        _fac.resize(m + 1);
        _invfac.resize(m + 1);
        _inv.resize(m + 1);

        for (int i = n + 1; i <= m; i++) {
            _fac[i] = _fac[i - 1] * i;
        }
        _invfac[m] = _fac[m].inv();
        for (int i = m; i > n; i--) {
            _invfac[i - 1] = _invfac[i] * i;
            _inv[i] = _invfac[i] * _fac[i - 1];
        }
        n = m;
    }

    Z fac(int m) {
        if (m > n) init(2 * m);
        return _fac[m];
    }
    Z invfac(int m) {
        if (m > n) init(2 * m);
        return _invfac[m];
    }
    Z inv(int m) {
        if (m > n) init(2 * m);
        return _inv[m];
    }
    Z binom(int n, int m) {
        if (n < m || m < 0) return 0;
        return fac(n) * invfac(m) * invfac(n - m);
    }
} comb;
```



## 博弈论

### 斐波那契博弈

**规则（单堆取石子）：**

1. 初始有 $n$ 个石子。
2. **先手第一步**：必须取走 $1\sim n-1$ 个（不能一次全取光）
3. **之后每一步**：若上一步对手取了 $k$ 个，则这一步**最多**取 $2k$ 个（至少 1 个）。
4. 取到最后一个石子的玩家获胜。

**结论**：

- $n$ 是斐波那契数 ⇒ 先手必败；（第二数学归纳法）
- 否则 ⇒ 先手必胜。（齐肯多夫定理分解 + 先手取完最小堆）

# 其他

## int128

```cpp
std::ostream &operator<<(std::ostream &os, i128 n) {
    if (n == 0) return os << '0';
    if (n < 0) os << '-', n = -n;
    string s;
    while (n > 0) {
        s += char('0' + n % 10);
        n /= 10;
    }
    reverse(s.begin(), s.end());
    return os << s;
}

i128 toi128(const string &s) {
    i128 n = 0;
    bool neg = false;
    int i = 0;
    if (s[i] == '-') neg = true, i++;
    for (; i < (int)s.size(); i++) {
        n = n * 10 + (s[i] - '0');
    }
    if (neg) n = -n;
    return n;
}

i128 sqrti128(i128 n) {
    i128 lo = 0, hi = 1e16;
    while (lo < hi) {
        i128 mid = (lo + hi + 1) / 2;
        if (mid * mid <= n) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    return lo;
}

i128 gcd(i128 a, i128 b) {
    while (b) {
        a %= b;
        swap(a, b);
    }
    return a;
}
```

## 基姆-拉尔森公式

```cpp
// 基姆-拉尔森公式
// 返回 0=Sunday, 1=Monday, ..., 6=Saturday
int weekday(int y, int m, int d) {
    if (m == 1 || m == 2) {
        m += 12;
        y -= 1;
    }
    int w = (d + 2 * m + 3 * (m + 1) / 5 + y + y / 4 - y / 100 + y / 400 + 1) % 7;
    return (w + 7) % 7;
}
```

## 随机化

### 伪随机数

```cpp
// return unsigned int
mt19937 rnd(time(nullptr)); 
// return unsigned long long
mt19937_64 rnd(chrono::steady_clock::now().time_since_epoch().count()); // 随机精度更高
```

