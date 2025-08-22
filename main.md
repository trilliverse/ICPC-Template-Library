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

- 数组 `0-based`；查询区间**左闭右开** $[l,r)$
- `Info` 类表示维护信息，重载 `operator +`

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

## 图的匹配

### 一般图最大匹配

- 带花树算法（Blossom Algorithm）$O(|V||E|^2)$
- 顶点索引 `0-indexed`；`g.add(u,v)` 双向边；
- 调用 `g.findMatching` 返回匹配数组，`match[u] = v` : 表示顶点 `u` 和顶点 `v` 相互匹配；`match[u] = -1`: 表示顶点 `u` 是未匹配点。可用 `std::count_if` 易得最大匹配对数。

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



# 数学

## ModInt类

- 快速幂、高精度模乘、安全取模（考虑负数）、扩展欧几里得求逆元
- `ModIntBase`：静态模数类（模数在编译时确定）
- `Barrett` 和 `DynModint`：动态模数类（模数在运行时输入）`DynModint<Id>::setMod(p)`

```cpp
template <typename T>
constexpr T power(T a, u64 b, T res = 1) {
    for (; b != 0; b /= 2; a *= a) {
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

