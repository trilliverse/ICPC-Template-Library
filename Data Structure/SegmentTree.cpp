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