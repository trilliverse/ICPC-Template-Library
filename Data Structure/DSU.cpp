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