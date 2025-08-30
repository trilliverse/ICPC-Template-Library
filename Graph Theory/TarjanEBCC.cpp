struct EBCC {
    int n, m;
    vector<vector<pair<int, int>>> g;  // {邻居, 边索引}
    vector<int> dfn, low;
    vector<bool> bridge;
    int cur;  // current time

    int cnt;                   // number of ebccs
    vector<int> comp;          // component id of node i (0-indexed)
    vector<vector<int>> node;  // nodes in each ebcc (0-indexed)

    EBCC(int _n) : n(_n), g(_n) {
        m = 0;  // number of edges
    }

    // 添加无向边，并赋予其唯一索引
    void add(int u, int v) {
        g[u].emplace_back(v, m);
        g[v].emplace_back(u, m);
        m++;
    }

    // Tarjan DFS，通过父边索引来识别桥
    void dfs1(int u, int faidx) {
        dfn[u] = low[u] = ++cur;
        for (const auto& edge : g[u]) {
            int v = edge.first;
            int curidx = edge.second;
            if (curidx == faidx) {
                continue;
            }

            if (!dfn[v]) {
                dfs1(v, curidx);
                low[u] = std::min(low[u], low[v]);
                if (low[v] > dfn[u]) {
                    bridge[curidx] = true;
                }
            } else {
                low[u] = std::min(low[u], dfn[v]);
            }
        }
    }

    // 常规DFS，不经过桥，划分E-BCC
    void dfs2(int u, int id) {
        comp[u] = id;
        node[id].push_back(u);  // 0-indexed

        for (const auto& edge : g[u]) {
            int v = edge.first;
            int curidx = edge.second;
            if (bridge[curidx] || comp[v] != -1) {
                continue;
            }
            dfs2(v, id);
        }
    }

    void work() {
        dfn.assign(n, 0);
        low.assign(n, 0);
        bridge.assign(m, false);
        cur = cnt = 0;
        comp.assign(n, -1);
        node.clear();

        for (int i = 0; i < n; i++) {
            if (!dfn[i]) dfs1(i, -1);
        }

        for (int i = 0; i < n; i++) {
            if (comp[i] == -1) {
                node.emplace_back();
                dfs2(i, cnt);
                cnt++;
            }
        }
    }
};