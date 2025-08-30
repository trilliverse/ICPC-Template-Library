struct VBCC {
    int n;
    const vector<vector<int>>& g;
    vector<int> dfn, low;
    stack<int> stk;
    int cur;  // current time

    int cnt;                   // number of vbccs
    vector<vector<int>> node;  // nodes in each vbcc
    vector<bool> cut;          // 割点

    VBCC(const vector<vector<int>>& _g) : n(_g.size()), g(_g) {
        dfn.assign(n, 0);
        low.assign(n, 0);
        cut.assign(n, false);
        while (!stk.empty()) stk.pop();
        node.clear();
        cur = cnt = 0;

        for (int i = 0; i < n; ++i) {
            if (!dfn[i]) dfs(i, -1);
        }

        // 如果题目要求孤立点也算一个V-BCC
        vector<bool> vis(n, false);
        for (const auto& c : node) {
            for (int x : c) {
                vis[x] = true;
            }
        }

        for (int i = 0; i < n; i++) {
            if (!vis[i]) {
                cnt++;
                node.push_back({i});
            }
        }
    }

    void dfs(int u, int fa) {
        dfn[u] = low[u] = ++cur;
        stk.push(u);
        int child = 0;  // number of children in DFS tree
        if (g[u].empty() && fa == -1) return;
        for (int v : g[u]) {
            if (v == fa) continue;
            if (!dfn[v]) {
                child++;
                dfs(v, u);
                low[u] = min(low[u], low[v]);
                if (low[v] >= dfn[u]) {
                    if (fa != -1) cut[u] = true;
                    cnt++;
                    node.emplace_back();
                    while (true) {
                        int t = stk.top();
                        stk.pop();
                        node.back().push_back(t);
                        if (t == v) break;
                    }
                    node.back().push_back(u);  // 割点
                }
            } else {
                low[u] = min(low[u], dfn[v]);
            }
        }
        if (fa == -1 && child > 1) {
            cut[u] = true;
        }
    }
};