struct SCC {
    int n;
    vector<vector<int>> &g;
    vector<int> dfn, low;
    stack<int> st;
    int cur, cnt;              // current time, number of sccs (0-indexed)
    vector<int> comp;          // component id of node i
    vector<vector<int>> node;  // nodes in each scc

    SCC(vector<vector<int>> &_g) : n((int)_g.size()), g(_g) {
        // init
        dfn.assign(n, 0);
        low.assign(n, 0);
        comp.assign(n, -1);
        while (!st.empty()) st.pop();
        cur = cnt = 0;
        node.clear();
        // work
        for (int i = 0; i < n; i++) {
            if (dfn[i] == 0) dfs(i);
        }
    }

    void dfs(int u) {
        dfn[u] = low[u] = ++cur;
        st.push(u);
        for (int v : g[u]) {
            if (dfn[v] == 0) {
                dfs(v);
                low[u] = min(low[u], low[v]);
            } else if (comp[v] == -1) {
                low[u] = min(low[u], dfn[v]);
            }
        }
        // head of an scc
        if (dfn[u] == low[u]) {
            while (true) {
                int x = st.top();
                st.pop();
                comp[x] = cnt;
                if ((int)node.size() <= cnt) node.push_back({});
                node[cnt].push_back(x);
                if (x == u) break;
            }
            cnt++;
        }
    }

    // build the DAG after Tarjan SCCs (逆拓扑序)
    vector<vector<int>> build() {
        vector<vector<int>> dag(cnt);
        for (int u = 0; u < n; u++) {
            for (int v : g[u]) {
                int uid = comp[u], vid = comp[v];
                if (uid != vid) dag[uid].push_back(vid);
            }
        }
        // remove duplicates in dag (if needed)
        for (auto &e : dag) {
            sort(e.begin(), e.end());
            e.erase(unique(e.begin(), e.end()), e.end());
        }
        return dag;
    }
};