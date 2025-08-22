struct SuffixArray {
    int n;
    vector<int> sa;   // 排名为 i 的后缀起始位置
    vector<int> rk;   // 第 i 个后缀的排名
    vector<int> lcp;  // lcp[i] = lcp(sa[i], sa[i-1]) 最长公共前缀长度

    SuffixArray(const string &s) {
        n = s.length();
        sa.resize(n);
        rk.resize(n);
        lcp.resize(n - 1);
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
                lcp[rk[i] - 1] = j;
            }
        }
    }
};