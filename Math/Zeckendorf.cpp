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