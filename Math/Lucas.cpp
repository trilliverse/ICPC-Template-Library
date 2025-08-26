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