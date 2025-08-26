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