std::ostream &operator<<(std::ostream &os, i128 n) {
    if (n == 0) return os << '0';
    if (n < 0) os << '-', n = -n;
    string s;
    while (n > 0) {
        s += char('0' + n % 10);
        n /= 10;
    }
    reverse(s.begin(), s.end());
    return os << s;
}

i128 toi128(const string &s) {
    i128 n = 0;
    bool neg = false;
    int i = 0;
    if (s[i] == '-') neg = true, i++;
    for (; i < (int)s.size(); i++) {
        n = n * 10 + (s[i] - '0');
    }
    if (neg) n = -n;
    return n;
}

i128 sqrti128(i128 n) {
    i128 lo = 0, hi = 1e16;
    while (lo < hi) {
        i128 mid = (lo + hi + 1) / 2;
        if (mid * mid <= n) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    return lo;
}

i128 gcd(i128 a, i128 b) {
    while (b) {
        a %= b;
        swap(a, b);
    }
    return a;
}