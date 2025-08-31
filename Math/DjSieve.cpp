#define int long long

struct Sieve {
    int n;
    vector<int> prime;
    vector<int> spf;      // smallest prime factor
    vector<int> mu;       // mobius
    vector<int> phi;      // euler
    vector<int> mu_pre;   // prefix sum of mobius
    vector<int> phi_pre;  // prefix sum of euler

    Sieve(int _n) : n(_n) {
        prime.reserve(n / (log(n) - 1.1));
        spf.resize(n + 1, 0);
        mu.resize(n + 1, 0);
        phi.resize(n + 1, 0);
        mu_pre.resize(n + 1, 0);
        phi_pre.resize(n + 1, 0);
        mu[1] = 1;
        phi[1] = 1;

        for (int i = 2; i <= n; i++) {
            if (spf[i] == 0) {
                prime.push_back(i);
                spf[i] = i;
                phi[i] = i - 1;
                mu[i] = -1;
            }

            for (int p : prime) {
                if (i * p > n) break;
                spf[i * p] = p;
                if (i % p == 0) {
                    mu[i * p] = 0;
                    phi[i * p] = phi[i] * p;
                    break;
                } else {
                    mu[i * p] = -mu[i];
                    phi[i * p] = phi[i] * (p - 1);
                }
            }
        }

        for (int i = 1; i <= n; i++) {
            mu_pre[i] = mu_pre[i - 1] + mu[i];
            phi_pre[i] = phi_pre[i - 1] + phi[i];
        }
    }

    int get_phi_pre(int n) const { return phi_pre[n]; }
    int get_mu_pre(int n) const { return mu_pre[n]; }
};

class DjSieve {
    const Sieve &base;
    unordered_map<int, int> sphi;  // prefix sum of euler
    unordered_map<int, int> smu;   // prefix sum of mobius

public:
    explicit DjSieve(const Sieve &sieve) : base(sieve) {}

    int get_phi_sum(int n) {
        if (n <= base.n) return base.get_phi_pre(n);
        auto it = sphi.find(n);
        if (it != sphi.end()) return it->second;

        int res = n * (n + 1) / 2;
        for (int l = 2, r; l <= n; l = r + 1) {
            r = n / (n / l);
            res -= (r - l + 1) * get_phi_sum(n / l);
        }
        return sphi[n] = res;
    }

    int get_mu_sum(int n) {
        if (n <= base.n) return base.get_mu_pre(n);
        auto it = smu.find(n);
        if (it != smu.end()) return it->second;

        int res = 1;
        for (int l = 2, r; l <= n; l = r + 1) {
            r = n / (n / l);
            res -= (r - l + 1) * get_mu_sum(n / l);
        }
        return smu[n] = res;
    }
};

constexpr int maxn = 1e6 + 5;  // n^(2/3)
Sieve sieve(maxn);
DjSieve dj(sieve);