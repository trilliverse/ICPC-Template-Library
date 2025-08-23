// 基姆-拉尔森公式
// 返回 0=Sunday, 1=Monday, ..., 6=Saturday
int weekday(int y, int m, int d) {
    if (m == 1 || m == 2) {
        m += 12;
        y -= 1;
    }
    int w = (d + 2 * m + 3 * (m + 1) / 5 + y + y / 4 - y / 100 + y / 400 + 1) % 7;
    return (w + 7) % 7;
}