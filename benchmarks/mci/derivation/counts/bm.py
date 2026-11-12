from fractions import Fraction
def berlekamp_massey(seq):
    """minimal C (C[0]=1) with sum_i C[i] a[n-i] = 0 over Q."""
    a = [Fraction(x) for x in seq]
    C = [Fraction(1)]; B = [Fraction(1)]; L = 0; m = 1; b = Fraction(1)
    for n in range(len(a)):
        d = a[n] + sum(C[i] * a[n - i] for i in range(1, L + 1))
        if d == 0: m += 1; continue
        T = C[:]
        coef = d / b
        C = C + [Fraction(0)] * (len(B) + m - len(C))
        for i in range(len(B)): C[i + m] -= coef * B[i]
        if 2 * L <= n:
            L = n + 1 - L; B = T; b = d; m = 1
        else: m += 1
    return [int(c) if c.denominator == 1 else c for c in C[:L + 1]], L
