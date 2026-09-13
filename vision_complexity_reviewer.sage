import math

# ======================================================================
# Arithmetic costs from Section 2
# ======================================================================

def lg(x):
    x = int(x)
    if x <= 0:
        return float('-inf')
    b = x.bit_length()
    if b > 60:
        return (b - 60) + math.log2(x >> (b - 60))
    return math.log2(x)

def lgM(d):
    r"""Return log2 M(d), with M(d) = d log d log log d as in Equation (1)."""
    ld = lg(d)
    l1 = max(ld, 1.0)
    l2 = max(math.log2(l1), 1.0)
    return ld + math.log2(l1) + math.log2(l2)

def lgU(d, n):
    r"""Return log2 U(d) for q = 2^n, using the bound in Equation (2)."""
    ld = lg(d)
    l1 = max(ld, 1.0)
    l2 = max(math.log2(l1), 1.0)
    return ld + math.log2(l1) + math.log2(l1 + n * l2)

def lse(vals):
    vals = [v for v in vals if v > float('-inf')]
    if not vals:
        return float('-inf')
    m = max(vals)
    return m + math.log2(sum(2.0 ** (v - m) for v in vals))

def C1(d, e, delta, B):
    r"""Cost C1(d,e,delta,B) for reducing one variable, as in Appendix A.1."""
    d, e, B = int(d), int(e), int(B)
    t1 = 3 * lg(d) if d > 0 else float('-inf')
    t2 = lg(d) + lgM(4 * (e + delta * d // 4 + 1) ** 2 * B)
    return lse([t1, t2])

def C2(d, e, delta, B):
    r"""Cost C2(d,e,delta,B) for reducing both variables, as in Appendix A.1."""
    d, e, B = int(d), int(e), int(B)
    t1 = 1 + 3 * lg(d) if d > 0 else float('-inf')
    t2 = lg(d) + lgM(4 * (d + 1) * (e + delta * d // 4 + 1) ** 2 * B)
    t3 = lg(d) + lgM(16 * (e + delta * d // 2 + 1) ** 2 * B)
    return lse([t1, t2, t3])

# ======================================================================
# Reduction costs from Appendix A.1
# ======================================================================

def A1_odd(N):
    r"""
    Evaluate the Appendix A.1 reduction costs for odd N.

    The returned values are the backward, forward, and total reduction costs,
    followed by the numbers of terms in the two sums.
    """
    if N % 2 == 0:
        raise ValueError("A1_odd requires odd N.")
    if N < 3:
        raise ValueError("The Vision MITM model is evaluated here for N >= 2.")
    rho = N + 1
    h = rho // 2

    def cen(c):
        return (c + 1) ** 2

    # The case N = 3 is evaluated directly, as specified in Appendix A.1.
    if N == 3:
        T = [C2(8, 0, 4, 1), C1(12, 76, 4, 1)]
        S = [C1(4, 8, 8, 1)]
        return lse(T), lse(S), lse([lse(T), lse(S)]), len(T), len(S)

    # Backward (output-side) reduction.
    T = [C2(8 * 4 ** j, 0, 8, 16 ** j) for j in range(0, h - 2)]
    T.append(C2(8 * 4 ** (h - 2), 0, 4, 16 ** (h - 2)))

    c = [None, 4 ** h]                                  # c^out_1 = 4^{rho/2}
    for s in range(1, h - 1):                           # s = 1 .. rho/2-2
        cs = c[s]
        T.append(C1(12, 36, 8, 13 ** (rho - 2 * s - 4) * cen(4 * cs)))
        for j in range(1, h - s - 1):
            T.append(C2(4 ** (j + 2) - 4, 12, 8,
                        4 * 16 ** (j - 1) * 13 ** (rho - 2 * s - 2 * j - 4)
                        * cen(4 * cs)))
        T.append(C2(4 ** (h - s + 1) - 4, 4 * cs, 4, 4 * 16 ** (h - s - 2)))

        mid = 16 * cs + 8 * (4 ** (h - s + 1) - 4)
        for j in range(0, h - s - 2):
            T.append(C2(40 * 4 ** j - 4, 12, 8,
                        16 ** j * 13 ** (rho - 2 * s - 2 * j - 6) * cen(mid)))
        T.append(C2(40 * 4 ** (h - s - 2) - 4, mid, 4, 16 ** (h - s - 2)))
        c.append(mid + 2 * (40 * 4 ** (h - s - 2) - 4))
    T.append(C1(12, 4 * c[h - 1] + 12, 4, 1))

    # Forward (input-side) reduction.
    S = [C1(4, 8, 8, 1)]
    S += [C2(4 ** j, 0, 8, 4 * 16 ** (j - 2)) for j in range(2, h)]

    c1 = 4 ** h                                          # c^in_1
    for j in range(0, h - 3):
        S.append(C2(40 * 4 ** j - 4, 12, 8,
                    16 ** j * 13 ** (rho - 2 * j - 8) * cen(4 * c1)))
    S.append(C2(40 * 4 ** (h - 3) - 4, 4 * c1, 8, 16 ** (h - 3)))

    ci = [None, c1, 4 * c1 + 4 * (40 * 4 ** (h - 3) - 4)]
    for s in range(2, h - 1):                            # s = 2 .. rho/2-2
        cs = ci[s]
        S.append(C1(12, 36, 8, 13 ** (rho - 2 * s - 4) * cen(4 * cs)))
        for j in range(1, h - s - 1):
            S.append(C2(4 ** (j + 2) - 4, 12, 8,
                        4 * 16 ** (j - 1) * 13 ** (rho - 2 * s - 2 * j - 4)
                        * cen(4 * cs)))
        S.append(C2(4 ** (h - s + 1) - 4, 4 * cs, 8, 4 * 16 ** (h - s - 2)))

        mid = 16 * cs + 16 * (4 ** (h - s + 1) - 4)
        for j in range(0, h - s - 2):
            S.append(C2(40 * 4 ** j - 4, 12, 8,
                        16 ** j * 13 ** (rho - 2 * s - 2 * j - 6) * cen(mid)))
        S.append(C2(40 * 4 ** (h - s - 2) - 4, mid, 8, 16 ** (h - s - 2)))
        ci.append(mid + 4 * (40 * 4 ** (h - s - 2) - 4))
    S.append(C1(12, 4 * ci[h - 1] + 24, 8, 1))

    return lse(T), lse(S), lse([lse(T), lse(S)]), len(T), len(S)


def A1_even(N):
    r"""
    Evaluate the Appendix A.1 reduction costs for even N.

    The returned values are the backward, forward, and total reduction costs,
    followed by the numbers of terms in the two sums.
    """
    if N % 2 == 1:
        raise ValueError("A1_even requires even N.")
    if N < 2:
        raise ValueError("The Vision MITM model is evaluated here for N >= 2.")

    # The case N = 2 is evaluated directly, as specified in Appendix A.1.
    if N == 2:
        T = []
        S = [C1(4, 4, 4, 1)]
        return lse(T), lse(S), lse([lse(T), lse(S)]), len(T), len(S)

    rho = N
    ell = rho // 2

    def cen(c):
        return (c + 1) ** 2

    # Backward (output-side) reduction.
    # For even N, the backward chain reaches the middle through a relation
    # with delta = 8.
    T = [C2(8 * 4 ** j, 0, 8, 16 ** j) for j in range(0, ell - 1)]

    c = [None, 2 * 4 ** ell]                              # c^out_1
    for s in range(1, ell - 1):                          # s = 1 .. ell-2
        cs = c[s]
        T.append(C1(12, 36, 8, 13 ** (rho - 2 * s - 4) * cen(4 * cs)))
        for j in range(1, ell - s - 1):
            T.append(C2(4 ** (j + 2) - 4, 12, 8,
                        4 * 16 ** (j - 1) * 13 ** (rho - 2 * s - 2 * j - 4)
                        * cen(4 * cs)))
        T.append(C2(4 ** (ell - s + 1) - 4, 4 * cs, 8,
                    4 * 16 ** (ell - s - 2)))

        mid = 16 * cs + 16 * (4 ** (ell - s + 1) - 4)
        for j in range(0, ell - s - 2):
            T.append(C2(40 * 4 ** j - 4, 12, 8,
                        16 ** j * 13 ** (rho - 2 * s - 2 * j - 6) * cen(mid)))
        T.append(C2(40 * 4 ** (ell - s - 2) - 4, mid, 8,
                    16 ** (ell - s - 2)))
        c.append(mid + 4 * (40 * 4 ** (ell - s - 2) - 4))
    T.append(C1(12, 4 * c[ell - 1] + 24, 8, 1))

    # Forward (input-side) reduction.
    # The final relation of the forward chain is the middle relation, with delta = 4.
    S = [C1(4, 8, 8, 1)]
    S += [C2(4 ** j, 0, 8, 4 * 16 ** (j - 2)) for j in range(2, ell)]
    S.append(C2(4 ** ell, 0, 4, 4 * 16 ** (ell - 2)))

    c1 = 2 * 4 ** ell                                    # c^in_1
    for j in range(0, ell - 2):
        S.append(C2(40 * 4 ** j - 4, 12, 8,
                    16 ** j * 13 ** (rho - 2 * j - 6) * cen(4 * c1)))
    S.append(C2(40 * 4 ** (ell - 2) - 4, 4 * c1, 4,
                16 ** (ell - 2)))

    ci = [None, c1, 4 * c1 + 2 * (40 * 4 ** (ell - 2) - 4)]
    for s in range(2, ell):                              # s = 2 .. ell-1
        cs = ci[s]
        S.append(C1(12, 36, 8, 13 ** (rho - 2 * s - 2) * cen(4 * cs)))
        for j in range(1, ell - s):
            S.append(C2(4 ** (j + 2) - 4, 12, 8,
                        4 * 16 ** (j - 1) * 13 ** (rho - 2 * s - 2 * j - 2)
                        * cen(4 * cs)))
        S.append(C2(4 ** (ell - s + 2) - 4, 4 * cs, 4,
                    4 * 16 ** (ell - s - 1)))

        mid = 16 * cs + 8 * (4 ** (ell - s + 2) - 4)
        for j in range(0, ell - s - 1):
            S.append(C2(40 * 4 ** j - 4, 12, 8,
                        16 ** j * 13 ** (rho - 2 * s - 2 * j - 4) * cen(mid)))
        S.append(C2(40 * 4 ** (ell - s - 1) - 4, mid, 4,
                    16 ** (ell - s - 1)))
        ci.append(mid + 2 * (40 * 4 ** (ell - s - 1) - 4))
    S.append(C1(12, 4 * ci[ell] + 12, 4, 1))

    return lse(T), lse(S), lse([lse(T), lse(S)]), len(T), len(S)

# ======================================================================
# Degree propagation and intermediate resultant costs
# ======================================================================

class Side(object):
    r"""
    Degree data for one elimination chain.

    The states are indexed from the boundary toward the middle.  The value
    delta[i] is 8 for a regular relation and 4 for a middle relation.
    """

    def __init__(self, L, delta, deg0, c_res=17):
        self.L = L
        self.delta = dict(delta)
        self.deg = dict((i, 0) for i in range(1, L + 1))
        self.deg.update(deg0)
        self.nv = dict((i, 2) for i in range(1, L + 1))
        self.cdeg = 0
        self.T_sub = []
        self.T_res_iter = []
        self.nsub = 0
        self.nres = 0
        self.c_res = c_res

    def _B(self, i, adj_is_central):
        b = 1
        for j in range(1, self.L + 1):
            if j == i or ((not adj_is_central) and j == i + 1):
                continue
            b *= (self.deg[j] + 1) ** self.nv[j]
        if not adj_is_central:
            b *= (self.cdeg + 1) ** 2
        return b

    def sweep(self, start):
        r"""Apply the quartic reductions from state `start` toward the middle."""
        for i in range(start, self.L + 1):
            m = self.nv[i]
            if m == 0:
                continue
            d = self.deg[i]
            adjc = (i == self.L)
            e = self.cdeg if adjc else self.deg[i + 1]
            dl = self.delta[i]
            B = self._B(i, adjc)
            self.T_sub.append((C2 if m == 2 else C1)(d, e, dl, B))
            self.nsub += 1
            inc = m * dl * d // 4
            if adjc:
                self.cdeg += inc
            else:
                self.deg[i + 1] += inc
            self.deg[i] = 3

    def resultant(self, s):
        r"""Apply one intermediate resultant at state s using Lemma 2."""
        t = min(self.deg[s], 3)
        adjc = (s == self.L)
        dfx = self.delta[s]
        S = 1
        for j in range(1, self.L + 1):
            n = self.nv[j] - (1 if j == s else 0)
            if n <= 0:
                continue
            df = dfx if ((not adjc) and j == s + 1) else 0
            S *= (4 * self.deg[j] + t * df + 1) ** n
        S *= (4 * self.cdeg + t * (dfx if adjc else 0) + 1) ** 2
        self.T_res_iter.append(math.log2(self.c_res) + lgM(S))
        self.nres += 1
        for j in range(1, self.L + 1):
            df = dfx if ((not adjc) and j == s + 1) else 0
            self.deg[j] = 4 * self.deg[j] + t * df
        self.cdeg = 4 * self.cdeg + t * (dfx if adjc else 0)
        self.nv[s] -= 1


def run_sides(N, c_res=17):
    r"""
    Compute the backward and forward reduction and resultant costs.

    For each chain, also return the final middle-state degree bound and the
    numbers of reductions and resultants.
    """
    rho = 2 * ((N + 1) // 2)
    h = rho // 2
    odd = (N % 2 == 1)

    # Backward (output-side) reduction.
    Lo = h - 1
    if Lo <= 0:
        out = (float('-inf'), float('-inf'), 8, 0, 0)
    else:
        delo = dict((i, 8) for i in range(1, Lo + 1))
        if odd:
            delo[Lo] = 4                             # Middle relation.
        Sd = Side(Lo, delo, {1: 8}, c_res)
        Sd.sweep(1)
        for s in range(1, Lo + 1):
            Sd.resultant(s)
            Sd.sweep(s)
            Sd.resultant(s)
            if s < Lo:
                Sd.sweep(s + 1)
        out = (lse(Sd.T_sub), lse(Sd.T_res_iter), Sd.cdeg, Sd.nsub, Sd.nres)

    # Forward (input-side) reduction.
    Li = h - 1 if odd else h
    deli = dict((i, 8) for i in range(1, Li + 1))
    if not odd:
        deli[Li] = 4                                 # Middle relation.
    Td = Side(Li, deli, {1: 1}, c_res)
    Td.resultant(1)
    Td.sweep(1)
    Td.resultant(1)
    for s in range(2, Li + 1):
        Td.sweep(s)
        Td.resultant(s)
        Td.sweep(s)
        Td.resultant(s)
    inn = (lse(Td.T_sub), lse(Td.T_res_iter), Td.cdeg, Td.nsub, Td.nres)
    return out, inn

# ======================================================================
# Complexity evaluation
# ======================================================================

# Parameters for the 128-bit Vision instances in Table 1.
# Set FIELD_DEGREE = 256 for the corresponding 256-bit instances.
FIELD_DEGREE = 128
OMEGA = 2.0
C_RES = 17


def vision_complexity(N):
    r"""
    Compute the five complexity terms of Section 3.3 for N attacked rounds.

    All returned values are base-2 logarithms of field-operation counts.
    """
    n = FIELD_DEGREE

    (T_sub_out_sim, T_res_iter_out, d_out, _, _), \
    (T_sub_in_sim, T_res_iter_in, d_in, _, _) = run_sides(N, C_RES)

    # Reduction cost from the explicit sums in Appendix A.1.
    A1 = A1_odd(N) if N % 2 else A1_even(N)
    T_sub = A1[2]

    # Intermediate resultant cost.
    T_res_iter = lse([T_res_iter_out, T_res_iter_in])

    # Final bivariate resultant cost.
    d_P = 2 * max(d_out, d_in)
    d_Q = 2 * min(d_out, d_in)
    log_d_P = lg(d_P)
    log_d_Q = lg(d_Q)

    T_res_biv = lse([
        lgM(d_P * d_P),
        log_d_P
        + (2 - 1.0 / OMEGA) * log_d_Q
        + 2 * math.log2(log_d_P)
        + math.log2(math.log2(log_d_P))
    ])

    # Univariate root-finding cost.
    d_uni = d_P * d_Q
    T_uni = lgU(d_uni, n)

    # Total complexity.
    T_total = lse([T_sub, T_res_iter, T_res_biv, T_uni])

    return {
        "N": N,
        "T_sub": T_sub,
        "T_res_iter": T_res_iter,
        "T_res_biv": T_res_biv,
        "T_uni": T_uni,
        "T_total": T_total,
    }


def report(N):
    r"""Print the five complexity terms for a fixed number of attacked rounds."""
    r = vision_complexity(N)
    print("N = %d" % N)
    print("log2 T_sub      = %.3f" % r["T_sub"])
    print("log2 T_res,iter = %.3f" % r["T_res_iter"])
    print("log2 T_res,biv  = %.3f" % r["T_res_biv"])
    print("log2 T_uni      = %.3f" % r["T_uni"])
    print("log2 T_total    = %.3f" % r["T_total"])


def reviewer_table():
    r"""Print the complexity values for N = 10,...,15."""
    print(" N   log2 T_sub   log2 T_res,iter   log2 T_res,biv   log2 T_uni   log2 T_total")
    for N in range(10, 16):
        r = vision_complexity(N)
        print("%2d   %10.3f   %15.3f   %14.3f   %10.3f   %12.3f"
              % (N, r["T_sub"], r["T_res_iter"], r["T_res_biv"],
                 r["T_uni"], r["T_total"]))


if __name__ == "__main__":
    reviewer_table()
