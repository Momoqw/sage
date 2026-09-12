# XHash8 fiber-based complexity evaluator for CICO-2, CICO-3, and CICO-4
# All returned complexities are log2 numbers of base-field operations.

import math

OMEGA = 2.8
N_AUX = 24

# (n_x, D, nu_0)
PARAMETERS = {
    2: (3, 7009493581824, 6),
    3: (5, 2**39 + 2**23, 6),
    4: (7, 2**37 + 2**29, 6),
}


def log2_int(x):
    x = int(x)
    if x <= 0:
        return float("-inf")
    b = x.bit_length()
    if b > 60:
        return (b - 60) + math.log2(x >> (b - 60))
    return math.log2(x)


def log2_sum(values):
    values = [v for v in values if v > float("-inf")]
    if not values:
        return float("-inf")
    m = max(values)
    return m + math.log2(sum(2.0 ** (v - m) for v in values))


def log2_M(d):
    # M(d) = d log d log log d
    ld = log2_int(d)
    l1 = max(ld, 1.0)
    l2 = max(math.log2(l1), 1.0)
    return ld + math.log2(l1) + math.log2(l2)


def log2_C(s, d_x, n_x):
    # Proposition 2:
    # C(s,d_x) = S_s * s * log(S_s) * log log(S_s),
    # S_s = (d_x+1)^{n_x} * 13^s.
    if s <= 0:
        return float("-inf")
    S_s = (int(d_x) + 1) ** n_x * 13 ** s
    return log2_int(s) + log2_M(S_s)


def log2_T_build(n_x, nu_0):
    # T_build = 336 * C(24, 7^{nu_0}).
    return math.log2(336) + log2_C(N_AUX, 7 ** nu_0, n_x)


def log2_T_aux(k, n_x, D, nu_0, omega=OMEGA):
    # d_x(i) = min(2D-2, 7^{nu_0+i})
    #
    # For each of the k constrained output polynomials, the i-th auxiliary
    # resultant contributes 7^omega coefficient multiplications, and each
    # multiplication is followed by reduction in the remaining auxiliaries.
    resultant_terms = []
    reduction_terms = []

    for i in range(1, N_AUX + 1):
        d_x = min(2 * D - 2, 7 ** (nu_0 + i))
        d_i = 13 ** (N_AUX - i) * (d_x + 1) ** n_x

        resultant_terms.append(
            omega * math.log2(7) + log2_M(d_i)
        )
        reduction_terms.append(
            omega * math.log2(7) + log2_C(N_AUX - i, d_x, n_x)
        )

    T_aux_res = math.log2(k) + log2_sum(resultant_terms)
    T_aux_red = math.log2(k) + log2_sum(reduction_terms)
    T_aux = log2_sum([T_aux_res, T_aux_red])

    return T_aux, T_aux_res, T_aux_red


def log2_T_inp(k, n_x, D, omega=OMEGA):
    d = 2 * D - 1
    log_ordinary = omega * log2_int(2 * D - 2)

    # Product tree for D factors.
    # Let E = 2^{ceil(log2 D)}. Padding to E leaves gives E-1
    # multiplication nodes in the binary product tree.
    # Use integer arithmetic to avoid floating-point ceil(log2(D)).
    ceil_log2_D = (int(D) - 1).bit_length()
    product_tree_mults = 2 ** ceil_log2_D - 1
    log_fiber = log2_int(product_tree_mults)

    # First input elimination.
    n_pairwise = math.comb(k, 2)
    first_coefficient = log2_sum([
        log2_int(n_pairwise) + log_ordinary,
        log2_int(k) + log_fiber,
    ])

    terms = [
        first_coefficient + log2_M(d ** (n_x - 1))
    ]

    # Subsequent input eliminations.
    n_ordinary = math.comb(k + 1, 2) - 1
    later_coefficient = log2_sum([
        log2_int(n_ordinary) + log_ordinary,
        log_fiber,
    ])

    for i in range(1, n_x - 1):
        terms.append(
            later_coefficient + log2_M(d ** i)
        )

    return log2_sum(terms)


def log2_T_uni(k, D):
    # After input elimination there are k(k+1)/2 retained univariate
    # constraints, together with x_1^D-c_1.
    n_gcd_polynomials = math.comb(k + 1, 2) + 1
    ld = math.log2(D)
    return (
        math.log2(n_gcd_polynomials)
        + math.log2(D)
        + 2 * math.log2(ld)
        + math.log2(max(math.log2(ld), 1.0))
    )


def xhash8_complexity(k, omega=OMEGA, verbose=True):
    """
    k = 2, 3, or 4.

    Returns log2(T_build), log2(T_aux), log2(T_inp),
    log2(T_uni), and log2(T_total).
    """
    if k not in PARAMETERS:
        raise ValueError("k must be one of 2, 3, 4.")

    n_x, D, nu_0 = PARAMETERS[k]

    T_build = log2_T_build(n_x, nu_0)
    T_aux, T_aux_res, T_aux_red = log2_T_aux(k, n_x, D, nu_0, omega)
    T_inp = log2_T_inp(k, n_x, D, omega)
    T_uni = log2_T_uni(k, D)
    T_total = log2_sum([T_build, T_aux, T_inp, T_uni])

    ans = {
        "k": k,
        "n_x": n_x,
        "D": D,
        "T_build": T_build,
        "T_aux": T_aux,
        "T_aux_res": T_aux_res,
        "T_aux_red": T_aux_red,
        "T_inp": T_inp,
        "T_uni": T_uni,
        "T_total": T_total,
    }

    if verbose:
        print("XHash8 CICO-%d" % k)
        print("log2 T_build = %.3f" % T_build)
        print("log2 T_aux   = %.3f" % T_aux)
        print("log2 T_inp   = %.3f" % T_inp)
        print("log2 T_uni   = %.3f" % T_uni)
        print("log2 T_total = %.3f" % T_total)

    return ans


def reviewer_table():
    print(" CICO   T_build    T_aux     T_inp     T_uni    T_total")
    for k in (2, 3, 4):
        r = xhash8_complexity(k, verbose=False)
        print("  %d    %8.3f  %8.3f  %8.3f  %8.3f  %8.3f"
              % (k, r["T_build"], r["T_aux"], r["T_inp"],
                 r["T_uni"], r["T_total"]))


if __name__ == "__main__":
    reviewer_table()
