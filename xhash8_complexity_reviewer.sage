# Complexity evaluation for the XHash8 CICO-2, CICO-3, and CICO-4 attacks.
# The cost decomposition and degree bounds follow Section 4.4.
# All reported values are base-2 logarithms of field-operation counts.

import math

OMEGA = 2.8
N_AUX = 24

# Parameters (n_x, D, nu_0) used for the three CICO instances.
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
    # Multiplication cost M(d) from Equation (1).
    ld = log2_int(d)
    l1 = max(ld, 1.0)
    l2 = max(math.log2(l1), 1.0)
    return ld + math.log2(l1) + math.log2(l2)


def log2_C(s, d_x, n_x):
    # Reduction cost C(s,d_x) from Proposition 2, with
    # S_s = (d_x+1)^{n_x} * 13^s.
    if s <= 0:
        return float("-inf")
    S_s = (int(d_x) + 1) ** n_x * 13 ** s
    return log2_int(s) + log2_M(S_s)


def log2_T_build(n_x, nu_0):
    # Section 4.4: T_build = 336 * C(24, 7^{nu_0}).
    return math.log2(336) + log2_C(N_AUX, 7 ** nu_0, n_x)


def log2_T_aux(k, n_x, D, nu_0, omega=OMEGA):
    # At the i-th auxiliary elimination, use
    # d_x(i) = min(2D-2, 7^{nu_0+i}).  The resultant and the subsequent
    # reduction give T_aux,res and T_aux,red, respectively.
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

    # A resultant with x_i^D-c_i is evaluated from Proposition 1 using
    # a binary product tree with at most 2^{ceil(log2 D)}-1 multiplications.
    ceil_log2_D = (int(D) - 1).bit_length()
    product_tree_mults = 2 ** ceil_log2_D - 1
    log_fiber = log2_int(product_tree_mults)

    # Elimination of the first symbolic input.
    n_pairwise = math.comb(k, 2)
    first_coefficient = log2_sum([
        log2_int(n_pairwise) + log_ordinary,
        log2_int(k) + log_fiber,
    ])

    terms = [
        first_coefficient + log2_M(d ** (n_x - 1))
    ]

    # Elimination of the remaining symbolic inputs.
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
    # The final GCD uses the retained univariate constraints together
    # with x_1^D-c_1.
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
    Compute the Section 4.4 complexity terms for CICO-k, with k in {2,3,4}.

    The returned costs are base-2 logarithms of field-operation counts.
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
