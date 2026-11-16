"""Exact (rational) verification of every identity of the SSNF dot-product decomposition (paper Figure 2): every claim is checked with fractions.Fraction, so a pass means the decomposition is an algebraic identity, not a numerical approximation.
Checks (statement numbers follow dotproduct.tex): 1 reconstruction range (Props 1-2), 2-4 four-term dot product for identical / nested / general partitions (Prop 3, Cor 1, Prop 4), 5-6 mixed-precision and PV forms (A.4), 7 l_b = 0 MX reduction (Cor 2), plus the A.6 worked example.
Run: python verify_dot.py
"""
from __future__ import annotations

from fractions import Fraction as F
import random

random.seed(0)

BLK = 32


# ---------------------------------------------------------------- field sampling
def sample(n: int, blk: int, hb: int, l_b: int, g_size: int) -> dict[str, int | list[int]]:
    """Random SSNF operand of n values: per-block scale exponent, per-value h, per-group l."""
    hmax = 2 ** (hb - 1) - 1
    lmin, lmax = (-(2 ** (l_b - 1)), 2 ** (l_b - 1) - 1) if l_b > 0 else (0, 0)
    nblk = n // blk
    e = [random.randint(-8, 8) for _ in range(nblk)]
    h = [random.randint(-hmax, hmax) for _ in range(n)]
    l = [random.randint(lmin, lmax) for _ in range(n // g_size)]
    return dict(n=n, blk=blk, hb=hb, l_b=l_b, g_size=g_size, e=e, h=h, l=l)


def recon(o: dict[str, int | list[int]]) -> list[F]:
    """Dequantized values: x_i = S_b * (h_i * 2^lb + l_g(i))."""
    out = []
    for i in range(o["n"]):
        S = F(2) ** o["e"][i // o["blk"]]
        out.append(S * (o["h"][i] * 2 ** o["l_b"] + o["l"][i // o["g_size"]]))
    return out


def word(o: dict[str, int | list[int]], i: int) -> int:
    """Integer reconstruction word of value i: h_i * 2^lb + l_g(i)."""
    return o["h"][i] * 2 ** o["l_b"] + o["l"][i // o["g_size"]]


# ---------------------------------------------------------------- 1. range
def check_range() -> None:
    """Verify Propositions 1-2: reconstruction words stay in int8 and hit the tight bound."""
    for hb, l_b in [(5, 3), (4, 4), (6, 2), (7, 1)]:
        o = sample(4096, BLK, hb, l_b, 32)
        w = [word(o, i) for i in range(o["n"])]
        assert min(w) >= -128 and max(w) <= 127, (hb, l_b, min(w), max(w))
        # tight bound of Proposition 1
        lo, hi = -(2 ** (hb + l_b - 1)) + 2 ** (l_b - 1), 2 ** (hb + l_b - 1) - 2 ** (l_b - 1) - 1
        assert lo == -(2 ** (hb - 1) - 1) * 2 ** l_b - 2 ** (l_b - 1)
        assert hi == (2 ** (hb - 1) - 1) * 2 ** l_b + 2 ** (l_b - 1) - 1
        assert min(w) >= lo and max(w) <= hi
        # an asymmetric unshared field would leave int8
        assert -(2 ** (hb - 1)) * 2 ** l_b - 2 ** (l_b - 1) < -128
    print("1. reconstruction range               OK")


# ---------------------------------------------------------------- 2. same partition
def dot_ref(a: dict[str, int | list[int]], b: dict[str, int | list[int]]) -> F:
    """Reference dot product of the dequantized operands."""
    return sum(x * y for x, y in zip(recon(a), recon(b)))


def dot_4term(a: dict[str, int | list[int]], b: dict[str, int | list[int]]) -> F:
    """Proposition 3: identical group partitions."""
    assert a["g_size"] == b["g_size"] and a["blk"] == b["blk"] and a["n"] == b["n"]
    blk, g_size, n = a["blk"], a["g_size"], a["n"]
    la, lb_ = a["l_b"], b["l_b"]
    total = F(0)
    for blk_id in range(n // blk):
        s, t = blk_id * blk, (blk_id + 1) * blk
        uu = sum(a["h"][i] * b["h"][i] for i in range(s, t))
        us = sum(b["l"][g] * sum(a["h"][i] for i in range(g * g_size, (g + 1) * g_size))
                 for g in range(s // g_size, t // g_size))
        su = sum(a["l"][g] * sum(b["h"][i] for i in range(g * g_size, (g + 1) * g_size))
                 for g in range(s // g_size, t // g_size))
        ss = g_size * sum(a["l"][g] * b["l"][g] for g in range(s // g_size, t // g_size))
        acc = (2 ** (la + lb_)) * uu + (2 ** la) * us + (2 ** lb_) * su + ss
        total += F(2) ** (a["e"][blk_id] + b["e"][blk_id]) * acc
    return total


def check_same_partition() -> None:
    """Verify the four-term dot product against the reference for every common g_size."""
    for g_size in [1, 2, 4, 8, 16, 32]:
        a = sample(256, BLK, 5, 3, g_size)
        b = sample(256, BLK, 4, 4, g_size)
        assert dot_4term(a, b) == dot_ref(a, b), g_size
    print("2. four-term, same partition          OK")


# ---------------------------------------------------------------- 3. nested group sizes
def dot_4term_nested(a: dict[str, int | list[int]], b: dict[str, int | list[int]]) -> F:
    """Corollary 1: gs_b divides gs_a (or vice versa); cross terms use each operand's own partition."""
    blk, n = a["blk"], a["n"]
    la, lb_ = a["l_b"], b["l_b"]
    ga, gb = a["g_size"], b["g_size"]
    total = F(0)
    for blk_id in range(n // blk):
        s, t = blk_id * blk, (blk_id + 1) * blk
        uu = sum(a["h"][i] * b["h"][i] for i in range(s, t))
        # unshared(a) x shared(b): group by b's partition
        us = sum(b["l"][g] * sum(a["h"][i] for i in range(g * gb, (g + 1) * gb))
                 for g in range(s // gb, t // gb))
        # unshared(b) x shared(a): group by a's partition
        su = sum(a["l"][g] * sum(b["h"][i] for i in range(g * ga, (g + 1) * ga))
                 for g in range(s // ga, t // ga))
        # shared x shared: nest the finer partition inside the coarser one
        if gb <= ga:
            ss = sum(a["l"][p] * sum(gb * b["l"][q]
                                     for q in range(p * ga // gb, (p + 1) * ga // gb))
                     for p in range(s // ga, t // ga))
        else:
            ss = sum(b["l"][q] * sum(ga * a["l"][p]
                                     for p in range(q * gb // ga, (q + 1) * gb // ga))
                     for q in range(s // gb, t // gb))
        acc = (2 ** (la + lb_)) * uu + (2 ** la) * us + (2 ** lb_) * su + ss
        total += F(2) ** (a["e"][blk_id] + b["e"][blk_id]) * acc
    return total


def check_nested() -> None:
    """Verify the nested-partition four-term dot product for every (gs_a, gs_b) pair."""
    for ga in [1, 2, 4, 8, 16, 32]:
        for gb in [1, 2, 4, 8, 16, 32]:
            a = sample(256, BLK, 5, 3, ga)
            b = sample(256, BLK, 4, 4, gb)
            assert dot_4term_nested(a, b) == dot_ref(a, b), (ga, gb)
    print("3. four-term, different group sizes   OK")


# ---------------------------------------------------------------- 4. general overlap form
def check_overlap() -> None:
    """Proposition 4: without nesting, |g_a^p intersect g_b^q| replaces the group size."""
    n = 24
    ga, gb = 4, 6  # neither divides the other
    la, lb_ = [random.randint(-4, 3) for _ in range(n // ga)], [random.randint(-4, 3) for _ in range(n // gb)]
    direct = sum(la[i // ga] * lb_[i // gb] for i in range(n))
    ov = F(0)
    for p in range(n // ga):
        for q in range(n // gb):
            cnt = len(set(range(p * ga, (p + 1) * ga)) & set(range(q * gb, (q + 1) * gb)))
            ov += cnt * la[p] * lb_[q]
    assert direct == ov
    print("4. general intersection-count form    OK")


# ---------------------------------------------------------------- 5. mixed precision
def check_mixed() -> None:
    """Verify the two-term mixed-precision form (one BF16 operand, Section A.4)."""
    a = sample(256, BLK, 5, 3, 8)
    y = [F(random.randint(-10 ** 6, 10 ** 6), 10 ** 5) for _ in range(a["n"])]
    ref = sum(x * v for x, v in zip(recon(a), y))
    blk, g_size, lb_ = a["blk"], a["g_size"], a["l_b"]
    total = F(0)
    for blk_id in range(a["n"] // blk):
        s, t = blk_id * blk, (blk_id + 1) * blk
        hy = sum(a["h"][i] * y[i] for i in range(s, t))
        ly = sum(a["l"][g] * sum(y[i] for i in range(g * g_size, (g + 1) * g_size))
                 for g in range(s // g_size, t // g_size))
        total += F(2) ** a["e"][blk_id] * ((2 ** lb_) * hy + ly)
    assert total == ref
    print("5. mixed precision, two-term form     OK")


# ---------------------------------------------------------------- 6. orthogonal group axis
def check_orthogonal() -> None:
    """Section A.4 Values: values are grouped along head_dim, the PV product contracts over tokens."""
    T, D, g_size, lb_, blk = 40, 64, 32, 4, BLK
    nblk = D // blk
    e = [[random.randint(-4, 4) for _ in range(nblk)] for _ in range(T)]
    h = [[random.randint(-7, 7) for _ in range(D)] for _ in range(T)]
    l = [[random.randint(-8, 7) for _ in range(D // g_size)] for _ in range(T)]
    p = [F(random.randint(0, 10 ** 5), 10 ** 5) for _ in range(T)]

    def V(t: int, d: int) -> F:
        return F(2) ** e[t][d // blk] * (h[t][d] * 2 ** lb_ + l[t][d // g_size])

    ref = [sum(p[t] * V(t, d) for t in range(T)) for d in range(D)]

    # decomposed: the shared term is one length-T dot product per (group, block) pair,
    # reused by every d in the group
    shared_dot = {}
    for g in range(D // g_size):
        b = (g * g_size) // blk  # group lies inside one scaling block
        assert ((g + 1) * g_size - 1) // blk == b
        shared_dot[g] = sum(p[t] * F(2) ** e[t][b] * l[t][g] for t in range(T))
    got = []
    for d in range(D):
        uu = sum(p[t] * F(2) ** e[t][d // blk] * h[t][d] for t in range(T))
        got.append((2 ** lb_) * uu + shared_dot[d // g_size])
    assert got == ref
    print("6. orthogonal group axis (PV)         OK")


# ---------------------------------------------------------------- 7. MX reduction
def check_mx() -> None:
    """Corollary 2: l_b = 0 reduces the decomposition to the MX dot product."""
    a = sample(256, BLK, 8, 0, 32)
    b = sample(256, BLK, 8, 0, 32)
    mx = F(0)
    for blk_id in range(a["n"] // BLK):
        s, t = blk_id * BLK, (blk_id + 1) * BLK
        mx += F(2) ** (a["e"][blk_id] + b["e"][blk_id]) * sum(a["h"][i] * b["h"][i] for i in range(s, t))
    assert mx == dot_ref(a, b)
    print("7. l_b = 0 reduces to MX              OK")


# ---------------------------------------------------------------- worked example (A.6)
def check_example() -> None:
    """blk = 4, g_size = 2, h_b = 3, l_b = 2 -- a small worked example."""
    lb_, g_size = 2, 2
    ea, eb = 1, 0
    ha, la = [3, -1, 2, -3], [1, -2]
    hb_, lb2 = [-2, 3, 1, 2], [-1, 2]
    A = [F(2) ** ea * (ha[i] * 2 ** lb_ + la[i // g_size]) for i in range(4)]
    B = [F(2) ** eb * (hb_[i] * 2 ** lb_ + lb2[i // g_size]) for i in range(4)]
    ref = sum(x * y for x, y in zip(A, B))
    uu = sum(ha[i] * hb_[i] for i in range(4))
    us = sum(lb2[g] * sum(ha[i] for i in range(g * g_size, (g + 1) * g_size)) for g in range(2))
    su = sum(la[g] * sum(hb_[i] for i in range(g * g_size, (g + 1) * g_size)) for g in range(2))
    ss = g_size * sum(la[g] * lb2[g] for g in range(2))
    acc = 16 * uu + 4 * us + 4 * su + ss
    got = F(2) ** (ea + eb) * acc
    assert got == ref, (got, ref)
    print("   worked example: A =", [str(v) for v in A], " B =", [str(v) for v in B])
    print("   words A =", [ha[i] * 4 + la[i // 2] for i in range(4)],
          " words B =", [hb_[i] * 4 + lb2[i // 2] for i in range(4)])
    print("   UU =", uu, " US =", us, " SU =", su, " SS =", ss,
          " acc =", acc, " result =", ref)


if __name__ == "__main__":
    check_range()
    check_same_partition()
    check_nested()
    check_overlap()
    check_mixed()
    check_orthogonal()
    check_mx()
    check_example()
    print("all identities verified exactly")
