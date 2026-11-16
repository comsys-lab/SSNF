"""Independent integer/rational numeric oracle; no GPU or third-party dependency."""
import math
import struct


def fp32_exact(number: int, exponent: int, zero_sign: int = 0) -> int:
    """Round number * 2**exponent to binary32 RNE, then flush subnormals."""
    if not number:
        return zero_sign << 31
    sign = int(number < 0)
    mag = abs(number)
    msb = mag.bit_length() - 1
    shift = msb - 23
    if shift > 0:
        sig, rem = divmod(mag, 1 << shift)
        half = 1 << (shift - 1)
        sig += rem > half or (rem == half and sig & 1)
        if sig == 1 << 24:
            sig >>= 1
            msb += 1
    else:
        sig = mag << -shift
    raw_exp = msb + exponent + 127
    if raw_exp <= 0:
        return sign << 31
    if raw_exp >= 255:
        return (sign << 31) | 0x7F800000
    return (sign << 31) | (raw_exp << 23) | (sig & 0x7FFFFF)


def fp32_parts(bits: int):
    exp = (bits >> 23) & 255
    if exp == 0:
        return 0, 0
    if exp == 255:
        raise ValueError("special floating-point value")
    coeff = (1 << 23) | (bits & 0x7FFFFF)
    return (-coeff if bits >> 31 else coeff), exp - 150


def fp32_add(a: int, b: int) -> int:
    ea, eb = (a >> 23) & 255, (b >> 23) & 255
    if (ea == 255 and a & 0x7FFFFF) or (eb == 255 and b & 0x7FFFFF):
        return 0x7FC00000
    if ea == eb == 255 and (a ^ b) >> 31:
        return 0x7FC00000
    if ea == 255:
        return a
    if eb == 255:
        return b
    ca, xa = fp32_parts(a)
    cb, xb = fp32_parts(b)
    exp = min(xa, xb)
    return fp32_exact((ca << (xa - exp)) + (cb << (xb - exp)), exp,
                      (a & b) >> 31 if ca == cb == 0 else 0)


def bf16_product(a: int, b: int) -> int:
    ea, eb = (a >> 7) & 255, (b >> 7) & 255
    sign = (a ^ b) >> 15
    if (ea == 255 and a & 127) or (eb == 255 and b & 127):
        return 0x7FC00000
    if (ea == 255 and eb == 0) or (eb == 255 and ea == 0):
        return 0x7FC00000
    if ea == 255 or eb == 255:
        return (sign << 31) | 0x7F800000
    if ea == 0 or eb == 0:
        return sign << 31
    number = (128 + (a & 127)) * (128 + (b & 127))
    return fp32_exact(-number if sign else number, ea + eb - 268)


def bf16_tree(a, b) -> int:
    values = [bf16_product(x, y) for x, y in zip(a, b)]
    assert len(values) == 32
    while len(values) > 1:
        values = [fp32_add(values[i], values[i + 1]) for i in range(0, len(values), 2)]
    return values[0]


def to_bf16(value: float) -> int:
    bits = struct.unpack('<I', struct.pack('<f', value))[0]
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 0xFFFF


def bf16_float(bits: int) -> float:
    return struct.unpack('<f', struct.pack('<I', bits << 16))[0]


def step_exponent(values):
    largest = max(map(abs, values))
    return max(-127, min(127, math.floor(math.log2(largest)) - 6)) if largest else 0


def ssnf_quantize(values):
    e = step_exponent(values)
    scaled = [math.ldexp(v, -e) for v in values]
    upper, shared = [], []
    for g in range(4):
        source = scaled[g * 8:(g + 1) * 8]
        candidates = []
        for low in range(-4, 4):
            high = [max(-15, min(15, round((v - low) / 8))) for v in source]
            error = sum((h * 8 + low - v) ** 2 for h, v in zip(high, source))
            candidates.append((error, low, high))
        _, low, high = min(candidates, key=lambda c: (c[0], c[1]))
        upper.extend(high)
        shared.append(low)
    return e, upper, shared


def ssnf_operand(e, upper, shared):
    assert len(upper) == 32 and len(shared) == 4
    assert all(-15 <= h <= 15 for h in upper)
    assert all(-4 <= l <= 3 for l in shared)
    record = e + 127
    for i, h in enumerate(upper):
        record |= (h & 31) << (8 + i * 5)
    for g, low in enumerate(shared):
        record |= (low & 7) << (168 + g * 3)
    return record.to_bytes(64, 'little')


def ssnf_dot(a, b):
    ea, ha, la = a
    eb, hb, lb = b
    # The oracle reconstructs full integers. The RTL uses four grouped sums.
    qa = [8 * h + la[i // 8] for i, h in enumerate(ha)]
    qb = [8 * h + lb[i // 8] for i, h in enumerate(hb)]
    total = sum(x * y for x, y in zip(qa, qb))
    return fp32_exact(total, ea + eb), total


def mxint8_quantize(values):
    step = step_exponent(values)
    codes = [max(-127, min(127, round(math.ldexp(x, -step)))) for x in values]
    # OCP raw block scale exponent E, element has implicit factor 2**-6.
    e = step + 6
    assert -127 <= e <= 127
    return e, codes


def mxint8_operand(e, codes):
    return bytes([e + 127]) + bytes(x & 255 for x in codes) + bytes(31)


def mxint8_dot(a, b):
    total = sum(x * y for x, y in zip(a[1], b[1]))
    return fp32_exact(total, a[0] + b[0] - 12), total


def bf16_operand(values):
    return struct.pack('<32H', *values)
