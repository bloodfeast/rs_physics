// Emission on the device: `for_each_in_burst` (src/particles/particle_effects.rs), draw
// for draw and rounding for rounding, so a burst expanded here gives the bits the CPU's
// `ParticleEffects::emit` gives from the same seed.
//
// The including module declares `records: array<u32>` (read-only storage holding the
// tables below at the word offsets it passes in) and `fn emit_zero() -> u32`, which must
// return 0 from memory the compiler cannot see into (a uniform).
//
// Where WGSL could differ from the CPU, and what pins each one:
//
// - Contraction. Vulkan lets a driver fuse a multiply and the add after it into one
//   rounding unless the SPIR-V carries NoContraction, which naga does not emit; Rust
//   never fuses. Every product that feeds an add goes through `rounded`, an XOR with a
//   zero the compiler cannot prove is zero, so the product is materialised as an f32
//   first. Products that feed only a multiply or a store need nothing.
// - Add, subtract and multiply are correctly rounded on Vulkan, as on the CPU.
// - Division (2.5 ulp on Vulkan) and square root (inherited from inversesqrt) are not.
//   `div_rn` and `sqrt_rn` below are correctly rounded: the device's own result as an
//   estimate, settled by exact integer comparisons (long division and a digit-by-digit
//   root as the fallback).
// - `x / 2^24` in `unit` is a multiply by 2^-24 here: exact either way, so the same.
// - u32 to f32: exact below 2^24, which is all `unit` and `sin_cos_turn` convert.
// - sin and cos: `sin_cos_turn` in i32 arithmetic and the same table as the CPU's.
// - Subnormals: a device may flush them. None arises from finite burst ranges of normal
//   magnitude; `sqrt_rn` and `div_rn` read a subnormal operand as zero (documented on
//   `GpuParticlePool::emit`).
// - Literals: 1e-6 and f32::EPSILON are written as their bits.

// One byte-sliced 32x32 GF(2) matrix: four tables of 256 images.
const JUMP_TABLE_WORDS: u32 = 1024u;
// Jump tables per hex digit position: digits 1 to 15.
const DIGIT_TABLES: u32 = 15u;
// round(2^24 sin), 2^12 intervals of a quarter turn, the end knot twice.
const SINE_WORDS: u32 = 4098u;
const QUARTER_BITS: u32 = 22u;
const SINE_FRACTION_BITS: u32 = 10u;
const TWO_TO_MINUS_24: u32 = 0x33800000u;
const F32_EPSILON: u32 = 0x34000000u;
// 1e-6f32: below it the direction is straight up, as `EffectRng::hemisphere`.
const DIRECTION_MIN_LEN: u32 = 0x358637bdu;

// Words of a staged record: position 3, velocity 3, remaining, lifetime, size, class.
const EMIT_RECORD_WORDS: u32 = 10u;
// Words of a segment-table entry: first particle, count, kind, payload offset.
const SEGMENT_WORDS: u32 = 4u;
const SEGMENT_RECORDS: u32 = 0u;

// A product (or any value) forced to its own f32 rounding before an add reads it.
fn rounded(x: f32) -> f32 {
    return bitcast<f32>(bitcast<u32>(x) ^ emit_zero());
}

fn xorshift(x0: u32) -> u32 {
    var x = x0;
    x = x ^ (x << 13u);
    x = x ^ (x >> 17u);
    x = x ^ (x << 5u);
    return x;
}

// `state` advanced `5 i` draws: the product of the stride tables (M^(5 d 16^k),
// byte-sliced at `tables`, `digits` positions of 15) for the non-zero hex digits of `i`.
fn jump_stride(state: u32, i: u32, tables: u32, digits: u32) -> u32 {
    var x = state;
    for (var k = 0u; k < digits; k = k + 1u) {
        let rest = i >> (4u * k);
        if (rest == 0u) {
            break;
        }
        let d = rest & 15u;
        if (d != 0u) {
            let t = tables + (k * DIGIT_TABLES + d - 1u) * JUMP_TABLE_WORDS;
            x = records[t + (x & 255u)]
                ^ records[t + 256u + ((x >> 8u) & 255u)]
                ^ records[t + 512u + ((x >> 16u) & 255u)]
                ^ records[t + 768u + (x >> 24u)];
        }
    }
    return x;
}

// `EffectRng::unit` of a drawn value.
fn unit_of(x: u32) -> f32 {
    return f32(x >> 8u) * bitcast<f32>(TWO_TO_MINUS_24);
}

// `EffectRng::range` of a drawn value.
fn range_of(lo: f32, hi: f32, x: u32) -> f32 {
    return lo + rounded((hi - lo) * unit_of(x));
}

fn quarter_sine(p: u32, sine: u32) -> i32 {
    let i = p >> SINE_FRACTION_BITS;
    let f = i32(p & ((1u << SINE_FRACTION_BITS) - 1u));
    let a = bitcast<i32>(records[sine + i]);
    let b = bitcast<i32>(records[sine + i + 1u]);
    return a + (((b - a) * f + (1 << (SINE_FRACTION_BITS - 1u))) >> SINE_FRACTION_BITS);
}

// `sin_cos_turn` (src/particles/turn_trig.rs): (sin, cos) of `turn / 2^24` of a turn.
fn sin_cos_turn(turn: u32, sine: u32) -> vec2<f32> {
    let q = (turn >> QUARTER_BITS) & 3u;
    let p = turn & ((1u << QUARTER_BITS) - 1u);
    let s = quarter_sine(p, sine);
    let c = quarter_sine((1u << QUARTER_BITS) - p, sine);
    let swap = -i32(q & 1u);
    let a = s ^ ((s ^ c) & swap);
    let b = c ^ ((s ^ c) & swap);
    let neg_s = -i32(q >> 1u);
    let neg_c = -i32((q ^ (q >> 1u)) & 1u);
    let si = (a ^ neg_s) - neg_s;
    let ci = (b ^ neg_c) - neg_c;
    let scale = bitcast<f32>(TWO_TO_MINUS_24);
    return vec2<f32>(f32(si) * scale, f32(ci) * scale);
}

// `a * b` for `a`, `b` below 2^26, exactly, as (high word, low word): 16-bit limbs.
fn mul_wide(a: u32, b: u32) -> vec2<u32> {
    let ah = a >> 16u;
    let al = a & 0xffffu;
    let bh = b >> 16u;
    let bl = b & 0xffffu;
    let low = al * bl;
    // Each cross term is below 2^26, so the sum fits.
    let mid = ah * bl + al * bh;
    let lo = low + (mid << 16u);
    let carry = select(0u, 1u, lo < low);
    return vec2<u32>(ah * bh + (mid >> 16u) + carry, lo);
}

// `a > b` for 64-bit values as (high, low).
fn wide_greater(a: vec2<u32>, b: vec2<u32>) -> bool {
    return a.x > b.x || (a.x == b.x && a.y > b.y);
}

// The correctly rounded square root's significand, digit by digit: `sqrt(M) / 2`
// rounded to nearest, M = `hi lo` (50 bits). The fallback of `sqrt_rn`.
fn sqrt_digits(hi: u32, lo: u32) -> u32 {
    var rem = 0u;
    var root = 0u;
    for (var j = 0u; j < 25u; j = j + 1u) {
        let k = 48u - 2u * j;
        var pair: u32;
        if (k >= 32u) {
            pair = (hi >> (k - 32u)) & 3u;
        } else {
            pair = (lo >> k) & 3u;
        }
        rem = (rem << 2u) | pair;
        let trial = (root << 2u) | 1u;
        root = root << 1u;
        if (rem >= trial) {
            rem = rem - trial;
            root = root | 1u;
        }
    }
    // No square root of an f32 lies on a rounding midpoint, so the remainder decides
    // only which side: round half up is round to nearest here.
    return (root >> 1u) + (root & 1u);
}

// The correctly rounded square root of a non-negative finite `x`: what the CPU's
// `f32::sqrt` (IEEE 754) returns. A subnormal reads as zero.
//
// x = m 2^p; the significand is shifted so the remaining exponent is even, giving the
// 50-bit radicand M = m 2^s, whose root's significand is `sqrt(M) / 2` in [2^23, 2^24).
// The rounded significand is the integer c with (2c - 1)^2 < M < (2c + 1)^2 (M is even
// and the bounds odd, so neither is ever equal). The device's own `sqrt` estimates c to
// a few units; six exact 64-bit comparisons against the odd squares round it bracket
// the answer, independently, so the chain is short. If the bracket does not hold (an
// estimate off by more than 2, which no conformant device gives), the digit-by-digit
// root decides. Either way the result is exact; the estimate only picks the path.
fn sqrt_rn(x: f32) -> f32 {
    return sqrt_rn_by(x, false);
}

// `sqrt_rn`, or with `digits` its fallback alone (for the tests).
fn sqrt_rn_by(x: f32, digits: bool) -> f32 {
    let bits = bitcast<u32>(x);
    let e = (bits >> 23u) & 0xffu;
    if (e == 0u) {
        return 0.0;
    }
    let m = (bits & 0x7fffffu) | 0x800000u;
    let p = i32(e) - 150;
    let s = select(26u, 25u, (p & 1) != 0);
    let radicand = vec2<u32>(m >> (32u - s), m << s);
    // sqrt(M) / 2 = sqrt(4m) 2^11 (s = 26) or sqrt(2m) 2^11 (s = 25); 4m and 2m are
    // exact in f32.
    let c0 = u32(sqrt(f32(m << (s - 24u))) * 2048.0);
    var below = 0u;
    for (var j = 0u; j < 6u; j = j + 1u) {
        let u = 2u * (c0 + j - 3u) + 1u;
        below = below + select(0u, 1u, wide_greater(radicand, mul_wide(u, u)));
    }
    var sig: u32;
    if (digits || below == 0u || below == 6u) {
        sig = sqrt_digits(radicand.x, radicand.y);
    } else {
        sig = c0 - 3u + below;
    }
    var exponent = (p - i32(s)) / 2 + 24;
    if (sig == 0x1000000u) {
        sig = 0x800000u;
        exponent = exponent + 1;
    }
    return bitcast<f32>((u32(exponent + 127) << 23u) | (sig & 0x7fffffu));
}

// The correctly rounded quotient significand `ma 2^23 / mb`, by long division: the
// fallback of `div_rn`.
fn div_digits(ma: u32, mb: u32) -> u32 {
    var rem = ma;
    var q = 0u;
    for (var j = 0u; j < 25u; j = j + 1u) {
        q = q << 1u;
        if (rem >= mb) {
            rem = rem - mb;
            q = q | 1u;
        }
        rem = rem << 1u;
    }
    // No quotient of two f32 lies on a rounding midpoint: round half up is to nearest.
    return (q >> 1u) + (q & 1u);
}

// The correctly rounded quotient `a / b` of a finite `a` and a positive normal `b`: what
// the CPU's `/` (IEEE 754) returns when the quotient is a normal f32. A subnormal `a`
// reads as zero, and a quotient below the normal range is zero (the CPU would give a
// subnormal).
//
// With the significands scaled so ma / mb is in [1, 2), the rounded significand is the
// integer c with (2c - 1) mb < ma 2^24 < (2c + 1) mb (never equal: no quotient of two
// f32 is a midpoint). As `sqrt_rn`: the device's division estimates c, six exact
// comparisons bracket it, and long division decides if they do not.
fn div_rn(a: f32, b: f32) -> f32 {
    return div_rn_by(a, b, false);
}

// `div_rn`, or with `digits` its fallback alone (for the tests).
fn div_rn_by(a: f32, b: f32, digits: bool) -> f32 {
    let ba = bitcast<u32>(a);
    let bb = bitcast<u32>(b);
    let sign = ba & 0x80000000u;
    let ea = (ba >> 23u) & 0xffu;
    let eb = (bb >> 23u) & 0xffu;
    if (ea == 0u) {
        return bitcast<f32>(sign);
    }
    var ma = (ba & 0x7fffffu) | 0x800000u;
    let mb = (bb & 0x7fffffu) | 0x800000u;
    var exponent = i32(ea) - i32(eb);
    if (ma < mb) {
        ma = ma << 1u;
        exponent = exponent - 1;
    }
    // ma and mb are below 2^25, exact in f32.
    let c0 = u32(f32(ma) / f32(mb) * 8388608.0);
    let numerator = vec2<u32>(ma >> 8u, ma << 24u);
    var below = 0u;
    for (var j = 0u; j < 6u; j = j + 1u) {
        let u = 2u * (c0 + j - 3u) + 1u;
        below = below + select(0u, 1u, wide_greater(numerator, mul_wide(u, mb)));
    }
    var sig: u32;
    if (digits || below == 0u || below == 6u) {
        sig = div_digits(ma, mb);
    } else {
        sig = c0 - 3u + below;
    }
    if (sig == 0x1000000u) {
        sig = 0x800000u;
        exponent = exponent + 1;
    }
    let biased = exponent + 127;
    if (biased <= 0) {
        return bitcast<f32>(sign);
    }
    if (biased >= 255) {
        return bitcast<f32>(sign | 0x7f800000u);
    }
    return bitcast<f32>(sign | (u32(biased) << 23u) | (sig & 0x7fffffu));
}

// `EffectRng::hemisphere` from its two draws: the azimuth's and y's.
fn hemisphere_of(azimuth_draw: u32, y_draw: u32, lift: f32, sine: u32) -> vec3<f32> {
    let y = range_of(-1.0, 1.0, y_draw);
    let r = sqrt_rn(max(1.0 - rounded(y * y), 0.0));
    let sc = sin_cos_turn(azimuth_draw >> 8u, sine);
    let d = vec3<f32>(r * sc.y, y + lift, r * sc.x);
    let len = sqrt_rn((rounded(d.x * d.x) + rounded(d.y * d.y)) + rounded(d.z * d.z));
    if (len > bitcast<f32>(DIRECTION_MIN_LEN)) {
        return vec3<f32>(div_rn(d.x, len), div_rn(d.y, len), div_rn(d.z, len));
    }
    return vec3<f32>(0.0, 1.0, 0.0);
}

struct Emitted {
    position: vec3<f32>,
    velocity: vec3<f32>,
    remaining: f32,
    lifetime: f32,
    size: f32,
    class_id: u32,
}

// Particle `j` of the burst whose payload starts at word `o`: its state before its first
// draw is the payload's state advanced `5 j` draws.
//
// Payload: state, origin xyz, class, speed lo hi, lifetime lo hi, size lo hi, lift.
fn burst_particle(o: u32, j: u32, tables: u32, levels: u32, sine: u32) -> Emitted {
    let x0 = jump_stride(records[o], j, tables, levels);
    let x1 = xorshift(x0);
    let x2 = xorshift(x1);
    let x3 = xorshift(x2);
    let x4 = xorshift(x3);
    let x5 = xorshift(x4);
    let w = o + 1u;
    let lift = bitcast<f32>(records[w + 10u]);
    let dir = hemisphere_of(x1, x2, lift, sine);
    let speed = range_of(bitcast<f32>(records[w + 4u]), bitcast<f32>(records[w + 5u]), x3);
    let life = max(
        range_of(bitcast<f32>(records[w + 6u]), bitcast<f32>(records[w + 7u]), x4),
        bitcast<f32>(F32_EPSILON),
    );
    let size = range_of(bitcast<f32>(records[w + 8u]), bitcast<f32>(records[w + 9u]), x5);
    var out: Emitted;
    out.position = vec3<f32>(bitcast<f32>(records[w]), bitcast<f32>(records[w + 1u]), bitcast<f32>(records[w + 2u]));
    out.velocity = vec3<f32>(dir.x * speed, dir.y * speed, dir.z * speed);
    out.remaining = life;
    out.lifetime = life;
    out.size = size;
    out.class_id = records[w + 3u];
    return out;
}

// A staged record at word `b`.
fn record_particle(b: u32) -> Emitted {
    var out: Emitted;
    out.position = vec3<f32>(bitcast<f32>(records[b]), bitcast<f32>(records[b + 1u]), bitcast<f32>(records[b + 2u]));
    out.velocity = vec3<f32>(bitcast<f32>(records[b + 3u]), bitcast<f32>(records[b + 4u]), bitcast<f32>(records[b + 5u]));
    out.remaining = bitcast<f32>(records[b + 6u]);
    out.lifetime = bitcast<f32>(records[b + 7u]);
    out.size = bitcast<f32>(records[b + 8u]);
    out.class_id = records[b + 9u];
    return out;
}

// The frame's particle `r` (0 the oldest): the segment holding it found by binary search
// over the segment table at word 0, then a record read or a burst expanded.
fn emitted(r: u32, segments: u32, tables: u32, levels: u32, sine: u32) -> Emitted {
    var lo = 0u;
    var hi = segments;
    while (hi - lo > 1u) {
        let mid = (lo + hi) / 2u;
        if (records[mid * SEGMENT_WORDS] <= r) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let e = lo * SEGMENT_WORDS;
    let j = r - records[e];
    let o = segments * SEGMENT_WORDS + records[e + 3u];
    if (records[e + 2u] == SEGMENT_RECORDS) {
        return record_particle(o + j * EMIT_RECORD_WORDS);
    }
    return burst_particle(o, j, tables, levels, sine);
}
