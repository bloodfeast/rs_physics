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
//   `div_rn` and `sqrt_rn` below are correctly rounded in integer arithmetic.
// - `x / 2^24` in `unit` is a multiply by 2^-24 here: exact either way, so the same.
// - u32 to f32: exact below 2^24, which is all `unit` and `sin_cos_turn` convert.
// - sin and cos: `sin_cos_turn` in i32 arithmetic and the same table as the CPU's.
// - Subnormals: a device may flush them. None arises from finite burst ranges of normal
//   magnitude; `sqrt_rn` and `div_rn` read a subnormal operand as zero (documented on
//   `GpuParticlePool::emit`).
// - Literals: 1e-6 and f32::EPSILON are written as their bits.

// One byte-sliced 32x32 GF(2) matrix: four tables of 256 images.
const JUMP_TABLE_WORDS: u32 = 1024u;
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

// `state` advanced `5 i` draws: the product of the stride tables (M^(5 2^b), byte-sliced
// at `tables`) for the set bits of `i`.
fn jump_stride(state: u32, i: u32, tables: u32, levels: u32) -> u32 {
    var x = state;
    for (var b = 0u; b < levels; b = b + 1u) {
        if ((i >> b) == 0u) {
            break;
        }
        if (((i >> b) & 1u) == 1u) {
            let t = tables + b * JUMP_TABLE_WORDS;
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

// The correctly rounded square root of a non-negative finite `x`, digit by digit on the
// significand: what the CPU's `f32::sqrt` (IEEE 754) returns. A subnormal reads as zero.
fn sqrt_rn(x: f32) -> f32 {
    let bits = bitcast<u32>(x);
    let e = (bits >> 23u) & 0xffu;
    if (e == 0u) {
        return 0.0;
    }
    let m = (bits & 0x7fffffu) | 0x800000u;
    // x = m 2^p; shift the significand so the remaining exponent is even, and so the
    // root of the 50-bit radicand M = m 2^s has 25 bits: 24 and a rounding bit.
    let p = i32(e) - 150;
    let s = select(26u, 25u, (p & 1) != 0);
    let hi = m >> (32u - s);
    let lo = m << s;
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
    var sig = root >> 1u;
    var exponent = (p - i32(s)) / 2 + 24;
    if ((root & 1u) == 1u && (rem != 0u || (sig & 1u) == 1u)) {
        sig = sig + 1u;
        if (sig == 0x1000000u) {
            sig = 0x800000u;
            exponent = exponent + 1;
        }
    }
    return bitcast<f32>((u32(exponent + 127) << 23u) | (sig & 0x7fffffu));
}

// The correctly rounded quotient `a / b` of a finite `a` and a positive normal `b`, by
// long division of the significands: what the CPU's `/` (IEEE 754) returns when the
// quotient is a normal f32. A subnormal `a` reads as zero, and a quotient below the
// normal range is zero (the CPU would give a subnormal).
fn div_rn(a: f32, b: f32) -> f32 {
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
    // ma / mb in [1, 2): 25 quotient bits, 24 and a rounding bit; the remainder is sticky.
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
    var sig = q >> 1u;
    if ((q & 1u) == 1u && (rem != 0u || (sig & 1u) == 1u)) {
        sig = sig + 1u;
        if (sig == 0x1000000u) {
            sig = 0x800000u;
            exponent = exponent + 1;
        }
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
    class: u32,
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
    out.class = records[w + 3u];
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
    out.class = records[b + 9u];
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
