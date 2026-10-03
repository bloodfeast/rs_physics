//! Jump-ahead for [`EffectRng`](crate::particles::EffectRng)'s xorshift32.
//!
//! Each of xorshift32's three steps (`x ^= x << 13`, `x ^= x >> 17`, `x ^= x << 5`) is
//! linear over GF(2), so one step is a 32x32 bit matrix `M` and `k` steps are `M^k`. A
//! matrix is stored as its 32 column images (`columns[b]` is the step applied to `1 << b`)
//! and applied by XOR-ing the columns of the state's set bits. For speed it is also kept
//! byte-sliced: four tables of 256 images, one per byte of the state, so applying it is
//! four loads and three XORs.
//!
//! Two sets of powers are kept:
//!
//! - `M^(2^b)` for `b` in `0..32`, behind [`EffectRng::jump`](crate::particles::EffectRng::jump),
//!   for any distance (the period is `2^32 - 1`, so a distance is reduced by it first).
//! - `M^(5 d 16^k)` for each hex digit `d` in `1..16` of position `k`, the emission
//!   stride ([`Burst::DRAWS_PER_PARTICLE`](crate::particles::Burst::DRAWS_PER_PARTICLE))
//!   times every digit value, for the GPU pool: particle `i` of a burst starts `5 i`
//!   draws in, reached with one table per non-zero hex digit of `i`.

use std::sync::OnceLock;

/// A GF(2) 32x32 matrix as its column images.
pub(crate) type Columns = [u32; 32];

/// A matrix byte-sliced: `table[256 k + v]` is the matrix applied to `v << 8 k`.
pub(crate) type ByteTable = [u32; 1024];

/// The xorshift32 step `EffectRng::next_u32` takes.
#[inline(always)]
pub(crate) const fn xorshift_step(mut x: u32) -> u32 {
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    x
}

/// `m` applied to `x`: the XOR of the columns of `x`'s set bits.
pub(crate) const fn apply(m: &Columns, x: u32) -> u32 {
    let mut out = 0;
    let mut b = 0;
    while b < 32 {
        if (x >> b) & 1 == 1 {
            out ^= m[b];
        }
        b += 1;
    }
    out
}

/// `a` after `b`: `(a b) x = a (b x)`.
pub(crate) const fn compose(a: &Columns, b: &Columns) -> Columns {
    let mut out = [0u32; 32];
    let mut k = 0;
    while k < 32 {
        out[k] = apply(a, b[k]);
        k += 1;
    }
    out
}

/// The one-step matrix `M`.
pub(crate) const fn step_matrix() -> Columns {
    let mut out = [0u32; 32];
    let mut b = 0;
    while b < 32 {
        out[b] = xorshift_step(1 << b);
        b += 1;
    }
    out
}

/// `m` byte-sliced, for [`apply_bytes`].
pub(crate) fn byte_table(m: &Columns) -> ByteTable {
    let mut table = [0u32; 1024];
    for k in 0..4 {
        for v in 0..256u32 {
            table[256 * k + v as usize] = apply(m, v << (8 * k));
        }
    }
    table
}

/// A byte-sliced matrix applied to `x`.
#[inline(always)]
pub(crate) fn apply_bytes(t: &ByteTable, x: u32) -> u32 {
    t[(x & 0xFF) as usize]
        ^ t[256 + ((x >> 8) & 0xFF) as usize]
        ^ t[512 + ((x >> 16) & 0xFF) as usize]
        ^ t[768 + (x >> 24) as usize]
}

/// `base^(2^b)` for `b` in `0..levels`, byte-sliced, by repeated squaring.
fn powers(base: Columns, levels: usize) -> Vec<ByteTable> {
    let mut m = base;
    let mut out = Vec::with_capacity(levels);
    for _ in 0..levels {
        out.push(byte_table(&m));
        m = compose(&m, &m);
    }
    out
}

/// The period of xorshift32 with shifts (13, 17, 5) from any non-zero state.
pub(crate) const PERIOD: u64 = (1 << 32) - 1;

/// `M^(2^b)` for `b` in `0..32`, built on first use (128 KB, about a millisecond).
fn step_powers() -> &'static [ByteTable] {
    static POWERS: OnceLock<Vec<ByteTable>> = OnceLock::new();
    POWERS.get_or_init(|| powers(step_matrix(), 32))
}

/// The state `steps` xorshift steps after `x`.
pub(crate) fn jump(mut x: u32, steps: u64) -> u32 {
    let mut k = steps % PERIOD;
    let tables = step_powers();
    let mut b = 0;
    while k != 0 {
        if k & 1 == 1 {
            x = apply_bytes(&tables[b], x);
        }
        k >>= 1;
        b += 1;
    }
    x
}

/// Tables per hex digit position: one for each non-zero digit value.
pub(crate) const DIGIT_TABLES: usize = 15;

/// `M^(stride * d * 16^k)` for `k` in `0..digits` and `d` in `1..16`, byte-sliced and laid
/// end to end, table `15 k + d - 1` at word `1024 (15 k + d - 1)`: the GPU pool's jump
/// tables. A thread reaches particle `i`'s first draw with one table per non-zero hex
/// digit of `i` (4 at most below 65,536), a chain a quarter as long as one table per set
/// bit, for 15 times the tables (60 KB a digit).
pub(crate) fn stride_tables(stride: u32, digits: usize) -> Vec<u32> {
    let one = step_matrix();
    let mut base = one;
    for _ in 1..stride {
        base = compose(&one, &base);
    }
    let mut out = Vec::with_capacity(digits * DIGIT_TABLES * 1024);
    for _ in 0..digits {
        let mut m = base;
        for _ in 0..DIGIT_TABLES {
            out.extend_from_slice(&byte_table(&m));
            m = compose(&base, &m);
        }
        // base^16: the next digit's unit.
        base = m;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sequential(mut x: u32, k: u64) -> u32 {
        for _ in 0..k {
            x = xorshift_step(x);
        }
        x
    }

    /// Jump-ahead by `k` equals `k` sequential steps: a sweep over small distances, the
    /// emission strides, powers of two and their neighbours, and a million.
    #[test]
    fn a_jump_is_that_many_steps() {
        let mut ks: Vec<u64> = (0..70).collect();
        for c in [1u64, 2, 3, 17, 255, 1000] {
            ks.push(5 * c);
        }
        for b in 0..21 {
            ks.extend([(1 << b) - 1, 1 << b, (1 << b) + 1, 5 << b]);
        }
        ks.push(1_000_000);
        ks.push(5_000_000);
        for seed in [1u32, 0xC0FFEE, 0x1234_5678, 0xFFFF_FFFF] {
            for &k in &ks {
                assert_eq!(
                    jump(seed, k),
                    sequential(seed, k),
                    "seed {seed:#x}, {k} steps"
                );
            }
        }
    }

    /// The period is `2^32 - 1`, so a jump by it (or a multiple) is the identity, and a
    /// distance past it wraps.
    #[test]
    fn a_jump_by_the_period_is_the_identity() {
        for seed in [1u32, 0xDEAD_BEEF] {
            assert_eq!(jump(seed, PERIOD), seed);
            assert_eq!(jump(seed, 3 * PERIOD), seed);
            assert_eq!(jump(seed, PERIOD + 7), sequential(seed, 7));
            assert_eq!(jump(seed, u64::MAX), jump(seed, u64::MAX % PERIOD));
        }
    }

    /// The stride tables reach particle `i`'s first draw: `5 i` steps.
    #[test]
    fn the_stride_tables_step_five_per_particle() {
        let digits = 6;
        let t = stride_tables(5, digits);
        assert_eq!(t.len(), 1024 * DIGIT_TABLES * digits);
        let seed = 0xC0FFEE;
        let mut indices: Vec<u32> = (0..40).collect();
        indices.extend([
            63,
            64,
            65,
            255,
            256,
            1000,
            4095,
            4096,
            65_535,
            1_000_000,
            (1 << 24) - 1,
        ]);
        for i in indices {
            let mut x = seed;
            for k in 0..digits {
                let d = ((i >> (4 * k)) & 15) as usize;
                if d != 0 {
                    let at = 1024 * (DIGIT_TABLES * k + d - 1);
                    let table: &ByteTable = t[at..at + 1024].try_into().unwrap();
                    x = apply_bytes(table, x);
                }
            }
            assert_eq!(x, sequential(seed, 5 * i as u64), "particle {i}");
        }
    }
}
