//! **L1. The laws, f32 on the GPU against f64 on the CPU.**
//!
//! 10,000 random geometries in 100 random airs run through the shader's own `laws`
//! function (by [`LawProbe`]), and every output is held against the f64 CPU laws in
//! `rs_physics::acoustics`: `spreading_gain`, `barrier_insertion_db`,
//! `foliage_attenuation_db`, `Air::absorption_db_per_m`, `band::fit_lowpass`,
//! `band::cutoff_ceiling`, `Ears::hear` and `doppler_ratio`.
//!
//! **The bound is derived, never typed.** Each output's bound is propagated in this file
//! through the WGSL expression's own operations, from the WGSL f32 accuracy table
//! (WGSL 15.7.4.1): `+ - *` correctly rounded, `/` 2.5 ulp, `exp2` 3 + 2|x| ulp, `log2` 3
//! ulp or 2^-21 absolute on [0.5, 2], `inverseSqrt` 2 ulp, `sqrt` inherited from
//! `1 / inverseSqrt`, `dot` inherited from its sum of products. An input's error enters
//! through the function's variation over the input's interval (every unary function used
//! is monotone there), so no derivative is linearised. WGSL allows an implementation to
//! reassociate, so sums of three or more terms use the order-independent bound
//! `n u sum |terms|`. Three method errors the shader carries on purpose are computed here
//! too: the truncated series for `x coth x` and `x cot x` (their next terms) and
//! Abramowitz and Stegun 4.4.46 for `asin` (its published 2e-8).
#![cfg(feature = "gpu")]

mod acoustics_common;
use acoustics_common::*;

use rs_physics::acoustics::band::{cutoff_ceiling, fit_lowpass, FIT_REFERENCE_HZ, NO_LOWPASS_HZ};
use rs_physics::acoustics::surfaces::{
    barrier_insertion_db, foliage_absorption_db_per_m, foliage_attenuation_db, lit_zone_limit,
    BARRIER_CAP_DB, BARRIER_GRAZING_DB, FOLIAGE_MAX_CREDITED_M,
};
use rs_physics::acoustics::{
    doppler_ratio, spatial::HEAD_RADIUS_M, spreading_gain, Air, Ears, DOPPLER_MAX_CLOSING,
    REFERENCE_M,
};
use rs_physics::gpu::acoustics::oracle::{LawCase, LawProbe};
use rs_physics::gpu::acoustics::*;

/// Unit roundoff of f32.
const U: f64 = 1.0 / (1u64 << 24) as f64;

fn ulp(x: f64) -> f64 {
    let x = x.abs().max(f32::MIN_POSITIVE as f64);
    2f64.powi(x.log2().floor() as i32 - 23)
}

/// A value the GPU computes, as the exact value of the WGSL expression (in f64) and a
/// bound on the GPU's distance from it.
#[derive(Clone, Copy, Debug)]
struct E {
    v: f64,
    e: f64,
}

fn x(v: f32) -> E {
    E {
        v: v as f64,
        e: 0.0,
    }
}

/// A constant the generator rounded to f32.
fn k(v: f64) -> E {
    E {
        v,
        e: ((v as f32) as f64 - v).abs(),
    }
}

/// The variation of a monotone `f` over `[v - e, v + e]`, clipped to its domain.
fn vary(v: f64, e: f64, lo: f64, hi: f64, f: impl Fn(f64) -> f64) -> f64 {
    let at = f(v);
    let a = f((v - e).max(lo));
    let b = f((v + e).min(hi));
    (a - at).abs().max((b - at).abs())
}

impl E {
    fn add(self, o: E) -> E {
        let v = self.v + o.v;
        let p = self.e + o.e;
        E {
            v,
            e: p + 0.5 * ulp(v.abs() + p),
        }
    }
    fn sub(self, o: E) -> E {
        self.add(E { v: -o.v, e: o.e })
    }
    fn mul(self, o: E) -> E {
        let v = self.v * o.v;
        let p = self.v.abs() * o.e + o.v.abs() * self.e + self.e * o.e;
        E {
            v,
            e: p + 0.5 * ulp(v.abs() + p),
        }
    }
    fn div(self, o: E) -> E {
        assert!(o.v.abs() > o.e, "a divisor within its own error of zero");
        let v = self.v / o.v;
        let p = (self.e + v.abs() * o.e) / (o.v.abs() - o.e);
        E {
            v,
            e: p + 2.5 * ulp(v.abs() + p),
        }
    }
    fn inverse_sqrt(self) -> E {
        let v = 1.0 / self.v.sqrt();
        let p = vary(self.v, self.e, f64::MIN_POSITIVE, f64::INFINITY, |a| {
            1.0 / a.sqrt()
        });
        E {
            v,
            e: p + 2.0 * ulp(v + p),
        }
    }
    /// `1 / inverseSqrt(x)`: the reciprocal of a value 2 ulp out, itself 2.5 ulp out.
    fn sqrt(self) -> E {
        if self.v == 0.0 && self.e == 0.0 {
            return E { v: 0.0, e: 0.0 };
        }
        let v = self.v.sqrt();
        let p = vary(self.v, self.e, 0.0, f64::INFINITY, f64::sqrt);
        let rel_w = 2.0 * 2f64.powi(-23);
        let own = (v + p) * rel_w / (1.0 - rel_w) + 2.5 * ulp((v + p) * (1.0 + 2.0 * rel_w));
        E { v, e: p + own }
    }
    fn exp2(self) -> E {
        let v = self.v.exp2();
        let p = vary(self.v, self.e, f64::NEG_INFINITY, f64::INFINITY, f64::exp2);
        E {
            v,
            e: p + (3.0 + 2.0 * (self.v.abs() + self.e)) * ulp(v + p),
        }
    }
    fn log2(self) -> E {
        let v = self.v.log2();
        let p = vary(self.v, self.e, f64::MIN_POSITIVE, f64::INFINITY, f64::log2);
        let (lo, hi) = (self.v - self.e, self.v + self.e);
        let inside = lo >= 0.5 && hi <= 2.0;
        let touches = hi >= 0.5 && lo <= 2.0;
        let ulps = 3.0 * ulp(v.abs() + p);
        let own = if inside {
            2f64.powi(-21)
        } else if touches {
            2f64.powi(-21).max(ulps)
        } else {
            ulps
        };
        E { v, e: p + own }
    }
    fn max_c(self, c: f64) -> E {
        E {
            v: self.v.max(c),
            e: self.e,
        }
    }
    fn min_c(self, c: f64) -> E {
        E {
            v: self.v.min(c),
            e: self.e,
        }
    }
    fn abs(self) -> E {
        E {
            v: self.v.abs(),
            e: self.e,
        }
    }
    fn neg(self) -> E {
        E {
            v: -self.v,
            e: self.e,
        }
    }
}

/// A sum of products, bounded independently of the order it is evaluated in.
fn dot(a: &[E], b: &[E]) -> E {
    let n = a.len() as f64;
    let mut v = 0.0;
    let mut mag = 0.0;
    let mut p = 0.0;
    for (x, y) in a.iter().zip(b) {
        v += x.v * y.v;
        mag += (x.v * y.v).abs() + x.v.abs() * y.e + y.v.abs() * x.e + x.e * y.e;
        p += x.v.abs() * y.e + y.v.abs() * x.e + x.e * y.e;
    }
    let gamma = n * U / (1.0 - n * U);
    E {
        v,
        e: p + gamma * mag,
    }
}

fn sum(terms: &[E]) -> E {
    let ones: Vec<E> = terms.iter().map(|_| E { v: 1.0, e: 0.0 }).collect();
    let d = dot(terms, &ones);
    // No multiplication happened; `dot`'s n counts one rounding per term, which covers the
    // n - 1 additions.
    d
}

fn horner(coeffs: &[f64], z: E) -> E {
    // p = c_last; p = p * z + c_k ... as the WGSL writes it.
    let mut p = k(*coeffs.last().unwrap());
    for c in coeffs.iter().rev().skip(1) {
        p = p.mul(z).add(k(*c));
    }
    p
}

/// The shader's `barrier_db`, bounded, with its method error; branches that the interval
/// straddles are both evaluated and the worse taken. Returns (bound on |gpu - exact law|).
fn barrier_bound(ex: E, lam: E) -> f64 {
    let n = E {
        v: 2.0 * ex.v,
        e: 2.0 * ex.e,
    }
    .div(lam);
    let exact = |n: f64| barrier_insertion_db(n * lam.v / 2.0, lam.v);
    let series = series_coefficients();
    let x0 = (2.0 * std::f64::consts::PI * lit_zone_limit()).sqrt();
    // What the shader computes for a given (computed) Fresnel number, bounded.
    let branch = |n: E, positive: bool| -> (f64, f64) {
        let xx = k(2.0 * std::f64::consts::PI).mul(n.abs()).sqrt();
        let (ratio, method) = if positive {
            if xx.v < 1.0 {
                let c: Vec<f64> = (0..=COTH_TERMS)
                    .map(|i| {
                        if i == 0 {
                            1.0
                        } else if i % 2 == 1 {
                            series[i - 1]
                        } else {
                            -series[i - 1]
                        }
                    })
                    .collect();
                let z = xx.mul(xx);
                let p = horner(&c[1..], z).mul(z).add(E { v: 1.0, e: 0.0 });
                // Alternating and decreasing under 1: the next term bounds the rest.
                let hi = xx.v + xx.e;
                (p, series[COTH_TERMS] * hi.powi(2 * COTH_TERMS as i32 + 2))
            } else {
                let q = k(-2.0 * std::f64::consts::LOG2_E).mul(xx).exp2();
                let one = E { v: 1.0, e: 0.0 };
                (xx.mul(one.add(q)).div(one.sub(q)), 0.0)
            }
        } else {
            if xx.v >= x0 {
                return (0.0, 0.0);
            }
            let c: Vec<f64> = (0..COT_TERMS).map(|i| -series[i]).collect();
            let z = xx.mul(xx);
            let p = horner(&c, z).mul(z).add(E { v: 1.0, e: 0.0 });
            // All terms past the first are negative and fall by at least (x / pi)^2.
            let hi = (xx.v + xx.e).min(x0);
            let r2 = (hi / std::f64::consts::PI).powi(2);
            (
                p,
                series[COT_TERMS] * hi.powi(2 * COT_TERMS as i32 + 2) / (1.0 - r2),
            )
        };
        let db = k(BARRIER_GRAZING_DB).add(k(20.0 * 2f64.log10()).mul(ratio.log2()));
        // d(dB)/d(ratio) = 20 / (ratio ln 10): the method error in the ratio, in dB.
        let method_db = 20.0 * method / ((ratio.v - method).max(1e-6) * std::f64::consts::LN_10);
        let v = db.v.clamp(0.0, BARRIER_CAP_DB);
        (v, db.e + method_db)
    };
    // The exact law's variation over the computed Fresnel number's interval (it is
    // monotone in N), plus the worst own error of any branch the interval reaches.
    let variation = vary(n.v, n.e, f64::NEG_INFINITY, f64::INFINITY, exact);
    let mut own: f64 = 0.0;
    for (lo_in, positive) in [(n.v + n.e > 0.0, true), (n.v - n.e < 0.0, false)] {
        if lo_in {
            let probe = if positive {
                E {
                    v: n.v.max(n.e.max(1e-30)),
                    e: n.e,
                }
            } else {
                E {
                    v: n.v.min(-n.e.max(1e-30)),
                    e: n.e,
                }
            };
            own = own.max(branch(probe, positive).1);
        }
    }
    // The lit branch stops at x >= x0 (rounded to f32), where the law is 0 exactly and
    // the series gives the few-ulp residue of the root: bounded by the law's value there.
    let x0_err = {
        let x0f = (x0 as f32) as f64;
        let n_at = -(x0f * x0f) / (2.0 * std::f64::consts::PI);
        exact(n_at).abs()
    };
    variation + own + x0_err + 1e-12
}

struct Case {
    gpu: SourceResult,
    header: DispatchHeader,
    air: Air,
    curve: Vec<(f32, f32)>,
    case: LawCase,
}

#[derive(Default)]
struct Worst {
    name: &'static str,
    ratio: f64,
    diff: f64,
    bound: f64,
}

impl Worst {
    fn see(&mut self, diff: f64, bound: f64) {
        assert!(
            diff <= bound,
            "{}: |gpu - cpu| = {diff:e} over its bound {bound:e}",
            self.name
        );
        let r = diff / bound;
        if r >= self.ratio {
            *self = Worst {
                name: self.name,
                ratio: r,
                diff,
                bound,
            };
        }
    }
}

#[test]
fn l1_the_f32_laws_hold_to_the_f64_laws_within_the_derived_bound() {
    let Some(gpu) = gpu() else { return };
    let probe = LawProbe::new(&gpu.device);
    let mut rng = Rng(0x5eed_0001);
    let mut cases: Vec<Case> = Vec::with_capacity(10_000);

    for _batch in 0..100 {
        let air = Air::new(
            rng.range(253.15, 313.15),
            rng.range(0.05, 1.0),
            rng.range(80_000.0, 105_000.0),
        )
        .unwrap();
        let yaw = rng.range(0.0, std::f64::consts::TAU);
        let listener = Listener {
            position: [
                rng.range(-500.0, 500.0) as f32,
                rng.range(0.0, 20.0) as f32,
                rng.range(-500.0, 500.0) as f32,
            ],
            forward: [yaw.sin() as f32, 0.0, -yaw.cos() as f32],
            right: [yaw.cos() as f32, 0.0, yaw.sin() as f32],
            velocity: [
                rng.range(-15.0, 15.0) as f32,
                0.0,
                rng.range(-15.0, 15.0) as f32,
            ],
        };
        let curve: Vec<(f32, f32)> = if rng.unit() < 0.5 {
            Vec::new()
        } else {
            let n = 1 + (rng.unit() * 8.0) as usize % 8;
            let mut m = 0.0f32;
            let mut hz = 20_000.0f32;
            (0..n)
                .map(|_| {
                    let p = (m, hz);
                    m += rng.range(5.0, 60.0) as f32;
                    hz *= rng.range(0.3, 1.0) as f32;
                    p
                })
                .collect()
        };
        let header = DispatchHeader::new(&listener, &air, &curve).unwrap();
        let batch: Vec<LawCase> = (0..100)
            .map(|i| {
                let dist = 10f64.powf(rng.range(0.0, 2.5));
                let az = rng.range(0.0, std::f64::consts::TAU);
                let el = rng.range(-0.4, 0.4);
                let l = listener.position;
                let ex = |rng: &mut Rng| -> f32 {
                    match (rng.unit() * 4.0) as u32 {
                        0 => rng.range(-3.0, 3.0) as f32,
                        1 => rng.range(-0.05, 0.05) as f32,
                        2 => (rng.range(-1.0, 1.0) * 1e-4) as f32,
                        _ => -3.4028235e38,
                    }
                };
                LawCase {
                    source: [
                        l[0] + (dist * el.cos() * az.cos()) as f32,
                        l[1] + (dist * el.sin()) as f32,
                        l[2] + (dist * el.cos() * az.sin()) as f32,
                    ],
                    directivity: rng.range(0.0, 2.0) as f32,
                    velocity: [
                        rng.range(-30.0, 30.0) as f32,
                        rng.range(-5.0, 5.0) as f32,
                        rng.range(-30.0, 30.0) as f32,
                    ],
                    tag: i,
                    excess: [ex(&mut rng), ex(&mut rng), ex(&mut rng), ex(&mut rng)],
                    probe_excess: ex(&mut rng),
                    foliage_m: if rng.unit() < 0.3 {
                        0.0
                    } else {
                        rng.range(0.0, 300.0) as f32
                    },
                    pad: [0.0; 2],
                }
            })
            .collect();
        let mut bytes = vec![0u8; 160 + 64 * batch.len()];
        LawProbe::pack(&header, &batch, &mut bytes[..]).unwrap();
        let stage = gpu.stage(&bytes);
        let dst = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 32 * batch.len() as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let mut enc = gpu.encoder();
        probe.encode(
            &mut enc,
            Staged {
                buffer: &stage,
                offset: 0,
                len: bytes.len() as u64,
            },
            batch.len() as u32,
            &dst,
        );
        gpu.submit_wait(enc);
        let got: Vec<SourceResult> =
            bytemuck::cast_slice(&gpu.read(&dst, 32 * batch.len() as u64)).to_vec();
        for (case, r) in batch.into_iter().zip(got) {
            cases.push(Case {
                gpu: r,
                header,
                air,
                curve: curve.clone(),
                case,
            });
        }
    }
    assert_eq!(cases.len(), 10_000);

    let mut worst = [
        Worst {
            name: "gain_l",
            ..Default::default()
        },
        Worst {
            name: "gain_r",
            ..Default::default()
        },
        Worst {
            name: "itd_s",
            ..Default::default()
        },
        Worst {
            name: "cutoff_hz",
            ..Default::default()
        },
        Worst {
            name: "pitch",
            ..Default::default()
        },
    ];
    for c in &cases {
        let h = &c.header;
        let air = &c.air;
        let c64 = air.speed_of_sound();

        // ---- The CPU oracle, f64. ----
        let (s, l) = (c.case.source, [h.listener[0], h.listener[1], h.listener[2]]);
        let right = [h.right[0], h.right[1], h.right[2]];
        let vs = c.case.velocity;
        let vl = [h.velocity[0], h.velocity[1], h.velocity[2]];
        let sf = s.map(|v| v as f64);
        let lf = l.map(|v| v as f64);
        let d64 = [lf[0] - sf[0], lf[1] - sf[1], lf[2] - sf[2]];
        let r64 = (d64[0] * d64[0] + d64[1] * d64[1] + d64[2] * d64[2]).sqrt();
        let mut loss = [0.0f64; 4];
        for b in 0..4 {
            let f = BANDS_HZ[b] as f64;
            loss[b] = air.absorption_db_per_m(f) * r64
                + barrier_insertion_db(c.case.excess[b] as f64, c64 / f)
                + foliage_attenuation_db(c.case.foliage_m as f64, f);
        }
        let fit = fit_lowpass(&BANDS_HZ.map(|v| v as f64), &loss);
        let curve: Vec<(f64, f64)> = c
            .curve
            .iter()
            .map(|&(m, hz)| (m as f64, hz as f64))
            .collect();
        let cutoff64 = match cutoff_ceiling(&curve, r64) {
            Some(ceiling) => fit.cutoff_hz.min(ceiling),
            None => fit.cutoff_hz,
        };
        let ears = Ears {
            position: lf,
            forward: [
                h.forward[0] as f64,
                h.forward[1] as f64,
                h.forward[2] as f64,
            ],
            right: right.map(|v| v as f64),
        };
        let heard = ears.hear(sf, air, PROBE_HZ as f64);
        let spread = spreading_gain(r64, c.case.directivity as f64);
        let beside64 = ((sf[0] - lf[0]) * right[0] as f64
            + (sf[1] - lf[1]) * right[1] as f64
            + (sf[2] - lf[2]) * right[2] as f64)
            / r64;
        let itd64 = if beside64 >= 0.0 {
            heard.interaural_delay
        } else {
            -heard.interaural_delay
        };
        let u = d64.map(|v| v / r64);
        let toward = vs[0] as f64 * u[0] + vs[1] as f64 * u[1] + vs[2] as f64 * u[2];
        let listener_toward = -(vl[0] as f64 * u[0] + vl[1] as f64 * u[1] + vl[2] as f64 * u[2]);
        let pitch64 = doppler_ratio(toward, listener_toward, air);
        let oracle = [
            spread * fit.gain * heard.left_gain,
            spread * fit.gain * heard.right_gain,
            itd64,
            cutoff64,
            pitch64,
        ];

        // ---- The bound, through the shader's operations, at the f32 speed of sound the
        // GPU was handed; and the same mirror at the f64 one, whose difference is the
        // effect of that input's rounding. ----
        let ex = c.case.excess.map(x);
        let at32 = mirror(c, x(h.listener[3]), ex);
        let at64 = mirror(c, E { v: c64, e: 0.0 }, ex);
        let got = [
            c.gpu.gain_l,
            c.gpu.gain_r,
            c.gpu.itd_s,
            c.gpu.cutoff_hz,
            c.gpu.pitch,
        ];
        for i in 0..5 {
            let input = (at32.out[i].v - at64.out[i].v).abs();
            let slack = 1e-12 * oracle[i].abs() + 1e-30;
            let bound = at32.out[i].e + at32.flip[i] + input + slack;
            worst[i].see((got[i] as f64 - oracle[i]).abs(), bound);
            // The mirror checks itself: at the oracle's c its exact value is the oracle's,
            // to its method errors (series, asin) and the flip allowance.
            assert!(
                (at64.out[i].v - oracle[i]).abs()
                    <= at64.out[i].e + at64.flip[i] + slack + 1e-9 * oracle[i].abs(),
                "the mirror of {} disagrees with the oracle: {} against {}",
                worst[i].name,
                at64.out[i].v,
                oracle[i],
            );
        }
        assert_eq!(c.gpu.tag(), c.case.tag);
    }
    println!(
        "L1 over {} cases, worst |gpu - cpu| as a share of its derived bound:",
        cases.len()
    );
    for w in &worst {
        println!(
            "  {:>9}: {:.3} of its bound ({:.3e} against {:.3e})",
            w.name, w.ratio, w.diff, w.bound
        );
    }
}

/// The shader's `laws` outputs, bounded: gain_l, gain_r, itd_s, cutoff_hz, pitch; and an
/// allowance per output for its one discontinuity, which ear is the near one when the
/// lateral cosine's sign is within its error.
struct Mirrored {
    out: [E; 5],
    flip: [f64; 5],
}

fn mirror(c: &Case, c_e: E, ex: [E; 4]) -> Mirrored {
    let h = &c.header;
    let air = &c.air;
    let (s, l) = (c.case.source, [h.listener[0], h.listener[1], h.listener[2]]);
    let right = [h.right[0], h.right[1], h.right[2]];
    let vs = c.case.velocity;
    let vl = [h.velocity[0], h.velocity[1], h.velocity[2]];

    let d = [
        x(l[0]).sub(x(s[0])),
        x(l[1]).sub(x(s[1])),
        x(l[2]).sub(x(s[2])),
    ];
    let r = dot(&d, &d).sqrt();
    let spread = x(c.case.directivity).max_c(0.0).div(r.max_c(REFERENCE_M));
    let fol = x(c.case.foliage_m).max_c(0.0).min_c(FOLIAGE_MAX_CREDITED_M);
    let mut loss = [E { v: 0.0, e: 0.0 }; 4];
    for b in 0..4 {
        let f = BANDS_HZ[b] as f64;
        // The header's absorption is the f64 figure rounded to f32: an input error.
        let alpha = k(air.absorption_db_per_m(f));
        let lam = c_e.mul(k(1.0 / f));
        let base = alpha.mul(r).add(k(foliage_absorption_db_per_m(f)).mul(fol));
        let bar = E {
            v: barrier_insertion_db(ex[b].v, lam.v),
            e: barrier_bound(ex[b], lam),
        };
        loss[b] = base.add(bar);
    }
    let y: Vec<E> = loss
        .iter()
        .map(|l| l.mul(k(10f64.log2() / 10.0)).exp2())
        .collect();
    let xs = BANDS_HZ.map(|f| (f as f64 / FIT_REFERENCE_HZ).powi(2));
    let mean = xs.iter().sum::<f64>() / 4.0;
    let sxx: f64 = xs.iter().map(|v| (v - mean).powi(2)).sum();
    let w: Vec<E> = xs.iter().map(|v| k((v - mean) / sxx)).collect();
    let b = dot(&w, &y);
    let y_mean = sum(&y).mul(E { v: 0.25, e: 0.0 });
    let a = y_mean.sub(b.mul(k(mean))).max_c(1.0);
    let g = a.inverse_sqrt();
    let mut cutoff = if b.v - b.e > 0.0 {
        k(FIT_REFERENCE_HZ)
            .mul(a.div(b).sqrt())
            .min_c(NO_LOWPASS_HZ)
    } else if b.v + b.e <= 0.0 {
        E {
            v: NO_LOWPASS_HZ,
            e: 0.0,
        }
    } else {
        // The slope's sign is in doubt: the shader returns 20 kHz or FREF sqrt(a / b),
        // and with b under 2 e_b the latter is at least FREF sqrt(a / (2 e_b)).
        let v = if b.v > 0.0 {
            (FIT_REFERENCE_HZ * (a.v / b.v).sqrt()).min(NO_LOWPASS_HZ)
        } else {
            NO_LOWPASS_HZ
        };
        let lo =
            (FIT_REFERENCE_HZ * ((a.v - a.e).max(1.0) / (2.0 * b.e)).sqrt()).min(NO_LOWPASS_HZ);
        E {
            v,
            e: (v - lo).abs().max(NO_LOWPASS_HZ - lo),
        }
    };
    if !c.curve.is_empty() {
        // Piecewise linear in distance: its steepest slope times the distance's error,
        // plus the lerp's own roundings (a subtraction, a division, a multiply and an add,
        // each within 2.5 ulp at the ceiling's or the distance's magnitude).
        let mut slope: f64 = 0.0;
        for p in c.curve.windows(2) {
            let span = (p[1].0 - p[0].0) as f64;
            if span > 0.0 {
                slope = slope.max(((p[1].1 - p[0].1) as f64 / span).abs());
            }
        }
        let pts: Vec<(f64, f64)> = c
            .curve
            .iter()
            .map(|&(m, hz)| (m as f64, hz as f64))
            .collect();
        let ceil_v = cutoff_ceiling(&pts, r.v).unwrap();
        let ceiling = E {
            v: ceil_v,
            e: slope * r.e + 10.0 * ulp(ceil_v) + slope * 10.0 * ulp(r.v),
        };
        cutoff = E {
            v: cutoff.v.min(ceiling.v),
            e: cutoff.e.max(ceiling.e),
        };
    }
    let sl = [
        x(s[0]).sub(x(l[0])),
        x(s[1]).sub(x(l[1])),
        x(s[2]).sub(x(l[2])),
    ];
    let beside = dot(&sl, &right.map(x)).div(r);
    let ab = beside.abs().min_c(1.0);
    // asin by A&S 4.4.46, as the shader evaluates it, plus the approximation's own 2e-8.
    let p = horner(&ASIN_COEFFICIENTS, ab);
    let one = E { v: 1.0, e: 0.0 };
    let lat = k(std::f64::consts::FRAC_PI_2).sub(one.sub(ab).sqrt().mul(p));
    let lateral = E {
        v: ab.v.asin(),
        e: lat.e + ASIN_ERROR + (lat.v - ab.v.asin()).abs(),
    };
    let itd_mag = k(HEAD_RADIUS_M).mul(lateral.add(ab)).div(c_e);
    let far = one.sub(k(shadow_k()).mul(beside.abs())).max_c(0.0);
    let right_side = beside.v >= 0.0;
    let (gl_s, gr_s) = if right_side { (far, one) } else { (one, far) };
    let amp = spread.mul(g);
    let (gl, gr) = (amp.mul(gl_s), amp.mul(gr_s));
    let itd = if right_side { itd_mag } else { itd_mag.neg() };
    let toward = dot(&vs.map(x), &d).div(r);
    let lt = dot(&vl.map(x), &d.map(|v| v.neg())).div(r);
    let lim = k(DOPPLER_MAX_CLOSING).mul(c_e);
    let closing = E {
        v: toward.v.clamp(-lim.v, lim.v),
        e: toward.e.max(lim.e),
    };
    let pitch = c_e.add(lt).div(c_e.sub(closing));

    let doubt = beside.v.abs() <= beside.e;
    let flip = if doubt {
        let g = (gl.v - gr.v).abs() + gl.e.max(gr.e);
        [g, g, 2.0 * itd.v.abs() + itd.e, 0.0, 0.0]
    } else {
        [0.0; 5]
    };
    Mirrored {
        out: [gl, gr, itd, cutoff, pitch],
        flip,
    }
}

/// The largest signed excess over the terrain columns between `s` and `l`, in f64: every
/// cell piece, refined by golden-section search (the L3 oracle, terrain only).
fn terrain_edge(scene: &Scene, s: [f64; 3], l: [f64; 3]) -> f64 {
    let d = [l[0] - s[0], l[1] - s[1], l[2] - s[2]];
    let cell = scene.cell as f64;
    let mut cuts = vec![0.0, 1.0];
    for (o, dd, n) in [(s[0], d[0], scene.cols), (s[2], d[2], scene.rows)] {
        if dd.abs() > 1e-300 {
            for k in 0..=n {
                let t = (k as f64 * cell - o) / dd;
                if t > 0.0 && t < 1.0 {
                    cuts.push(t);
                }
            }
        }
    }
    cuts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let signed = |t: f64, h: f64| {
        let p = [s[0] + t * d[0], s[1] + t * d[1], s[2] + t * d[2]];
        let q = [p[0], h, p[2]];
        let dist = |a: [f64; 3], b: [f64; 3]| {
            ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
        };
        let e = dist(q, s) + dist(l, q) - dist(l, s);
        if h > p[1] {
            e
        } else {
            -e
        }
    };
    let mut best = f64::MIN;
    for w in cuts.windows(2) {
        let (t0, t1) = (w[0], w[1]);
        if t1 <= t0 {
            continue;
        }
        let m = 0.5 * (t0 + t1);
        let Some(h) = scene.height_at(s[0] + m * d[0], s[2] + m * d[2]) else {
            continue;
        };
        best = best.max(signed(t0, h)).max(signed(t1, h));
        let g = (5f64.sqrt() - 1.0) / 2.0;
        let (mut a, mut b) = (t0, t1);
        for _ in 0..100 {
            let (c, dd) = (b - g * (b - a), a + g * (b - a));
            if signed(c, h) > signed(dd, h) {
                b = dd
            } else {
                a = c
            }
        }
        best = best.max(signed(0.5 * (a + b), h));
    }
    best
}

/// **L13. Temporal stability.** A source walks over a ridge crest at 11.5 m/s, the fastest
/// recorded ground speed, sampled at 1/60 s. No step in `gain_l`, `gain_r` or `cutoff_hz`
/// may exceed the continuous law's own change over that step (its largest slope there,
/// by finite differences of the f64 law at eight sub-steps, times the distance moved),
/// plus the GPU's derived distance from the law at each end: the L3 bound on the edge
/// carried through the L1 mirror.
#[test]
fn l13_a_source_walking_over_a_crest_does_not_click() {
    let Some(gpu) = gpu() else { return };
    let air = Air::standard();
    let c64 = air.speed_of_sound();
    // A smooth ridge along z at x = 100 m, 6 m high, sigma 8 m, sampled into 2 m columns.
    let ridge = |x: f64| 6.0 * (-((x - 100.0) / 8.0).powi(2) / 2.0).exp();
    let mut scene = Scene::flat(140, 100, 2.0, 0.0);
    for r in 0..100 {
        for cc in 0..140 {
            scene.heights[r * 140 + cc] = ridge(cc as f64 * 2.0 + 1.0) as f32;
        }
    }
    let mut ac = GpuAcoustics::new(&gpu.device, AcousticLimits::MAX).unwrap();
    scene.load(&gpu, &mut ac);
    let listener = Listener {
        position: [60.0, 1.6, 80.0],
        forward: [0.0, 0.0, -1.0],
        right: [1.0, 0.0, 0.0],
        velocity: [0.0; 3],
    };
    let ladder = [(0.0f32, 20_000.0f32), (45.0, 1_400.0), (130.0, 480.0)];
    let header = DispatchHeader::new(&listener, &air, &ladder).unwrap();
    let speed = 11.5f64;
    let step = speed / 60.0;
    let at = |k: f64| -> [f64; 3] {
        let x = 80.0 + k * step;
        [x, ridge(x) + 1.0, 110.0]
    };
    let steps = ((130.0 - 80.0) / step) as usize;
    let positions: Vec<[f32; 3]> = (0..=steps)
        .map(|k| at(k as f64).map(|v| v as f32))
        .collect();
    let mut gpu_out: Vec<SourceResult> = Vec::new();
    for chunk in positions.chunks(MAX_SOURCES as usize) {
        let sources: Vec<Source> = chunk
            .iter()
            .enumerate()
            .map(|(i, p)| {
                Source::new(*p, 1.0, [speed as f32, 0.0, 0.0], i as u32, NO_MOVER).unwrap()
            })
            .collect();
        let out = dispatch(&gpu, &mut ac, &header, &sources, &[]);
        gpu_out.extend_from_slice(out.results());
    }
    let l = listener.position.map(|v| v as f64);
    // The continuous law at a point: the f64 edge and the f64 laws.
    let law = |p: [f64; 3]| -> [f64; 3] {
        let e = terrain_edge(&scene, p, l);
        let d = [l[0] - p[0], l[1] - p[1], l[2] - p[2]];
        let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        let mut loss = [0.0; 4];
        for b in 0..4 {
            let f = BANDS_HZ[b] as f64;
            loss[b] = air.absorption_db_per_m(f) * r + barrier_insertion_db(e, c64 / f);
        }
        let fit = fit_lowpass(&BANDS_HZ.map(|v| v as f64), &loss);
        let pts: Vec<(f64, f64)> = ladder
            .iter()
            .map(|&(m, hz)| (m as f64, hz as f64))
            .collect();
        let cutoff = fit.cutoff_hz.min(cutoff_ceiling(&pts, r).unwrap());
        let ears = Ears {
            position: l,
            forward: [0.0, 0.0, -1.0],
            right: [1.0, 0.0, 0.0],
        };
        let heard = ears.hear(p, &air, PROBE_HZ as f64);
        let g = spreading_gain(r, 1.0) * fit.gain;
        [g * heard.left_gain, g * heard.right_gain, cutoff]
    };
    // The GPU's bound against the law at a sampled position.
    let bound = |k: usize| -> [f64; 3] {
        let p32 = positions[k];
        let p = p32.map(|v| v as f64);
        let e = terrain_edge(&scene, p, l);
        let len = ((l[0] - p[0]).powi(2) + (l[1] - p[1]).powi(2) + (l[2] - p[2]).powi(2)).sqrt();
        let mag = p
            .iter()
            .chain(l.iter())
            .map(|v| v.abs())
            .fold(0.0, f64::max);
        let dt = [(p[0], l[0] - p[0], 280.0), (p[2], l[2] - p[2], 200.0)]
            .iter()
            .map(|&(o, dd, hi): &(f64, f64, f64)| 4.0 * U * (hi + o.abs()) / dd.abs().max(1e-300))
            .fold(8.0 * U, f64::max);
        let e3 = 16.0 * U * (2.0 * len + e.abs() + 2.0 * mag) + 8.0 * U * mag + 2.0 * len * dt;
        let case = Case {
            gpu: SourceResult::default(),
            header,
            air,
            curve: ladder.to_vec(),
            case: LawCase {
                source: p32,
                directivity: 1.0,
                velocity: [speed as f32, 0.0, 0.0],
                tag: 0,
                excess: [e as f32; 4],
                probe_excess: e as f32,
                foliage_m: 0.0,
                pad: [0.0; 2],
            },
        };
        let ex = [E { v: e, e: e3 }; 4];
        let m32 = mirror(&case, x(header.listener[3]), ex);
        let m64 = mirror(&case, E { v: c64, e: 0.0 }, ex);
        [0, 1, 3].map(|i| m32.out[i].e + m32.flip[i] + (m32.out[i].v - m64.out[i].v).abs())
    };
    let mut worst = [0.0f64; 3];
    let mut largest_step = [0.0f64; 3];
    let bounds: Vec<[f64; 3]> = (0..positions.len()).map(bound).collect();
    for k in 0..steps {
        let sub = 8;
        let mut slope = [0.0f64; 3];
        let mut prev = law(at(k as f64));
        for j in 1..=sub {
            let next = law(at(k as f64 + j as f64 / sub as f64));
            for o in 0..3 {
                slope[o] = slope[o].max((next[o] - prev[o]).abs() / (step / sub as f64));
            }
            prev = next;
        }
        let g0 = [gpu_out[k].gain_l, gpu_out[k].gain_r, gpu_out[k].cutoff_hz];
        let g1 = [
            gpu_out[k + 1].gain_l,
            gpu_out[k + 1].gain_r,
            gpu_out[k + 1].cutoff_hz,
        ];
        for o in 0..3 {
            let moved = (g1[o] as f64 - g0[o] as f64).abs();
            let allowed = slope[o] * step + bounds[k][o] + bounds[k + 1][o];
            assert!(
                moved <= allowed,
                "step {k}, output {o}: moved {moved:e}, the law allows {allowed:e}"
            );
            worst[o] = worst[o].max(moved / allowed);
            largest_step[o] = largest_step[o].max(moved);
        }
    }
    let crossed = gpu_out.iter().any(|r| r.blocked()) && gpu_out.iter().any(|r| !r.blocked());
    assert!(crossed, "the walk never crossed the shadow boundary");
    println!(
        "L13: {steps} steps of {step:.4} m over the crest; worst step as a share of the law's: gain_l {:.3}, gain_r {:.3}, cutoff {:.3}; largest steps {:.3e}, {:.3e}, {:.1} Hz",
        worst[0], worst[1], worst[2], largest_step[0], largest_step[1], largest_step[2]
    );
}
