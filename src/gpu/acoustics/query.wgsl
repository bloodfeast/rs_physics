// The acoustic query: one workgroup per source, one for the listener field, and the law
// probe the oracles drive. The constants above this line are generated from the f64 CPU
// laws by `shader::source`; none is typed here.
//
// Every loop is bounded by a constant, and every lane reaches every barrier.

@group(0) @binding(0) var<storage, read> inb: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read_write> outb: array<u32>;
@group(0) @binding(2) var<storage, read> scene: array<u32>;

const BIG: f32 = 3.4028234663852886e38; // f32::MAX

// Dispatch header, in 16-byte vectors.
const HV_LISTENER: u32 = 0u;
const HV_RIGHT: u32 = 2u;
const HV_VELOCITY: u32 = 3u;
const HV_AIR: u32 = 4u;
const HV_LEGIBILITY: u32 = 5u;
const HV_COUNTS: u32 = 9u;
const HV_SOURCES: u32 = 10u;

// ---------------------------------------------------------------------------------------
// Scene access.

fn sf(i: u32) -> f32 {
    return bitcast<f32>(scene[i]);
}

struct Grid {
    cols: u32,
    rows: u32,
    cell: f32,
    inv_cell: f32,
    origin: vec2<f32>,
}

fn grid() -> Grid {
    return Grid(scene[H_COLS], scene[H_ROWS], sf(H_CELL), sf(H_INV_CELL),
                vec2<f32>(sf(H_ORIGIN_X), sf(H_ORIGIN_Z)));
}

fn height(g: Grid, c: u32, r: u32) -> f32 {
    return sf(scene[H_OFF_HEIGHTS] + r * g.cols + c);
}

fn byte_at(off: u32, idx: u32) -> u32 {
    return (scene[off + idx / 4u] >> ((idx % 4u) * 8u)) & 0xFFu;
}

fn unord(u: u32) -> f32 {
    return bitcast<f32>(select(~u, u & 0x7FFFFFFFu, (u & 0x80000000u) != 0u));
}

// Rows of a 3x4 transform, from four words each.
struct Rows {
    r0: vec4<f32>,
    r1: vec4<f32>,
    r2: vec4<f32>,
}

fn scene_rows(off: u32) -> Rows {
    return Rows(
        vec4<f32>(sf(off), sf(off + 1u), sf(off + 2u), sf(off + 3u)),
        vec4<f32>(sf(off + 4u), sf(off + 5u), sf(off + 6u), sf(off + 7u)),
        vec4<f32>(sf(off + 8u), sf(off + 9u), sf(off + 10u), sf(off + 11u)));
}

// The inverse of an affine 3x4, or a zero matrix for a degenerate one (which then crosses
// nothing).
fn inverse_rows(m: Rows) -> Rows {
    let a0 = m.r0.xyz;
    let a1 = m.r1.xyz;
    let a2 = m.r2.xyz;
    let c0 = cross(a1, a2);
    let c1 = cross(a2, a0);
    let c2 = cross(a0, a1);
    let det = dot(a0, c0);
    if (abs(det) < 1e-30) {
        return Rows(vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0));
    }
    let inv = 1.0 / det;
    // Rows of A^-1 are the components of the cofactor columns.
    let i0 = vec3<f32>(c0.x, c1.x, c2.x) * inv;
    let i1 = vec3<f32>(c0.y, c1.y, c2.y) * inv;
    let i2 = vec3<f32>(c0.z, c1.z, c2.z) * inv;
    let t = vec3<f32>(m.r0.w, m.r1.w, m.r2.w);
    return Rows(vec4<f32>(i0, -dot(i0, t)), vec4<f32>(i1, -dot(i1, t)), vec4<f32>(i2, -dot(i2, t)));
}

fn to_local(inv: Rows, p: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(dot(inv.r0.xyz, p) + inv.r0.w, dot(inv.r1.xyz, p) + inv.r1.w,
                     dot(inv.r2.xyz, p) + inv.r2.w);
}

fn to_local_dir(inv: Rows, d: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(dot(inv.r0.xyz, d), dot(inv.r1.xyz, d), dot(inv.r2.xyz, d));
}

// Clip `o + t d` to the slab |x| <= 0.5 on one axis.
fn slab(o: f32, d: f32, t0: ptr<function, f32>, t1: ptr<function, f32>) {
    if (abs(d) < 1e-30) {
        if (abs(o) > 0.5) {
            *t0 = 1.0;
            *t1 = 0.0;
        }
        return;
    }
    let a = (-0.5 - o) / d;
    let b = (0.5 - o) / d;
    *t0 = max(*t0, min(a, b));
    *t1 = min(*t1, max(a, b));
}

// ---------------------------------------------------------------------------------------
// The signed excess.

// `|SQ| + |QL| - |SL|` with Q the point at height `h` under the segment at `t`, signed
// positive where `h` stands above the line.
fn signed_excess(s: vec3<f32>, l: vec3<f32>, d: vec3<f32>, len: f32, t: f32, h: f32) -> f32 {
    let p = s + t * d;
    let q = vec3<f32>(p.x, h, p.z);
    let e = length(q - s) + length(l - q) - len;
    return select(-e, e, h > p.y);
}

// The largest signed excess over `t` in [t0, t1] at height `h`. The unsigned excess is
// convex in `t` (a sum of two hyperbolae), so where `h` is above the line the maximum is at
// an end, and where it is below, the least clearance is at the convex minimum
// `t* = |a| / (|a| + |b|)`: three evaluations are exact.
fn excess_over(s: vec3<f32>, l: vec3<f32>, d: vec3<f32>, len: f32, t0: f32, t1: f32,
               h: f32) -> f32 {
    let a = abs(h - s.y);
    let b = abs(h - l.y);
    var ts = 0.5;
    if (a + b > 0.0) {
        ts = a / (a + b);
    }
    let tm = clamp(ts, t0, t1);
    let e0 = signed_excess(s, l, d, len, t0, h);
    let e1 = signed_excess(s, l, d, len, t1, h);
    let em = signed_excess(s, l, d, len, tm, h);
    return max(max(e0, e1), em);
}

struct Crossing {
    ta: f32,
    tb: f32,
}

// Where the segment's footprint crosses a box's upright footprint, in segment `t`.
fn crossing(inv: Rows, s: vec3<f32>, d: vec3<f32>) -> Crossing {
    let o = to_local(inv, s);
    let dl = to_local_dir(inv, d);
    var t0 = 0.0;
    var t1 = 1.0;
    slab(o.x, dl.x, &t0, &t1);
    slab(o.z, dl.z, &t0, &t1);
    return Crossing(t0, t1);
}

// One obstacle's contribution to the per-band maxima, with the Fresnel rule applied per
// band at the crossing's midpoint.
fn obstacle(m: Rows, hw: f32, x: Crossing, s: vec3<f32>, l: vec3<f32>, d: vec3<f32>,
            len: f32, c: f32, best: ptr<function, vec4<f32>>, probe: ptr<function, f32>) {
    let half_y = 0.5 * (abs(m.r1.x) + abs(m.r1.y) + abs(m.r1.z));
    let top = m.r1.w + half_y;
    let bottom = m.r1.w - half_y;
    let ya = s.y + x.ta * d.y;
    let yb = s.y + x.tb * d.y;
    if (ya < bottom && yb < bottom) {
        return; // the path passes beneath it
    }
    let e = excess_over(s, l, d, len, x.ta, x.tb, top);
    let tm = 0.5 * (x.ta + x.tb);
    let d1 = tm * len;
    let d2 = len - d1;
    let zone = d1 * d2 / len; // r1^2 = lambda * zone
    let hw2 = hw * hw;
    let lam = c * BAND_INV_HZ;
    let admitted = vec4<f32>(hw2) >= lam * zone;
    *best = select(*best, max(*best, vec4<f32>(e)), admitted);
    if (hw2 >= c * PROBE_INV_HZ * zone) {
        *probe = max(*probe, e);
    }
}

// ---------------------------------------------------------------------------------------
// The laws, in f32. The oracle is the f64 CPU form in `crate::acoustics`.

// x coth x: its series under 1, where the exponential form cancels; the exponential form
// above, where it does not.
fn x_coth_x(x: f32) -> f32 {
    if (x < 1.0) {
        let z = x * x;
        var p = COTH_C7;
        p = p * z + COTH_C6;
        p = p * z + COTH_C5;
        p = p * z + COTH_C4;
        p = p * z + COTH_C3;
        p = p * z + COTH_C2;
        p = p * z + COTH_C1;
        return p * z + 1.0;
    }
    let q = exp2(-2.0 * LOG2_E * x);
    return x * (1.0 + q) / (1.0 - q);
}

// x cot x on the lit zone, [0, LIT_X0], by its series: `tan` is inherited from `sin / cos`
// in WGSL at 2^-11 absolute, which near x = 0 would be a relative error of 2^-11 / x.
fn x_cot_x(x: f32) -> f32 {
    let z = x * x;
    var p = COT_C8;
    p = p * z + COT_C7;
    p = p * z + COT_C6;
    p = p * z + COT_C5;
    p = p * z + COT_C4;
    p = p * z + COT_C3;
    p = p * z + COT_C2;
    p = p * z + COT_C1;
    return p * z + 1.0;
}

fn barrier_db(excess: f32, lambda: f32) -> f32 {
    let n = 2.0 * excess / lambda;
    if (n == 0.0) {
        return BARRIER_GRAZING_DB;
    }
    let x = sqrt(TWO_PI * abs(n));
    var ratio = 1.0;
    if (n > 0.0) {
        ratio = x_coth_x(x);
    } else {
        if (x >= LIT_X0) {
            return 0.0;
        }
        ratio = x_cot_x(x);
    }
    return clamp(BARRIER_GRAZING_DB + DB_PER_LOG2 * log2(ratio), 0.0, BARRIER_CAP_DB);
}

// acos on [0, 1] by Abramowitz and Stegun 4.4.46 (|error| <= 2e-8); WGSL's asin is
// inherited from atan2 at 4096 ulp.
fn asin_unit(x: f32) -> f32 {
    var p = ASIN_A7;
    p = p * x + ASIN_A6;
    p = p * x + ASIN_A5;
    p = p * x + ASIN_A4;
    p = p * x + ASIN_A3;
    p = p * x + ASIN_A2;
    p = p * x + ASIN_A1;
    p = p * x + ASIN_A0;
    return HALF_PI - sqrt(1.0 - x) * p;
}

fn leg_point(k: u32) -> vec2<f32> {
    let v = bitcast<vec4<f32>>(inb[HV_LEGIBILITY + k / 2u]);
    return select(v.zw, v.xy, k % 2u == 0u);
}

fn legibility(r: f32, n: u32) -> f32 {
    let first = leg_point(0u);
    if (r <= first.x) {
        return first.y;
    }
    let last = leg_point(n - 1u);
    if (r >= last.x) {
        return last.y;
    }
    for (var k = 1u; k < MAX_LEGIBILITY; k = k + 1u) {
        if (k >= n) {
            break;
        }
        let a = leg_point(k - 1u);
        let b = leg_point(k);
        if (r <= b.x) {
            let span = b.x - a.x;
            var t = 1.0;
            if (span > 0.0) {
                t = (r - a.x) / span;
            }
            return a.y + (b.y - a.y) * t;
        }
    }
    return last.y;
}

// The whole per-source law chain, from the march's maxima to a result written at `at`,
// run by a whole workgroup: lanes 0 to 3 take one band each (air, barrier, foliage, the
// band's transmission), lane 4 the ears and the Doppler ratio, lane 5 spreading and the
// legibility ceiling; lane 0 then fits and writes. Every lane must call it. Each band's
// operations are the ones a single lane would do, in the same order.
var<workgroup> law_y: array<f32, 4>;
var<workgroup> law_ears: vec4<f32>;
var<workgroup> law_misc: vec4<f32>;

fn laws(lane: u32, s: vec3<f32>, directivity: f32, vs: vec3<f32>, tag: u32, ex: vec4<f32>,
        ex_probe: f32, foliage: f32, at: u32) {
    let lv = bitcast<vec4<f32>>(inb[HV_LISTENER]);
    let l = lv.xyz;
    let c = lv.w;
    let d = l - s;
    let r = length(d);
    if (lane < 4u) {
        let alpha = bitcast<vec4<f32>>(inb[HV_AIR])[lane];
        let fol = clamp(foliage, 0.0, FOLIAGE_MAX_M);
        let lam = c * BAND_INV_HZ[lane];
        let loss = alpha * r + FOLIAGE_DB_PER_M[lane] * fol + barrier_db(ex[lane], lam);
        // The fit's ordinate: 1/T = 10^(L/10).
        law_y[lane] = exp2(loss * LOG2_10_OVER_10);
    } else if (lane == 4u) {
        // The ears: Woodworth timing and the head shadow at the probe band.
        let right = bitcast<vec4<f32>>(inb[HV_RIGHT]).xyz;
        let vl = bitcast<vec4<f32>>(inb[HV_VELOCITY]).xyz;
        var itd = 0.0;
        var gl = 1.0;
        var gr = 1.0;
        var pitch = 1.0;
        if (r > 0.0) {
            let beside = dot(s - l, right) / r;
            let ab = min(abs(beside), 1.0);
            let lateral = asin_unit(ab);
            itd = HEAD_RADIUS_M * (lateral + ab) / c;
            let far = max(1.0 - SHADOW_K * abs(beside), 0.0);
            if (beside >= 0.0) {
                gl = far;
            } else {
                gr = far;
                itd = -itd;
            }
            let toward = dot(vs, d) / r;
            let listener_toward = dot(vl, -d) / r;
            let closing = clamp(toward, -DOPPLER_MAX * c, DOPPLER_MAX * c);
            pitch = (c + listener_toward) / (c - closing);
        }
        law_ears = vec4<f32>(gl, gr, itd, pitch);
    } else if (lane == 5u) {
        let n_leg = inb[HV_COUNTS].z;
        var ceiling = -1.0;
        if (n_leg > 0u) {
            ceiling = legibility(r, n_leg);
        }
        law_misc = vec4<f32>(max(directivity, 0.0) / max(r, REFERENCE_M), ceiling, 0.0, 0.0);
    }
    workgroupBarrier();
    if (lane == 0u) {
        // The fit: 1/T = a + b x, closed-form least squares over the bands.
        let y = vec4<f32>(law_y[0], law_y[1], law_y[2], law_y[3]);
        let b = dot(FIT_W, y);
        let y_mean = (y.x + y.y + y.z + y.w) * 0.25;
        let a = max(y_mean - b * FIT_X_MEAN, 1.0);
        let g = inverseSqrt(a);
        var cutoff = NO_LOWPASS_HZ;
        if (b > 0.0) {
            cutoff = min(FIT_REFERENCE_HZ * sqrt(a / b), NO_LOWPASS_HZ);
        }
        var flags = 0u;
        let ceiling = law_misc.y;
        if (ceiling >= 0.0 && ceiling < cutoff) {
            cutoff = ceiling;
            flags = flags | 4u;
        }
        if (ex_probe > 0.0) {
            flags = flags | 1u;
        }
        if (2.0 * ex_probe / (c * PROBE_INV_HZ) > -LIT_N0) {
            flags = flags | 2u;
        }
        let ears = law_ears;
        let amp = law_misc.x * g;
        outb[at] = bitcast<u32>(amp * ears.x);
        outb[at + 1u] = bitcast<u32>(amp * ears.y);
        outb[at + 2u] = bitcast<u32>(ears.z);
        outb[at + 3u] = bitcast<u32>(cutoff);
        outb[at + 4u] = bitcast<u32>(ears.w);
        outb[at + 5u] = bitcast<u32>(ex_probe);
        outb[at + 6u] = bitcast<u32>(foliage);
        outb[at + 7u] = (tag & 0xFFFFFFu) | (flags << 24u);
    }
}

// ---------------------------------------------------------------------------------------
// The source workgroup.

var<workgroup> red_band: array<vec4<f32>, 64>;
var<workgroup> red_probe: array<vec2<f32>, 64>;

fn clip1(o: f32, d: f32, lo: f32, hi: f32, t0: ptr<function, f32>, t1: ptr<function, f32>) {
    if (abs(d) < 1e-30) {
        if (o < lo || o > hi) {
            *t0 = 1.0;
            *t1 = 0.0;
        }
        return;
    }
    let a = (lo - o) / d;
    let b = (hi - o) / d;
    *t0 = max(*t0, min(a, b));
    *t1 = min(*t1, max(a, b));
}

// This lane's share of the terrain, the statics and the foliage along the path: the
// grid-clipped segment is split into 64 equal ranges of `t`, and each lane walks its own
// cells. A static is evaluated once, by the piece its footprint crossing starts in.
fn walk_terrain(g: Grid, s: vec3<f32>, l: vec3<f32>, d: vec3<f32>, len: f32, c: f32, lane: u32,
                best: ptr<function, vec4<f32>>, probe: ptr<function, f32>,
                fol: ptr<function, f32>) {
    let lo = g.origin;
    let hi = g.origin + vec2<f32>(f32(g.cols), f32(g.rows)) * g.cell;
    var c0 = 0.0;
    var c1 = 1.0;
    clip1(s.x, d.x, lo.x, hi.x, &c0, &c1);
    clip1(s.z, d.z, lo.y, hi.y, &c0, &c1);
    if (!(c0 < c1)) {
        return;
    }
    let span = c1 - c0;
    let ta = c0 + span * f32(lane) / 64.0;
    let tb = c0 + span * f32(lane + 1u) / 64.0;
    let p = s + ta * d;
    var ci = clamp(i32(floor((p.x - g.origin.x) * g.inv_cell)), 0, i32(g.cols) - 1);
    var cj = clamp(i32(floor((p.z - g.origin.y) * g.inv_cell)), 0, i32(g.rows) - 1);
    let step_i = select(-1, 1, d.x > 0.0);
    let step_j = select(-1, 1, d.z > 0.0);
    let has_foliage = scene[H_HAS_FOLIAGE] != 0u;
    let n_statics = scene[H_STATICS];
    var t = ta;
    for (var k = 0u; k < MAX_LANE_STEPS; k = k + 1u) {
        var tx = BIG;
        if (d.x > 0.0) {
            tx = (g.origin.x + f32(ci + 1) * g.cell - s.x) / d.x;
        } else if (d.x < 0.0) {
            tx = (g.origin.x + f32(ci) * g.cell - s.x) / d.x;
        }
        var tz = BIG;
        if (d.z > 0.0) {
            tz = (g.origin.y + f32(cj + 1) * g.cell - s.z) / d.z;
        } else if (d.z < 0.0) {
            tz = (g.origin.y + f32(cj) * g.cell - s.z) / d.z;
        }
        let te = max(min(min(tx, tz), tb), t);
        let cell = u32(cj) * g.cols + u32(ci);
        let h = height(g, u32(ci), u32(cj));
        let e = excess_over(s, l, d, len, t, te, h);
        *best = max(*best, vec4<f32>(e));
        *probe = max(*probe, e);
        if (has_foliage) {
            let rho = f32(byte_at(scene[H_OFF_FOLIAGE], cell)) / 255.0;
            *fol = *fol + rho * (te - t) * len;
        }
        if (n_statics > 0u) {
            let last = te >= tb;
            let claim_lo = select(t, -BIG, lane == 0u && k == 0u);
            let claim_hi = select(te, BIG, last && lane == 63u);
            let head = scene[scene[H_OFF_HEADS] + cell];
            let end = scene[scene[H_OFF_HEADS] + cell + 1u];
            for (var q = 0u; q < MAX_STATICS; q = q + 1u) {
                if (head + q >= end) {
                    break;
                }
                let si = scene[scene[H_OFF_LIST] + head + q];
                let inv = scene_rows(scene[H_OFF_INV] + 12u * si);
                let x = crossing(inv, s, d);
                if (x.ta < x.tb && x.ta >= claim_lo && x.ta < claim_hi) {
                    let base = scene[H_OFF_STATICS] + 16u * si;
                    obstacle(scene_rows(base), sf(base + 13u), x, s, l, d, len, c, best, probe);
                }
            }
        }
        if (te >= tb) {
            break;
        }
        if (tx <= tz) {
            ci = ci + step_i;
        }
        if (tz <= tx) {
            cj = cj + step_j;
        }
        if (ci < 0 || cj < 0 || ci >= i32(g.cols) || cj >= i32(g.rows)) {
            break;
        }
        t = te;
    }
}

fn mover_rows(n_src: u32, j: u32) -> Rows {
    let base = HV_SOURCES + 2u * n_src + 4u * j;
    return Rows(bitcast<vec4<f32>>(inb[base]), bitcast<vec4<f32>>(inb[base + 1u]),
                bitcast<vec4<f32>>(inb[base + 2u]));
}

// Source `i` of the dispatch, by a whole workgroup.
fn source_query(i: u32, lane: u32) {
    let counts = inb[HV_COUNTS];
    let n_src = counts.x;
    let n_mov = counts.y;
    let a = inb[HV_SOURCES + 2u * i];
    let b = inb[HV_SOURCES + 2u * i + 1u];
    let s = bitcast<vec3<f32>>(a.xyz);
    let directivity = bitcast<f32>(a.w);
    let vs = bitcast<vec3<f32>>(b.xyz);
    let word = b.w;
    let lv = bitcast<vec4<f32>>(inb[HV_LISTENER]);
    let l = lv.xyz;
    let c = lv.w;
    let d = l - s;
    let len = length(d);

    var best = vec4<f32>(-BIG);
    var probe = -BIG;
    var fol = 0.0;
    if (len > 0.0 && i < n_src) {
        let g = grid();
        if (g.cols > 0u && g.rows > 0u) {
            walk_terrain(g, s, l, d, len, c, lane, &best, &probe, &fol);
        }
        let ignore = word >> 24u;
        if (lane < n_mov && lane != ignore) {
            let m = mover_rows(n_src, lane);
            let extra = inb[HV_SOURCES + 2u * n_src + 4u * lane + 3u];
            let x = crossing(inverse_rows(m), s, d);
            if (x.ta < x.tb) {
                obstacle(m, bitcast<f32>(extra.y), x, s, l, d, len, c, &best, &probe);
            }
        }
    }
    red_band[lane] = best;
    red_probe[lane] = vec2<f32>(probe, fol);
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride = stride >> 1u) {
        if (lane < stride) {
            red_band[lane] = max(red_band[lane], red_band[lane + stride]);
            let o = red_probe[lane + stride];
            red_probe[lane] = vec2<f32>(max(red_probe[lane].x, o.x), red_probe[lane].y + o.y);
        }
        workgroupBarrier();
    }
    let band = red_band[0];
    let probe_fol = red_probe[0];
    laws(lane, s, directivity, vs, word, band, probe_fol.x, probe_fol.y, OUT_SOURCES + 8u * i);
    if (lane == 0u) {
        // The march's own answer, for the oracles: six stores a source.
        let at = OUT_EDGES + 8u * i;
        outb[at] = bitcast<u32>(band.x);
        outb[at + 1u] = bitcast<u32>(band.y);
        outb[at + 2u] = bitcast<u32>(band.z);
        outb[at + 3u] = bitcast<u32>(band.w);
        outb[at + 4u] = bitcast<u32>(probe_fol.x);
        outb[at + 5u] = bitcast<u32>(probe_fol.y);
    }
}

// ---------------------------------------------------------------------------------------
// The listener field.

struct Hit {
    t: f32,       // metres along the unit ray; negative for none
    n: vec3<f32>, // the normal it reflects about
    face: vec3<f32>, // the face it hit, to leave by
    material: u32,
}

fn material_r(m: u32) -> vec4<f32> {
    let off = scene[H_OFF_MATERIALS] + 4u * m;
    return vec4<f32>(sf(off), sf(off + 1u), sf(off + 2u), sf(off + 3u));
}

fn smooth_normal(g: Grid, ci: u32, cj: u32) -> vec3<f32> {
    let il = select(ci - 1u, ci, ci == 0u);
    let ih = min(ci + 1u, g.cols - 1u);
    let jl = select(cj - 1u, cj, cj == 0u);
    let jh = min(cj + 1u, g.rows - 1u);
    let dx = (height(g, ih, cj) - height(g, il, cj)) / (f32(max(ih - il, 1u)) * g.cell);
    let dz = (height(g, ci, jh) - height(g, ci, jl)) / (f32(max(jh - jl, 1u)) * g.cell);
    return normalize(vec3<f32>(-dx, 1.0, -dz));
}

// The static and terrain hits inside one cell piece [u0, u1] of the ray.
fn hit_cell(g: Grid, o: vec3<f32>, d: vec3<f32>, ci: u32, cj: u32, u0: f32, u1: f32,
            entered: vec3<f32>, best: ptr<function, Hit>) {
    let cell = cj * g.cols + ci;
    // Statics listed in the cell: the nearest entry at or after u0.
    let head = scene[scene[H_OFF_HEADS] + cell];
    let end = scene[scene[H_OFF_HEADS] + cell + 1u];
    for (var q = 0u; q < MAX_STATICS; q = q + 1u) {
        if (head + q >= end) {
            break;
        }
        let si = scene[scene[H_OFF_LIST] + head + q];
        let inv = scene_rows(scene[H_OFF_INV] + 12u * si);
        let ol = to_local(inv, o);
        let dl = to_local_dir(inv, d);
        var t0 = -BIG;
        var t1 = BIG;
        var axis = 0u;
        for (var k = 0u; k < 3u; k = k + 1u) {
            if (abs(dl[k]) < 1e-30) {
                if (abs(ol[k]) > 0.5) {
                    t0 = BIG;
                }
                continue;
            }
            let a = (-0.5 - ol[k]) / dl[k];
            let b = (0.5 - ol[k]) / dl[k];
            let near = min(a, b);
            if (near > t0) {
                t0 = near;
                axis = k;
            }
            t1 = min(t1, max(a, b));
        }
        if (t0 <= t1 && t0 >= u0 && t0 <= u1 && ((*best).t < 0.0 || t0 < (*best).t)) {
            var row = inv.r0.xyz;
            if (axis == 1u) {
                row = inv.r1.xyz;
            } else if (axis == 2u) {
                row = inv.r2.xyz;
            }
            let n = normalize(row) * -sign(dl[axis]);
            let base = scene[H_OFF_STATICS] + 16u * si;
            *best = Hit(t0, n, n, scene[base + 12u]);
        }
    }
    // The terrain column.
    let h = height(g, ci, cj);
    let y0 = o.y + u0 * d.y;
    let y1 = o.y + u1 * d.y;
    if (min(y0, y1) <= h) {
        var t = u0;
        var face = entered;
        if (y0 > h) {
            t = (h - o.y) / d.y;
            face = vec3<f32>(0.0, 1.0, 0.0);
        }
        if ((*best).t < 0.0 || t < (*best).t) {
            let ns = smooth_normal(g, ci, cj);
            var n = face;
            if (dot(d, ns) < 0.0 && dot(reflect(d, ns), face) > 0.0) {
                n = ns;
            }
            *best = Hit(t, n, face, byte_at(scene[H_OFF_MATERIAL], cell));
        }
    }
}

// March a unit ray from `o`, from `tmin` to `tmax` metres, against the terrain and the statics: 16 m
// blocks of the max pyramid first, skipping any block the ray passes wholly above, then
// the cells of the blocks it does not.
fn march(g: Grid, o: vec3<f32>, d: vec3<f32>, tmin: f32, tmax: f32) -> Hit {
    var none = Hit(-1.0, vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), 0u);
    if (g.cols == 0u || g.rows == 0u) {
        return none;
    }
    let lo = g.origin;
    let hi = g.origin + vec2<f32>(f32(g.cols), f32(g.rows)) * g.cell;
    var c0 = tmin;
    var c1 = tmax;
    clip1(o.x, d.x, lo.x, hi.x, &c0, &c1);
    clip1(o.z, d.z, lo.y, hi.y, &c0, &c1);
    if (!(c0 < c1)) {
        return none;
    }
    let block = g.cell * f32(1u << PYRAMID_TOP);
    let bc = scene[H_PYR_DIMS + 2u * (PYRAMID_TOP - 1u)];
    let br = scene[H_PYR_DIMS + 2u * (PYRAMID_TOP - 1u) + 1u];
    let pyr = scene[H_OFF_PYR + PYRAMID_TOP - 1u];
    let p0 = o + c0 * d;
    var bi = clamp(i32(floor((p0.x - lo.x) / block)), 0, i32(bc) - 1);
    var bj = clamp(i32(floor((p0.z - lo.y) / block)), 0, i32(br) - 1);
    let si = select(-1, 1, d.x > 0.0);
    let sj = select(-1, 1, d.z > 0.0);
    var t = c0;
    var entered = vec3<f32>(0.0, 1.0, 0.0);
    for (var kb = 0u; kb < MAX_BLOCK_STEPS; kb = kb + 1u) {
        var tx = BIG;
        if (d.x > 0.0) {
            tx = (lo.x + f32(bi + 1) * block - o.x) / d.x;
        } else if (d.x < 0.0) {
            tx = (lo.x + f32(bi) * block - o.x) / d.x;
        }
        var tz = BIG;
        if (d.z > 0.0) {
            tz = (lo.y + f32(bj + 1) * block - o.z) / d.z;
        } else if (d.z < 0.0) {
            tz = (lo.y + f32(bj) * block - o.z) / d.z;
        }
        let te = max(min(min(tx, tz), c1), t);
        let top = sf(pyr + u32(bj) * bc + u32(bi));
        let ymin = min(o.y + t * d.y, o.y + te * d.y);
        if (ymin <= top) {
            // The cells of this block, from t to te.
            let pc = o + t * d;
            let cell_lo_i = u32(bi) << PYRAMID_TOP;
            let cell_lo_j = u32(bj) << PYRAMID_TOP;
            let cell_hi_i = min(cell_lo_i + (1u << PYRAMID_TOP), g.cols) - 1u;
            let cell_hi_j = min(cell_lo_j + (1u << PYRAMID_TOP), g.rows) - 1u;
            var ci = clamp(i32(floor((pc.x - lo.x) * g.inv_cell)), i32(cell_lo_i), i32(cell_hi_i));
            var cj = clamp(i32(floor((pc.z - lo.y) * g.inv_cell)), i32(cell_lo_j), i32(cell_hi_j));
            var u = t;
            var found = none;
            for (var kc = 0u; kc < MAX_CELL_STEPS; kc = kc + 1u) {
                var ux = BIG;
                if (d.x > 0.0) {
                    ux = (lo.x + f32(ci + 1) * g.cell - o.x) / d.x;
                } else if (d.x < 0.0) {
                    ux = (lo.x + f32(ci) * g.cell - o.x) / d.x;
                }
                var uz = BIG;
                if (d.z > 0.0) {
                    uz = (lo.y + f32(cj + 1) * g.cell - o.z) / d.z;
                } else if (d.z < 0.0) {
                    uz = (lo.y + f32(cj) * g.cell - o.z) / d.z;
                }
                let ue = max(min(min(ux, uz), te), u);
                hit_cell(g, o, d, u32(ci), u32(cj), u, ue, entered, &found);
                if (found.t >= 0.0) {
                    return found;
                }
                if (ue >= te) {
                    break;
                }
                if (ux <= uz) {
                    ci = ci + si;
                    entered = vec3<f32>(-f32(si), 0.0, 0.0);
                }
                if (uz <= ux) {
                    cj = cj + sj;
                    entered = vec3<f32>(0.0, 0.0, -f32(sj));
                }
                if (ci < i32(cell_lo_i) || cj < i32(cell_lo_j) || ci > i32(cell_hi_i)
                    || cj > i32(cell_hi_j)) {
                    break;
                }
                u = ue;
            }
        }
        if (te >= c1) {
            break;
        }
        if (tx <= tz) {
            bi = bi + si;
            entered = vec3<f32>(-f32(si), 0.0, 0.0);
        }
        if (tz <= tx) {
            bj = bj + sj;
            entered = vec3<f32>(0.0, 0.0, -f32(sj));
        }
        if (bi < 0 || bj < 0 || bi >= i32(bc) || bj >= i32(br)) {
            break;
        }
        t = te;
    }
    return none;
}

var<private> FIELD_DIRS: array<vec4<f32>, 64> = FIELD_DIR_TABLE;

// ---------------------------------------------------------------------------------------
// The field rays: one workgroup per ray. Each segment of the ray is split into 64 equal
// lengths, one per lane, and the nearest hit wins, so a ray costs about two cells a lane
// rather than a lane marching 55 m alone.

var<workgroup> first_lane: atomic<u32>;
var<workgroup> hit_shared: array<vec4<f32>, 2>; // n.xyz and t; face.xyz and material

// The nearest of the lanes' hits. Lane k marches the k-th sixty-fourth of the segment and
// reports only hits inside it, so the nearest hit is the lowest lane's: one atomic, not a
// tree. The answer comes back workgroup-uniform, so the caller may branch on it and still
// reach its next barrier in uniform control flow.
fn nearest(lane: u32, h: Hit) -> Hit {
    if (lane == 0u) {
        atomicStore(&first_lane, 64u);
        hit_shared[0] = vec4<f32>(0.0, 1.0, 0.0, -1.0);
        hit_shared[1] = vec4<f32>(0.0, 1.0, 0.0, 0.0);
    }
    workgroupBarrier();
    if (h.t >= 0.0) {
        atomicMin(&first_lane, lane);
    }
    workgroupBarrier();
    if (lane == atomicLoad(&first_lane)) {
        hit_shared[0] = vec4<f32>(h.n, h.t);
        hit_shared[1] = vec4<f32>(h.face, bitcast<f32>(h.material));
    }
    let a = workgroupUniformLoad(&hit_shared[0]);
    let b = workgroupUniformLoad(&hit_shared[1]);
    workgroupBarrier();
    return Hit(a.w, a.xyz, b.xyz, bitcast<u32>(b.w));
}

// Field ray `ray`, by a whole workgroup.
fn field_ray(ray: u32, lane: u32) {
    let lv = bitcast<vec4<f32>>(inb[HV_LISTENER]);
    let l = lv.xyz;
    let c = lv.w;
    let right = bitcast<vec4<f32>>(inb[HV_RIGHT]).xyz;
    let g = grid();
    let dw = FIELD_DIRS[ray];
    let range = TAP_WINDOW_S * c * 0.5;
    let piece = (range - T_START) / 64.0;
    let t0 = T_START + piece * f32(lane);
    let t1 = T_START + piece * f32(lane + 1u);

    let h1 = nearest(lane, march(g, l, dw.xyz, t0, t1));
    var rec = array<f32, 16>();
    rec[0] = -1.0;
    rec[4] = -1.0;
    if (h1.t >= 0.0) {
        let p1 = l + h1.t * dw.xyz;
        let d1 = reflect(dw.xyz, h1.n);
        let r1 = material_r(h1.material);
        let rbar1 = (r1.x + r1.y + r1.z + r1.w) * 0.25;
        let path1 = 2.0 * h1.t;
        rec[0] = h1.t;
        rec[1] = d1.x;
        rec[2] = d1.y;
        rec[3] = d1.z;
        rec[8] = abs(dot(dw.xyz, h1.n));
        rec[9] = bitcast<f32>(h1.material);
        rec[10] = path1 / c;
        rec[11] = rbar1 * REFERENCE_M / max(path1, REFERENCE_M);
        rec[12] = dot(dw.xyz, right);
        // The second leg, from just off the face, split the same way.
        let o2 = p1 + h1.face * T_START;
        let h2 = nearest(lane, march(g, o2, d1, t0, t1));
        if (h2.t >= 0.0) {
            let p2 = o2 + h2.t * d1;
            let d2 = reflect(d1, h2.n);
            let r2 = material_r(h2.material);
            let rbar2 = (r2.x + r2.y + r2.z + r2.w) * 0.25;
            let path2 = h1.t + h2.t + length(p2 - l);
            rec[4] = h2.t;
            rec[5] = d2.x;
            rec[6] = d2.y;
            rec[7] = d2.z;
            rec[13] = path2 / c;
            rec[14] = rbar1 * rbar2 * REFERENCE_M / max(path2, REFERENCE_M);
            rec[15] = dot(normalize(p2 - l), right);
        }
    }
    if (lane < 16u) {
        outb[OUT_RAYS + 16u * ray + lane] = bitcast<u32>(rec[lane]);
    }
}

// ---------------------------------------------------------------------------------------
// The query: the 64 field rays and the sources in one dispatch, so they overlap. The rays
// go first because they are the longer.

@compute @workgroup_size(64)
fn query(@builtin(workgroup_id) wg: vec3<u32>,
         @builtin(local_invocation_index) lane: u32) {
    if (wg.x < FIELD_RAYS) {
        field_ray(wg.x, lane);
    } else {
        source_query(wg.x - FIELD_RAYS, lane);
    }
}

// ---------------------------------------------------------------------------------------
// The field's reduction: taps, the quadrature of 4V/S, the clear fraction and RT60, from
// the 64 rays' records.

var<workgroup> arr_delay: array<f32, 128>;
var<workgroup> arr_gain: array<f32, 128>;
var<workgroup> arr_pan: array<f32, 128>;
var<workgroup> bin_gain: array<atomic<u32>, 64>;
var<workgroup> bin_arrival: array<atomic<u32>, 64>;
var<workgroup> red_field: array<vec4<f32>, 64>;
var<workgroup> bin_plain: array<u32, 64>;
var<workgroup> bin_chosen: array<u32, 64>;

fn ray_word(ray: u32, k: u32) -> f32 {
    return bitcast<f32>(outb[OUT_RAYS + 16u * ray + k]);
}

@compute @workgroup_size(64)
fn field_reduce(@builtin(local_invocation_index) lane: u32) {
    let c = bitcast<vec4<f32>>(inb[HV_LISTENER]).w;
    let dw = FIELD_DIRS[lane];
    atomicStore(&bin_gain[lane], 0u);
    atomicStore(&bin_arrival[lane], 0xFFFFFFFFu);
    // Unused taps are zero: written first, and ordered before the chosen taps' writes.
    if (lane < 8u) {
        let at = OUT_FIELD + 4u * lane;
        outb[at] = 0u;
        outb[at + 1u] = 0u;
        outb[at + 2u] = 0u;
        outb[at + 3u] = 0u;
    }
    storageBarrier();
    let t1 = ray_word(lane, 0u);
    var sums = vec4<f32>(0.0, 0.0, 0.0, dw.w); // w l^3, w l^2 / |cos|, (that) x alpha, w escaped
    arr_gain[2u * lane] = 0.0;
    arr_gain[2u * lane + 1u] = 0.0;
    if (t1 >= 0.0) {
        let cosi = ray_word(lane, 8u);
        let r1 = material_r(outb[OUT_RAYS + 16u * lane + 9u]);
        let area = dw.w * t1 * t1 / max(cosi, COS_FLOOR);
        let alpha = 1.0 - r1.y * r1.y; // energy absorption at 500 Hz
        sums = vec4<f32>(dw.w * t1 * t1 * t1, area, area * alpha, 0.0);
        arr_delay[2u * lane] = ray_word(lane, 10u);
        arr_gain[2u * lane] = ray_word(lane, 11u);
        arr_pan[2u * lane] = ray_word(lane, 12u);
        if (ray_word(lane, 4u) >= 0.0) {
            arr_delay[2u * lane + 1u] = ray_word(lane, 13u);
            arr_gain[2u * lane + 1u] = ray_word(lane, 14u);
            arr_pan[2u * lane + 1u] = ray_word(lane, 15u);
        }
    }
    red_field[lane] = sums;
    workgroupBarrier();

    // Taps: the strongest arrival per 5 ms bin.
    for (var k = 0u; k < 2u; k = k + 1u) {
        let idx = 2u * lane + k;
        let gain = arr_gain[idx];
        let bin = u32(arr_delay[idx] / TAP_BIN_S);
        if (gain > 0.0 && bin < TAP_BINS) {
            atomicMax(&bin_gain[bin], bitcast<u32>(gain));
        }
    }
    workgroupBarrier();
    for (var k = 0u; k < 2u; k = k + 1u) {
        let idx = 2u * lane + k;
        let gain = arr_gain[idx];
        let bin = u32(arr_delay[idx] / TAP_BIN_S);
        if (gain > 0.0 && bin < TAP_BINS && bitcast<u32>(gain) == atomicLoad(&bin_gain[bin])) {
            atomicMin(&bin_arrival[bin], idx);
        }
    }
    for (var stride = 32u; stride > 0u; stride = stride >> 1u) {
        if (lane < stride) {
            red_field[lane] = red_field[lane] + red_field[lane + stride];
        }
        workgroupBarrier();
    }
    // The eight strongest bins: each lane counts the bins stronger than its own (ties to
    // the earlier bin), from a plain copy, and a bin with fewer than eight above it is a
    // tap. Taps go out in delay order: a tap's slot is the number of chosen bins before it.
    bin_plain[lane] = atomicLoad(&bin_gain[lane]);
    workgroupBarrier();
    let mine = bin_plain[lane];
    var above = 0u;
    for (var b = 0u; b < TAP_BINS; b = b + 1u) {
        let gb = bin_plain[b];
        above = above + select(0u, 1u, gb > mine || (gb == mine && b < lane));
    }
    let chosen = mine > 0u && above < 8u;
    bin_chosen[lane] = select(0u, 1u, chosen);
    workgroupBarrier();
    if (chosen) {
        var slot = 0u;
        for (var b = 0u; b < TAP_BINS; b = b + 1u) {
            slot = slot + select(0u, bin_chosen[b], b < lane);
        }
        let idx = atomicLoad(&bin_arrival[lane]);
        let at = OUT_FIELD + 4u * slot;
        outb[at] = bitcast<u32>(arr_delay[idx]);
        outb[at + 1u] = bitcast<u32>(arr_gain[idx]);
        outb[at + 2u] = bitcast<u32>(clamp(arr_pan[idx], -1.0, 1.0));
        outb[at + 3u] = 0u;
    }
    if (lane == 0u) {
        let s = red_field[0];
        var mfp = 0.0;
        var rt60 = 0.0;
        let clear = s.w / FIELD_TOTAL_WEIGHT;
        if (s.y > 0.0) {
            mfp = 4.0 * s.x / (3.0 * s.y);
            let a = clear + (1.0 - clear) * (s.z / s.y);
            if (a < 1.0) {
                rt60 = SABINE_LN * mfp * 0.25 / (c * -log(1.0 - a));
            }
        }
        outb[OUT_FIELD + 32u] = bitcast<u32>(mfp);
        outb[OUT_FIELD + 33u] = bitcast<u32>(rt60);
        outb[OUT_FIELD + 34u] = bitcast<u32>(clear);
        outb[OUT_FIELD + 35u] = 0u;
    }
}

// ---------------------------------------------------------------------------------------
// The law probe: the same `laws` over caller-supplied maxima, for the f32-against-f64
// oracle. Case i is four vectors: source and directivity, velocity and tag, the per-band
// excess, and the probe excess with the foliage path.

@compute @workgroup_size(64)
fn probe_laws(@builtin(workgroup_id) wg: vec3<u32>,
              @builtin(local_invocation_index) lane: u32) {
    let i = wg.x;
    if (i >= inb[HV_COUNTS].x) {
        return;
    }
    let base = HV_SOURCES + 4u * i;
    let a = bitcast<vec4<f32>>(inb[base]);
    let b = inb[base + 1u];
    let ex = bitcast<vec4<f32>>(inb[base + 2u]);
    let pf = bitcast<vec4<f32>>(inb[base + 3u]);
    laws(lane, a.xyz, a.w, bitcast<vec3<f32>>(b.xyz), b.w, ex, pf.x, pf.y, 8u * i);
}
