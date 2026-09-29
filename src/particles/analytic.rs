//! Closed-form motion for effect particles and loose pieces: nothing is integrated.
//!
//! [`crate::particles::ParticleEffects`] steps a pool: semi-implicit Euler, a damping factor
//! a step, a host-side ground pass. Every effect that pool carries obeys one law, **constant
//! acceleration plus linear drag toward a wind**, and that law has a closed form. So a
//! particle or a thrown piece can be written down once at birth, as its launch and the
//! time it lands, and evaluated at any later instant by anything that holds the record: a
//! CPU, a GPU compute pass, a replay. It is then **independent of the frame rate** by
//! construction (nothing accumulates a step), and costs nothing between birth and the
//! frame that draws it.
//!
//! # The law
//!
//! `dv/dt = a - k (v - w)`: gravity (or buoyancy) `a`, a linear drag rate `k` (1/s) acting on
//! the velocity relative to the air, and the air's velocity `w`. Its solution, with
//! `e1(t) = (1 - e^(-k t)) / k` and `e2(t) = (t - e1(t)) / k`:
//!
//! ```text
//! v(t) = w + (v0 - w) e^(-k t) + a e1(t)
//! x(t) = x0 + w t + (v0 - w) e1(t) + a e2(t)
//! ```
//!
//! At `k = 0` the limits are `e1 = t`, `e2 = t^2 / 2`: plain ballistics, which is what a
//! thrown body part is (its linear drag coefficient is a few per cent a second; see
//! the Ridgeline corpse module's arithmetic). [`decay_integral`] and [`decay_integral2`]
//! take the limits exactly and stay accurate for small `k t` where the textbook form
//! cancels.
//!
//! # What happens at the ground
//!
//! The ground is the only interaction, and it is solved **once, at birth**: [`land`]
//! marches the arc against the host's height function and refines the crossing by
//! bisection. After that a caller has three closed forms to choose from:
//!
//! * [`bounce`]: one reflection off the landing plane, with a restitution along the
//!   normal and a factor kept along the plane (a spark);
//! * [`slide_distance`] and [`rest_time`]: an exponential slide to rest at a rate `k`
//!   (a piece of debris, which does not bounce);
//! * nothing: the particle dies on contact (a drop that becomes a stain).
//!
//! # Randomness
//!
//! [`pcg_hash`] is a stateless integer hash: a record's randomness is a function of its
//! identity (an event id, a tick, an index), so no generator state crosses a thread and the
//! same event emits the same records in a replay. [`unit_float`] turns it into `[0, 1)`.
//! The GPU evaluator of a record reproduces these functions bit for bit on integers and to
//! float rounding on the rest.
//!
//! # Example
//!
//! ```
//! use rs_physics::particles::analytic::{land, Launch};
//!
//! // A piece thrown up and out from a metre above flat ground.
//! let launch = Launch::ballistic([0.0, 1.0, 0.0], [3.0, 4.0, 0.0], 9.81);
//! let landing = land(&launch, 0.0, 5.0, 64, |_, _| 0.0).expect("it comes down");
//! // Ballistics: y = 1 + 4 t - 4.905 t^2 = 0 at t = (4 + sqrt(16 + 19.62)) / 9.81.
//! let exact = (4.0 + (16.0f32 + 19.62).sqrt()) / 9.81;
//! assert!((landing.t - exact).abs() < 1e-4);
//! assert!((landing.position[0] - 3.0 * exact).abs() < 1e-3);
//! ```

/// Below this `k t` the decay integrals are taken by their series: `e2`'s textbook form
/// `(t - e1) / k` loses about `-log2(x)` bits to cancellation (a 1e-4 relative error at
/// `x = 1e-3` in `f32`), and a shader has no `expm1` for `e1`. At `x = 0.5` the series'
/// first omitted term is `x^10 / 12!`, under `f32`'s epsilon, and the textbook forms have
/// lost under two bits.
pub const SERIES_BELOW: f32 = 0.5;

/// Terms of the series [`decay_integral`] and [`decay_integral2`] sum below
/// [`SERIES_BELOW`].
pub const SERIES_TERMS: u32 = 10;

/// Bisection steps [`land`] takes after the march brackets a crossing: 2^-20 of a march
/// step, far under a millimetre for any step a caller would choose.
pub const REFINE_STEPS: u32 = 20;

/// `(1 - e^(-k t)) / k`: the time-integral of `e^(-k s)` over `[0, t]`, seconds.
///
/// Exact at `k = 0` (it is `t`), and taken by its series for `k t` under
/// [`SERIES_BELOW`], where `1 - e^(-k t)` would cancel.
///
/// # Arguments
///
/// * `k` - the decay rate, 1/s, at least 0.
/// * `t` - the time, seconds, at least 0.
///
/// # Returns
///
/// The integral, seconds; `t` when `k` is 0 and `1 / k` as `t` grows.
///
/// # Examples
///
/// ```
/// use rs_physics::particles::analytic::decay_integral;
/// assert_eq!(decay_integral(0.0, 2.0), 2.0);
/// assert!((decay_integral(5.5, 100.0) - 1.0 / 5.5).abs() < 1e-6);
/// ```
#[inline]
pub fn decay_integral(k: f32, t: f32) -> f32 {
    let x = k * t;
    if x < SERIES_BELOW {
        // t sum (-x)^n / (n + 1)!.
        series(x, 1) * t
    } else {
        -(-x).exp_m1() / k
    }
}

/// `(t - e1(t)) / k`, where `e1` is [`decay_integral`]: the double integral of
/// `e^(-k s)`, seconds squared. It carries a constant acceleration's displacement under
/// drag.
///
/// # Arguments
///
/// * `k` - the decay rate, 1/s, at least 0.
/// * `t` - the time, seconds, at least 0.
///
/// # Returns
///
/// The integral, s^2; `t^2 / 2` when `k` is 0.
///
/// # Examples
///
/// ```
/// use rs_physics::particles::analytic::decay_integral2;
/// assert_eq!(decay_integral2(0.0, 2.0), 2.0);
/// ```
#[inline]
pub fn decay_integral2(k: f32, t: f32) -> f32 {
    let x = k * t;
    if x < SERIES_BELOW {
        series(x, 2) * t * t
    } else {
        (t - decay_integral(k, t)) / k
    }
}

/// `sum_{n < SERIES_TERMS} (-x)^n / (n + m)!` by Horner: each term is the last times
/// `-x / (n + m)`, so the sum is `(1/m!) (1 - x/(m+1) (1 - x/(m+2) (1 - ...)))`.
#[inline]
fn series(x: f32, m: u32) -> f32 {
    let mut sum = 1.0f32;
    for n in (1..SERIES_TERMS).rev() {
        sum = 1.0 - x * sum / (n + m) as f32;
    }
    let mut first = 1.0f32;
    for i in 2..=m {
        first /= i as f32;
    }
    sum * first
}

/// A launch: everything the closed form needs, in the host's world units (metres,
/// seconds).
///
/// # Examples
///
/// ```
/// use rs_physics::particles::analytic::Launch;
/// let l = Launch::ballistic([0.0; 3], [1.0, 0.0, 0.0], 9.81);
/// assert_eq!(l.position(1.0)[0], 1.0);
/// assert!((l.position(1.0)[1] + 4.905).abs() < 1e-6);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Launch {
    /// Where it starts, m.
    pub origin: [f32; 3],
    /// Its velocity at birth, m/s.
    pub velocity: [f32; 3],
    /// The constant acceleration on it, m/s^2: gravity, or a hot gas's buoyancy.
    pub acceleration: [f32; 3],
    /// The linear drag rate, 1/s: how fast its velocity relaxes toward `wind`. 0 for
    /// plain ballistics.
    pub drag: f32,
    /// The air's velocity it relaxes toward, m/s. Ignored when `drag` is 0.
    pub wind: [f32; 3],
}

impl Launch {
    /// A launch under gravity alone: no drag, no wind.
    ///
    /// # Arguments
    ///
    /// * `origin` - where it starts, m.
    /// * `velocity` - its velocity at birth, m/s.
    /// * `gravity` - the downward acceleration, m/s^2 (positive; applied along -y).
    ///
    /// # Returns
    ///
    /// The launch.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::particles::analytic::Launch;
    /// let l = Launch::ballistic([0.0, 2.0, 0.0], [0.0; 3], 9.81);
    /// assert_eq!(l.acceleration, [0.0, -9.81, 0.0]);
    /// ```
    pub fn ballistic(origin: [f32; 3], velocity: [f32; 3], gravity: f32) -> Launch {
        Launch {
            origin,
            velocity,
            acceleration: [0.0, -gravity, 0.0],
            drag: 0.0,
            wind: [0.0; 3],
        }
    }

    /// Where it is `t` seconds after birth, in flight (the ground is the caller's:
    /// see [`land`]).
    ///
    /// # Arguments
    ///
    /// * `t` - seconds since birth, at least 0.
    ///
    /// # Returns
    ///
    /// The position, m.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::particles::analytic::Launch;
    /// let mut l = Launch::ballistic([0.0; 3], [10.0, 0.0, 0.0], 0.0);
    /// l.drag = 2.0;
    /// // Linear drag with no wind stops it at v0 / k.
    /// assert!((l.position(60.0)[0] - 5.0).abs() < 1e-4);
    /// ```
    #[inline]
    pub fn position(&self, t: f32) -> [f32; 3] {
        let e1 = decay_integral(self.drag, t);
        let e2 = decay_integral2(self.drag, t);
        let (w, wt) = if self.drag > 0.0 { (self.wind, t) } else { ([0.0; 3], 0.0) };
        let mut out = [0.0; 3];
        for i in 0..3 {
            out[i] = self.origin[i] + w[i] * wt + (self.velocity[i] - w[i]) * e1 + self.acceleration[i] * e2;
        }
        out
    }

    /// Its velocity `t` seconds after birth, in flight.
    ///
    /// # Arguments
    ///
    /// * `t` - seconds since birth, at least 0.
    ///
    /// # Returns
    ///
    /// The velocity, m/s.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::particles::analytic::Launch;
    /// let l = Launch::ballistic([0.0; 3], [0.0, 5.0, 0.0], 9.81);
    /// assert!((l.velocity_at(0.5)[1] - (5.0 - 4.905)).abs() < 1e-5);
    /// ```
    #[inline]
    pub fn velocity_at(&self, t: f32) -> [f32; 3] {
        let decay = (-self.drag * t).exp();
        let e1 = decay_integral(self.drag, t);
        let w = if self.drag > 0.0 { self.wind } else { [0.0; 3] };
        let mut out = [0.0; 3];
        for i in 0..3 {
            out[i] = w[i] + (self.velocity[i] - w[i]) * decay + self.acceleration[i] * e1;
        }
        out
    }
}

/// Where and when a launch meets the ground.
///
/// # Examples
///
/// ```
/// use rs_physics::particles::analytic::{land, Launch};
/// let l = Launch::ballistic([0.0, 5.0, 0.0], [0.0; 3], 9.81);
/// let hit = land(&l, 0.0, 2.0, 32, |_, _| 0.0).unwrap();
/// assert_eq!(hit.normal, [0.0, 1.0, 0.0]);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Landing {
    /// Seconds after birth.
    pub t: f32,
    /// Its position then, m: `clearance` over the ground under it.
    pub position: [f32; 3],
    /// The ground's unit normal under it, from the height function's own central
    /// differences.
    pub normal: [f32; 3],
}

/// **Find where a launch comes down.** March the arc in `steps` equal steps over
/// `[0, t_max]`; at the first step whose point is within `clearance` of the ground, bisect
/// the bracket [`REFINE_STEPS`] times. The ground is the host's (`height(x, z)`), sampled
/// only here, at birth.
///
/// A point already at or under the ground at birth lands at `t = 0`. A march step
/// longer than a bump can step over it; the caller sizes `steps` against its ground's
/// cell and the launch's speed.
///
/// # Arguments
///
/// * `launch` - the launch.
/// * `clearance` - how far over the ground its reference point is when it touches, m
///   (0 for a point; a body's centre sits its own half-thickness up).
/// * `t_max` - the latest landing looked for, seconds.
/// * `steps` - march steps over `t_max`, at least 1.
/// * `height` - the ground's height at `(x, z)`, m.
///
/// # Returns
///
/// The landing, or `None` when it is still in the air at `t_max`.
///
/// # Examples
///
/// ```
/// use rs_physics::particles::analytic::{land, Launch};
/// // Onto a 1-in-10 slope rising along +x.
/// let l = Launch::ballistic([0.0, 3.0, 0.0], [2.0, 0.0, 0.0], 9.81);
/// let hit = land(&l, 0.1, 3.0, 48, |x, _| 0.1 * x).unwrap();
/// assert!((hit.position[1] - (0.1 * hit.position[0] + 0.1)).abs() < 1e-4);
/// assert!(hit.normal[0] < 0.0 && hit.normal[1] > 0.99);
/// ```
pub fn land(launch: &Launch, clearance: f32, t_max: f32, steps: u32, height: impl Fn(f32, f32) -> f32) -> Option<Landing> {
    let gap = |t: f32| {
        let p = launch.position(t);
        p[1] - clearance - height(p[0], p[2])
    };
    let steps = steps.max(1);
    let dt = t_max / steps as f32;
    let (mut lo, mut hi) = (0.0f32, f32::NAN);
    if gap(0.0) <= 0.0 {
        hi = 0.0;
    } else {
        for i in 1..=steps {
            let t = dt * i as f32;
            if gap(t) <= 0.0 {
                hi = t;
                break;
            }
            lo = t;
        }
    }
    if hi.is_nan() {
        return None;
    }
    if hi > 0.0 {
        for _ in 0..REFINE_STEPS {
            let mid = 0.5 * (lo + hi);
            if gap(mid) <= 0.0 {
                hi = mid;
            } else {
                lo = mid;
            }
        }
    }
    let mut position = launch.position(hi);
    let ground = height(position[0], position[2]);
    position[1] = ground + clearance;
    Some(Landing {
        t: hi,
        position,
        normal: ground_normal(position[0], position[2], NORMAL_SPAN_M, &height),
    })
}

/// The half-span [`land`] takes its central differences over, m: a centimetre, finer than
/// any ground a host samples and far over `f32`'s resolution at map scale.
pub const NORMAL_SPAN_M: f32 = 0.01;

/// The unit normal of a height function at `(x, z)`, by central differences over
/// `span` either side.
///
/// # Arguments
///
/// * `x`, `z` - where, m.
/// * `span` - the half-span of the differences, m, above 0.
/// * `height` - the ground's height at `(x, z)`, m.
///
/// # Returns
///
/// `(-dh/dx, 1, -dh/dz)` normalised.
///
/// # Examples
///
/// ```
/// use rs_physics::particles::analytic::ground_normal;
/// let n = ground_normal(0.0, 0.0, 0.5, |_, z| z);
/// let s = std::f32::consts::FRAC_1_SQRT_2;
/// assert!((n[1] - s).abs() < 1e-5 && (n[2] + s).abs() < 1e-5);
/// ```
pub fn ground_normal(x: f32, z: f32, span: f32, height: impl Fn(f32, f32) -> f32) -> [f32; 3] {
    let dx = height(x + span, z) - height(x - span, z);
    let dz = height(x, z + span) - height(x, z - span);
    let n = [-dx, 2.0 * span, -dz];
    let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    if len > 0.0 && len.is_finite() {
        [n[0] / len, n[1] / len, n[2] / len]
    } else {
        [0.0, 1.0, 0.0]
    }
}

/// **One bounce off a plane**: the velocity's component into the plane reversed and
/// scaled by `restitution`, the component along it scaled by `tangential`.
///
/// # Arguments
///
/// * `velocity` - arriving, m/s.
/// * `normal` - the plane's unit normal.
/// * `restitution` - kept of the normal speed, 0 to 1.
/// * `tangential` - kept of the speed along the plane, 0 to 1.
///
/// # Returns
///
/// The leaving velocity, m/s. A velocity already leaving the plane is returned as it is.
///
/// # Examples
///
/// ```
/// use rs_physics::particles::analytic::bounce;
/// let v = bounce([2.0, -4.0, 0.0], [0.0, 1.0, 0.0], 0.5, 0.55);
/// assert!((v[0] - 1.1).abs() < 1e-6 && (v[1] - 2.0).abs() < 1e-6);
/// ```
pub fn bounce(velocity: [f32; 3], normal: [f32; 3], restitution: f32, tangential: f32) -> [f32; 3] {
    let vn = velocity[0] * normal[0] + velocity[1] * normal[1] + velocity[2] * normal[2];
    if vn >= 0.0 {
        return velocity;
    }
    let mut out = [0.0; 3];
    for i in 0..3 {
        let along_n = vn * normal[i];
        let along_t = velocity[i] - along_n;
        out[i] = along_t * tangential - along_n * restitution;
    }
    out
}

/// How far a slide with exponential drag `k` has gone `t` seconds in, per unit of the
/// speed it started at: [`decay_integral`], named for what it is here.
///
/// # Arguments
///
/// * `k` - the ground drag, 1/s.
/// * `t` - seconds since the slide began.
///
/// # Returns
///
/// Metres per (m/s) of starting speed; `1 / k` as `t` grows.
///
/// # Examples
///
/// ```
/// use rs_physics::particles::analytic::slide_distance;
/// assert!((slide_distance(5.5, 10.0) * 5.5 - 1.0).abs() < 1e-6);
/// ```
#[inline]
pub fn slide_distance(k: f32, t: f32) -> f32 {
    decay_integral(k, t)
}

/// **When an exponential decay passes under a threshold**: `ln(speed / rest) / k`, the
/// moment a slide at drag `k` starting at `speed` falls to `rest`.
///
/// # Arguments
///
/// * `speed` - the starting speed (or spin rate), any unit.
/// * `k` - the decay rate, 1/s, above 0.
/// * `rest` - the threshold, the same unit as `speed`, above 0.
///
/// # Returns
///
/// Seconds; 0 when it starts at or under the threshold.
///
/// # Examples
///
/// ```
/// use rs_physics::particles::analytic::rest_time;
/// assert!((rest_time(0.12 * std::f32::consts::E, 5.5, 0.12) - 1.0 / 5.5).abs() < 1e-6);
/// assert_eq!(rest_time(0.1, 5.5, 0.12), 0.0);
/// ```
#[inline]
pub fn rest_time(speed: f32, k: f32, rest: f32) -> f32 {
    if speed <= rest || k <= 0.0 {
        0.0
    } else {
        (speed / rest).ln() / k
    }
}

/// **A stateless 32-bit hash** (PCG's RXS-M-XS output over one LCG step, Jarzynski and
/// Olano 2020, "Hash Functions for GPU Rendering"): a record's randomness from its
/// identity, the same on the CPU and in a shader.
///
/// # Arguments
///
/// * `x` - the input word (combine several with [`pcg_hash`] of an xor or a sum).
///
/// # Returns
///
/// The hash.
///
/// # Examples
///
/// ```
/// use rs_physics::particles::analytic::pcg_hash;
/// assert_eq!(pcg_hash(7), pcg_hash(7));
/// assert_ne!(pcg_hash(1), pcg_hash(2));
/// ```
#[inline]
pub fn pcg_hash(x: u32) -> u32 {
    let state = x.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
    let word = ((state >> ((state >> 28) + 4)) ^ state).wrapping_mul(277_803_737);
    (word >> 22) ^ word
}

/// A hash's top 24 bits as a float in `[0, 1)`: exact in `f32`, so a shader reproduces it
/// bit for bit.
///
/// # Arguments
///
/// * `hash` - a [`pcg_hash`] output.
///
/// # Returns
///
/// `(hash >> 8) / 2^24`.
///
/// # Examples
///
/// ```
/// use rs_physics::particles::analytic::unit_float;
/// assert_eq!(unit_float(0), 0.0);
/// assert!(unit_float(u32::MAX) < 1.0);
/// ```
#[inline]
pub fn unit_float(hash: u32) -> f32 {
    (hash >> 8) as f32 * (1.0 / 16_777_216.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 1 ms semi-implicit Euler reference of the same law: the closed form must match it
    /// to its own truncation error.
    fn euler(l: &Launch, t: f32) -> [f32; 3] {
        let h = 1e-4f64;
        let n = (t as f64 / h).round() as usize;
        let (mut x, mut v) = (l.origin.map(|c| c as f64), l.velocity.map(|c| c as f64));
        for _ in 0..n {
            for i in 0..3 {
                let a = l.acceleration[i] as f64 - l.drag as f64 * (v[i] - l.wind[i] as f64);
                v[i] += a * h;
                x[i] += v[i] * h;
            }
        }
        x.map(|c| c as f32)
    }

    #[test]
    fn the_closed_form_is_the_law() {
        let mut state = 12345u32;
        let mut next = || {
            state = pcg_hash(state);
            unit_float(state) * 2.0 - 1.0
        };
        for _ in 0..40 {
            let l = Launch {
                origin: [next() * 10.0, 2.0 + next(), next() * 10.0],
                velocity: [next() * 8.0, next() * 8.0, next() * 8.0],
                acceleration: [0.0, -9.81, 0.0],
                drag: (next() + 1.0) * 2.0,
                wind: [next() * 3.0, 0.0, next() * 3.0],
            };
            for &t in &[0.0f32, 0.05, 0.4, 1.3] {
                let (a, b) = (l.position(t), euler(&l, t));
                for i in 0..3 {
                    assert!((a[i] - b[i]).abs() < 5e-3, "t {t}: {a:?} against {b:?}");
                }
            }
        }
    }

    #[test]
    fn no_drag_is_ballistics_exactly() {
        let l = Launch::ballistic([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], 9.81);
        let t = 0.75f32;
        let p = l.position(t);
        assert_eq!(p[0], 1.0 + 4.0 * t);
        assert!((p[1] - (2.0 + 5.0 * t - 0.5 * 9.81 * t * t)).abs() < 1e-6);
        let v = l.velocity_at(t);
        assert!((v[1] - (5.0 - 9.81 * t)).abs() < 1e-6);
    }

    #[test]
    fn the_series_meets_the_exact_form() {
        // Either side of the switch the two agree to f32 rounding.
        for &k in &[1e-6f32, 1e-3, 0.5, 5.5] {
            for &t in &[1e-4f32, 1e-3, 0.1, 1.0] {
                // In f64 and by forms that do not cancel: expm1 for e1, and e2's own series
                // t^2 sum (-x)^n / (n + 2)!, which converges fast for every x here.
                let x = k as f64 * t as f64;
                let exact = -(-x).exp_m1() / k as f64;
                let (mut term, mut exact2) = (0.5f64, 0.0f64);
                for n in 0..40 {
                    exact2 += term;
                    term *= -x / (n as f64 + 3.0);
                }
                exact2 *= t as f64 * t as f64;
                assert!((decay_integral(k, t) as f64 - exact).abs() <= 1e-6 * exact.max(1e-3), "e1 k {k} t {t}");
                assert!((decay_integral2(k, t) as f64 - exact2).abs() <= 1e-4 * exact2.max(1e-6), "e2 k {k} t {t}");
            }
        }
    }

    #[test]
    fn the_landing_is_on_the_ground() {
        let bumps = |x: f32, z: f32| 0.3 * (x * 1.7).sin() * (z * 0.9).cos();
        for i in 0..32 {
            let a = i as f32 * 0.4;
            let l = Launch::ballistic([0.0, 2.0, 0.0], [3.0 * a.cos(), 3.0, 3.0 * a.sin()], 9.81);
            let hit = land(&l, 0.05, 4.0, 200, bumps).expect("lands");
            let p = hit.position;
            assert!((p[1] - bumps(p[0], p[2]) - 0.05).abs() < 1e-5);
            // The refined time's own position is within the bisection's width of it.
            let q = l.position(hit.t);
            assert!(((q[0] - p[0]).powi(2) + (q[2] - p[2]).powi(2)).sqrt() < 1e-4);
        }
    }

    #[test]
    fn a_launch_that_never_comes_down_says_so() {
        let l = Launch::ballistic([0.0, 1.0, 0.0], [0.0, 50.0, 0.0], 9.81);
        assert!(land(&l, 0.0, 1.0, 16, |_, _| 0.0).is_none());
    }

    #[test]
    fn a_bounce_keeps_what_it_says() {
        let n = [0.0, 1.0, 0.0];
        let v = bounce([3.0, -2.0, -1.0], n, 0.32, 0.55);
        assert!((v[1] - 0.64).abs() < 1e-6);
        assert!((v[0] - 1.65).abs() < 1e-6 && (v[2] + 0.55).abs() < 1e-6);
    }

    #[test]
    fn the_hash_is_the_reference() {
        // Jarzynski and Olano's pcg, restated from the paper's listing as its own oracle:
        // `state = v * 747796405u + 2891336453u; word = ((state >> ((state >> 28u) + 4u)) ^
        // state) * 277803737u; return (word >> 22u) ^ word;`.
        let reference = |v: u32| {
            let state = v.wrapping_mul(747796405).wrapping_add(2891336453);
            let word = ((state >> ((state >> 28) + 4)) ^ state).wrapping_mul(277803737);
            (word >> 22) ^ word
        };
        for v in [0u32, 1, 2, 12345, u32::MAX] {
            assert_eq!(pcg_hash(v), reference(v));
        }
        // And a spread of outputs over 2^16 inputs.
        let mut buckets = [0u32; 16];
        for i in 0..65_536u32 {
            buckets[(pcg_hash(i) >> 28) as usize] += 1;
        }
        for b in buckets {
            assert!((b as i64 - 4096).abs() < 300, "{buckets:?}");
        }
    }
}
