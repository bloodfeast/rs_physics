//! Regression tests from the 2026-09-29 joint/constraint solver review
//! (`docs/reviews/2026-09-29-correctness-performance.md`).
//!
//! Each test pins one defect against an analytic expectation (pendulum period,
//! free-fall distance, energy or momentum that must not be created, a
//! projection that must land exactly on the constraint). All but the first
//! (a control) failed on the code as reviewed. The `#[ignore]`d ones pin
//! defects that are documented in the review but not fixed yet; run them with
//! `cargo test --lib -- --ignored` to see the current behaviour.

use std::cell::RefCell;
use std::rc::Rc;

use crate::constraints::{
    Constraint3D, ConstraintSolver, Contact3D, ContactPoint3D, Fixed3D, Hinge3D, Joint3D, Rope3D,
    RopeChain3D, Spring, Spring3D, UnifiedSolver3D,
};
use crate::models::{Object, ObjectIn3D, PhysicalObject3D, Shape3D};
use crate::utils::{PhysicsConstants, PhysicsError};
use crate::world::{
    PhysicsWorld, WorldConfig, WorldConstraint, WorldJoint3D, WorldRope3D, WorldSpring3D,
};

const G: f64 = 9.81;

fn obj3(mass: f64, pos: (f64, f64, f64), vel: (f64, f64, f64)) -> ObjectIn3D {
    ObjectIn3D::new(mass, vel.0, vel.1, vel.2, pos)
}

fn ball(mass: f64, pos: (f64, f64, f64)) -> PhysicalObject3D {
    PhysicalObject3D::new(
        mass,
        (0.0, 0.0, 0.0),
        pos,
        Shape3D::Sphere(0.1),
        None,
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        PhysicsConstants::default(),
    )
}

fn quiet_world(hz: f64) -> PhysicsWorld {
    PhysicsWorld::new(
        WorldConfig::default()
            .with_frequency(hz)
            .with_gravity(0.0, -G, 0.0)
            .without_aerodynamic_drag()
            .with_real_time(false),
    )
}

/// Period from the first two positive-to-negative zero crossings of `x(t)`.
fn period_from(samples: &[(f64, f64)]) -> f64 {
    let mut crossings = Vec::new();
    for w in samples.windows(2) {
        let ((t0, x0), (t1, x1)) = (w[0], w[1]);
        if x0 > 0.0 && x1 <= 0.0 {
            crossings.push(t0 + (t1 - t0) * x0 / (x0 - x1));
        }
    }
    assert!(crossings.len() >= 2, "need two crossings, got {}", crossings.len());
    crossings[1] - crossings[0]
}

/// Small-angle pendulum period, with the first finite-amplitude correction.
fn pendulum_period(length: f64, theta0: f64) -> f64 {
    2.0 * std::f64::consts::PI * (length / G).sqrt() * (1.0 + theta0 * theta0 / 16.0)
}

/// Undamped 1 m hinge pendulum released from rest `theta0` from vertical.
fn hinge_pendulum(theta0: f64) -> Hinge3D {
    let anchor = (0.0, 0.0, 0.0);
    let frame = obj3(f64::INFINITY, anchor, (0.0, 0.0, 0.0));
    let bob = obj3(1.0, (theta0.sin(), -theta0.cos(), 0.0), (0.0, 0.0, 0.0));
    Hinge3D::new(frame, bob, anchor, (0.0, 0.0, 1.0))
        .expect("valid hinge")
        .with_angular_damping(0.0)
}

// ---------------------------------------------------------------------------
// Hinge3D
// ---------------------------------------------------------------------------

/// Control: driven directly, once per step, the hinge gets the period right.
/// (Passes. It isolates the world/solver failures below to how they call it.)
#[test]
fn review_hinge_standalone_period_matches_analytic() {
    let mut hinge = hinge_pendulum(0.1);
    let dt = 1e-3;
    let mut samples = Vec::new();
    for i in 0..6000 {
        hinge.solve_with_gravity(dt, -G).unwrap();
        samples.push(((i + 1) as f64 * dt, hinge.object2.position.x));
    }
    let expected = pendulum_period(1.0, 0.1);
    let measured = period_from(&samples);
    assert!(
        (measured - expected).abs() / expected < 0.01,
        "standalone hinge period {measured} vs analytic {expected}"
    );
}

/// In a PhysicsWorld the hinge is integrated once per *solver iteration*
/// (8 by default), so hinge time runs 8x faster than world time.
#[test]
fn review_world_hinge_pendulum_period_matches_analytic() {
    let mut world = quiet_world(240.0);
    let id = world.add_constraint(WorldConstraint::Hinge(hinge_pendulum(0.1)));
    let dt = 1.0 / 240.0;
    let mut samples = Vec::new();
    for i in 0..(240 * 6) {
        world.step();
        let x = match world.get_constraint(id) {
            Some(WorldConstraint::Hinge(h)) => h.object2.position.x,
            _ => unreachable!(),
        };
        samples.push(((i + 1) as f64 * dt, x));
    }
    let expected = pendulum_period(1.0, 0.1);
    let measured = period_from(&samples);
    assert!(
        (measured - expected).abs() / expected < 0.02,
        "world hinge pendulum period {measured:.4} s, analytic {expected:.4} s (ratio {:.3})",
        expected / measured
    );
}

/// Lets a test observe a constraint after handing it to UnifiedSolver3D,
/// which has no API to read a constraint (or its bodies) back.
struct Shared<T>(Rc<RefCell<T>>);

impl Constraint3D for Shared<Hinge3D> {
    fn solve(&mut self, dt: f64) -> Result<(), PhysicsError> {
        self.0.borrow_mut().solve(dt)
    }
    fn calculate_error(&self) -> f64 {
        self.0.borrow().calculate_error()
    }
    fn get_lambda(&self) -> f64 {
        self.0.borrow().lambda
    }
    fn set_lambda(&mut self, l: f64) {
        self.0.borrow_mut().lambda = l;
    }
}

impl Constraint3D for Shared<Spring3D> {
    fn solve(&mut self, dt: f64) -> Result<(), PhysicsError> {
        self.0.borrow_mut().solve(dt)
    }
    fn calculate_error(&self) -> f64 {
        self.0.borrow().calculate_error()
    }
    fn get_lambda(&self) -> f64 {
        self.0.borrow().lambda
    }
    fn set_lambda(&mut self, l: f64) {
        self.0.borrow_mut().lambda = l;
    }
}

impl Constraint3D for Shared<Joint3D> {
    fn solve(&mut self, dt: f64) -> Result<(), PhysicsError> {
        self.0.borrow_mut().solve(dt)
    }
    fn calculate_error(&self) -> f64 {
        self.0.borrow().calculate_error()
    }
    fn get_lambda(&self) -> f64 {
        self.0.borrow().lambda
    }
    fn set_lambda(&mut self, l: f64) {
        self.0.borrow_mut().lambda = l;
    }
}

/// One `UnifiedSolver3D::solve(dt)` must advance a hinge by one `dt`, not by
/// one `dt` per iteration.
#[test]
#[ignore = "known defect, not fixed in this change: see docs/reviews/2026-09-29-correctness-performance.md"]
fn review_unified_solver_advances_hinge_by_one_dt_per_solve() {
    // Horizontal 1 m arm released from rest: alpha = g / L at t = 0.
    let anchor = (0.0, 0.0, 0.0);
    let frame = obj3(f64::INFINITY, anchor, (0.0, 0.0, 0.0));
    let bob = obj3(1.0, (1.0, 0.0, 0.0), (0.0, 0.0, 0.0));
    let hinge = Rc::new(RefCell::new(
        Hinge3D::new(frame, bob, anchor, (0.0, 0.0, 1.0))
            .unwrap()
            .with_angular_damping(0.0),
    ));
    let mut solver = UnifiedSolver3D::new(10, 1e-6).unwrap();
    solver.add_constraint(Box::new(Shared(hinge.clone())));

    let dt = 0.01;
    let result = solver.solve(dt).unwrap();
    let omega = hinge.borrow().angular_velocity.abs();
    let expected = G / 1.0 * dt;
    assert!(
        (omega - expected).abs() / expected < 0.01,
        "after one solve(dt) |omega| = {omega:.5}, expected g/L*dt = {expected:.5} \
         ({} solver iterations ran)",
        result.iterations
    );
}

/// Same for Spring3D, which also integrates positions inside `solve`.
#[test]
#[ignore = "known defect, not fixed in this change: see docs/reviews/2026-09-29-correctness-performance.md"]
fn review_unified_solver_advances_spring_by_one_dt_per_solve() {
    let anchor = obj3(f64::INFINITY, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0));
    let bob = obj3(1.0, (1.1, 0.0, 0.0), (0.0, 0.0, 0.0)); // stretched 0.1 m
    let spring = Rc::new(RefCell::new(Spring3D::new(anchor, bob, 100.0, 1.0, 0.0).unwrap()));
    let mut solver = UnifiedSolver3D::new(10, 1e-12).unwrap();
    solver.add_constraint(Box::new(Shared(spring.clone())));

    let dt = 1e-3;
    solver.solve(dt).unwrap();
    let v = spring.borrow().object2.velocity.x;
    let expected = -100.0 * 0.1 / 1.0 * dt; // a = -k x / m
    assert!(
        (v - expected).abs() / expected.abs() < 0.01,
        "after one solve(dt) v = {v:.5} m/s, expected -k x dt / m = {expected:.5} m/s"
    );
}

/// Angular damping is `omega *= 1 - c` per call, so its effect depends on the
/// timestep. One simulated second must look the same at 60 Hz and 600 Hz.
#[test]
fn review_hinge_damping_is_timestep_independent() {
    let run = |hz: usize| -> f64 {
        let anchor = (0.0, 0.0, 0.0);
        let frame = obj3(f64::INFINITY, anchor, (0.0, 0.0, 0.0));
        let bob = obj3(1.0, (1.0, 0.0, 0.0), (0.0, 0.0, 0.0));
        let mut h = Hinge3D::new(frame, bob, anchor, (0.0, 0.0, 1.0)).unwrap(); // default damping
        let dt = 1.0 / hz as f64;
        let mut max_speed: f64 = 0.0;
        for _ in 0..hz {
            h.solve_with_gravity(dt, -G).unwrap();
            max_speed = max_speed.max(h.angular_velocity.abs());
        }
        max_speed
    };
    let w60 = run(60);
    let w600 = run(600);
    assert!(
        (w60 - w600).abs() / w60.max(w600) < 0.1,
        "peak |omega| over 1 s: {w60:.4} rad/s at 60 Hz vs {w600:.4} rad/s at 600 Hz"
    );
}

/// A hinge allows rotation *about* its axis; the body's offset *along* the
/// axis must survive. The arm is projected onto the swing plane at
/// construction and the position is rebuilt from that projection.
#[test]
fn review_hinge_preserves_offset_along_axis() {
    // Hinge line is the x axis through the origin; the door's centre sits
    // 0.5 m along the hinge line and 1 m out from it.
    let anchor = (0.0, 0.0, 0.0);
    let frame = obj3(f64::INFINITY, anchor, (0.0, 0.0, 0.0));
    let door = obj3(2.0, (0.5, 0.0, 1.0), (0.0, 0.0, 0.0));
    let mut hinge = Hinge3D::new(frame, door, anchor, (1.0, 0.0, 0.0)).unwrap();

    hinge.solve_with_gravity(1e-4, -G).unwrap();

    let x = hinge.object2.position.x;
    assert!((x - 0.5).abs() < 1e-9, "rotation about x must not change x: expected 0.5, got {x}");
}

/// With a dynamic frame, the hinge reaction must act on object1: a free pivot
/// plus bob, with no external horizontal force, keeps zero horizontal momentum.
#[test]
#[ignore = "known defect, not fixed in this change: see docs/reviews/2026-09-29-correctness-performance.md"]
fn review_hinge_applies_reaction_to_dynamic_object1() {
    let anchor = (0.0, 0.0, 0.0);
    let frame = obj3(1.0, anchor, (0.0, 0.0, 0.0));
    let bob = obj3(1.0, (1.0, 0.0, 0.0), (0.0, 0.0, 0.0));
    let mut hinge = Hinge3D::new(frame, bob, anchor, (0.0, 0.0, 1.0))
        .unwrap()
        .with_angular_damping(0.0);
    for _ in 0..200 {
        hinge.solve_with_gravity(1e-3, -G).unwrap();
    }
    let px = hinge.object1.mass * hinge.object1.velocity.x
        + hinge.object2.mass * hinge.object2.velocity.x;
    assert!(
        px.abs() < 1e-6,
        "horizontal momentum must stay 0, got {px:.4} kg m/s (object1 v = {:?})",
        (hinge.object1.velocity.x, hinge.object1.velocity.y)
    );
}

// ---------------------------------------------------------------------------
// World constraints (Joint / Rope / Spring / RopeChain inside PhysicsWorld)
// ---------------------------------------------------------------------------

/// A bob hanging at rest under a world joint must stay at rest. The joint
/// corrects positions but never velocity, so gravity accumulates velocity
/// without bound and the stretch grows linearly with time.
#[test]
fn review_world_joint_hanging_bob_stays_bounded() {
    let mut world = quiet_world(120.0);
    let bob = world.add_object(ball(1.0, (0.0, 4.0, 0.0)));
    world.add_constraint(WorldConstraint::Joint(WorldJoint3D::anchored((0.0, 5.0, 0.0), bob, 1.0)));
    for _ in 0..2000 {
        world.step();
    }
    let o = &world.get_object(bob).unwrap().object;
    let dist = (o.position.x.powi(2) + (o.position.y - 5.0).powi(2) + o.position.z.powi(2)).sqrt();
    let speed = (o.velocity.x.powi(2) + o.velocity.y.powi(2) + o.velocity.z.powi(2)).sqrt();
    assert!(
        (dist - 1.0).abs() < 0.05 && speed < 0.5,
        "after 2000 steps: joint length {dist:.3} m (target 1.0), bob speed {speed:.2} m/s"
    );
}

/// Same failure for a world rope holding a hanging bob.
#[test]
fn review_world_rope_hanging_bob_stays_bounded() {
    let mut world = quiet_world(120.0);
    let bob = world.add_object(ball(1.0, (0.0, 4.0, 0.0)));
    world.add_constraint(WorldConstraint::Rope(WorldRope3D::anchored((0.0, 5.0, 0.0), bob, 1.0)));
    for _ in 0..2000 {
        world.step();
    }
    let o = &world.get_object(bob).unwrap().object;
    let dist = (o.position.x.powi(2) + (o.position.y - 5.0).powi(2) + o.position.z.powi(2)).sqrt();
    let speed = (o.velocity.x.powi(2) + o.velocity.y.powi(2) + o.velocity.z.powi(2)).sqrt();
    assert!(
        dist < 1.05 && speed < 0.5,
        "after 2000 steps: rope length {dist:.3} m (max 1.0), bob speed {speed:.2} m/s"
    );
}

/// A world spring (k = 100 N/m, m = 1 kg) must oscillate at omega = sqrt(k/m).
/// Its impulse is applied once per solver iteration, i.e. 8x per step.
#[test]
fn review_world_spring_period_matches_analytic() {
    let mut world = PhysicsWorld::new(
        WorldConfig::default()
            .with_frequency(1000.0)
            .with_gravity(0.0, 0.0, 0.0)
            .without_aerodynamic_drag()
            .with_real_time(false),
    );
    let bob = world.add_object(ball(1.0, (1.1, 0.0, 0.0)));
    world.add_constraint(WorldConstraint::Spring(WorldSpring3D::anchored(
        (0.0, 0.0, 0.0),
        bob,
        100.0,
        1.0,
        0.0,
    )));
    let dt = 1.0 / 1000.0;
    let mut samples = Vec::new();
    for i in 0..3000 {
        world.step();
        let x = world.get_object(bob).unwrap().object.position.x - 1.0;
        samples.push(((i + 1) as f64 * dt, x));
    }
    let expected = 2.0 * std::f64::consts::PI / 10.0;
    let measured = period_from(&samples);
    assert!(
        (measured - expected).abs() / expected < 0.02,
        "world spring period {measured:.4} s vs analytic {expected:.4} s (ratio {:.3})",
        expected / measured
    );
}

/// Zero gravity, rope chain coasting at 1 m/s: one world step must move it
/// v * dt. `solve_rope_chain` calls `chain.integrate(dt)` on every world
/// solver iteration (8 by default).
#[test]
fn review_world_rope_chain_moves_v_dt_per_step() {
    let chain =
        RopeChain3D::from_points(&[(0.0, 0.0, 0.0), (0.5, 0.0, 0.0)], 1.0, false).unwrap();
    let mut world = PhysicsWorld::new(
        WorldConfig::default()
            .with_frequency(240.0)
            .with_gravity(0.0, 0.0, 0.0)
            .without_aerodynamic_drag()
            .with_real_time(false),
    );
    let id = world.add_constraint(WorldConstraint::RopeChain(chain));
    if let Some(WorldConstraint::RopeChain(c)) = world.get_constraint_mut(id) {
        for p in c.particles.iter_mut() {
            p.velocity.z = 1.0;
        }
    }
    world.step();
    let z = world.get_constraint(id).unwrap().get_particle_positions().unwrap()[0].2;
    let expected = 1.0 / 240.0;
    assert!(
        (z - expected).abs() / expected < 0.03,
        "coasting at 1 m/s for one 1/240 s step moved {z:.5} m, expected {expected:.5} m \
         ({:.2} x)",
        z / expected
    );
}

/// An unanchored rope chain is in free fall and must drop 1/2 g t^2 in vacuum.
/// On top of the 8x integration above, `solve_rope_chain` applies
/// `apply_damping(0.02)` (2% per call, not per second) on every iteration.
#[test]
fn review_world_rope_chain_free_fall_distance() {
    let chain =
        RopeChain3D::from_points(&[(0.0, 10.0, 0.0), (0.5, 10.0, 0.0)], 1.0, false).unwrap();
    let mut world = quiet_world(240.0);
    let id = world.add_constraint(WorldConstraint::RopeChain(chain));
    for _ in 0..240 {
        world.step();
    }
    let y = world.get_constraint(id).unwrap().get_particle_positions().unwrap()[0].1;
    let fell = 10.0 - y;
    let expected = 0.5 * G;
    assert!(
        (fell - expected).abs() / expected < 0.05,
        "free-falling rope dropped {fell:.3} m in 1 s, expected {expected:.3} m"
    );
}

// ---------------------------------------------------------------------------
// RopeChain3D
// ---------------------------------------------------------------------------

/// One PBD distance projection with a static end must land exactly on the rest
/// length whatever the particle mass. `solve_segment` divides the error by the
/// total inverse mass and then multiplies by the mass *fraction*, so the step
/// is scaled by the particle mass.
#[test]
fn review_rope_chain_projection_is_mass_independent() {
    let mut report = Vec::new();
    let mut ok = true;
    for &mass in &[0.1, 1.0, 10.0] {
        let mut chain =
            RopeChain3D::from_points(&[(0.0, 0.0, 0.0), (1.1, 0.0, 0.0)], mass, true).unwrap();
        chain.segment_lengths[0] = 1.0; // stretched by 0.1 m
        chain.solve(1.0 / 60.0, 1).unwrap();
        let len = chain.current_length();
        ok &= (len - 1.0).abs() < 1e-9;
        report.push(format!("{mass} kg -> {len:.4} m"));
    }
    assert!(ok, "segment length after one projection (expected 1.0 m): {}", report.join(", "));
}

/// `alpha = compliance / dt^2` is 0/0 = NaN at dt = 0 with the default
/// compliance of 0, and the NaN is written into the particle positions.
#[test]
fn review_rope_chain_dt_zero_does_not_nan() {
    let mut chain =
        RopeChain3D::from_points(&[(0.0, 0.0, 0.0), (1.1, 0.0, 0.0)], 1.0, true).unwrap();
    chain.segment_lengths[0] = 1.0;
    chain.solve(0.0, 1).unwrap();
    let p = chain.get_particle_positions()[1];
    assert!(p.0.is_finite() && p.1.is_finite() && p.2.is_finite(), "dt = 0 produced {p:?}");
}

/// A horizontal 10 x 0.1 m chain of 5 kg particles released under gravity with
/// `RopeChain3D::step` must stay finite and near its rest length.
#[test]
fn review_rope_chain_heavy_particles_stay_bounded() {
    let mut chain = RopeChain3D::new((0.0, 0.0, 0.0), 10, 0.1, 5.0).unwrap();
    for (i, p) in chain.particles.iter_mut().enumerate() {
        p.position.x = i as f64 * 0.1;
        p.position.y = 0.0;
    }
    let mut worst_len: f64 = 0.0;
    let mut first_bad = None;
    for step in 0..2000 {
        chain.step(1.0 / 60.0, -G, 10, None).unwrap();
        let len = chain.current_length();
        worst_len = worst_len.max(len);
        if first_bad.is_none() && !(len < 1.2) {
            first_bad = Some(step);
        }
    }
    assert!(
        first_bad.is_none(),
        "rest length 1.0 m; total length exceeded 1.2 m at step {first_bad:?}, worst {worst_len:e} m"
    );
}

// ---------------------------------------------------------------------------
// Rope3D
// ---------------------------------------------------------------------------

/// A stretched rope whose ends are already approaching carries no tension:
/// the solve may move positions but must not add kinetic energy.
#[test]
fn review_rope3d_does_not_accelerate_approaching_bodies() {
    let anchor = obj3(f64::INFINITY, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0));
    let bob = obj3(1.0, (1.1, 0.0, 0.0), (-1.0, 0.0, 0.0)); // moving inward at 1 m/s
    let mut rope = Rope3D::new(anchor, bob, 1.0).unwrap();
    rope.solve(0.01).unwrap();
    let v = rope.object2.velocity.x;
    assert!(
        (v + 1.0).abs() < 1e-9,
        "inward speed must stay 1 m/s (KE 0.5 J); got {:.3} m/s (KE {:.3} J)",
        -v,
        0.5 * v * v
    );
}

/// A rope is an internal force: it must not change the pair's total momentum.
#[test]
fn review_rope3d_conserves_momentum_light_object1() {
    let light = obj3(1.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0));
    let heavy = obj3(10.0, (1.1, 0.0, 0.0), (0.0, 0.0, 0.0)); // stretched 0.1 m, at rest
    let mut rope = Rope3D::new(light, heavy, 1.0).unwrap();
    rope.solve(0.01).unwrap();
    let p = rope.object1.mass * rope.object1.velocity.x + rope.object2.mass * rope.object2.velocity.x;
    assert!(p.abs() < 1e-9, "total momentum went 0 -> {p:.3} kg m/s from one rope solve");
}

// ---------------------------------------------------------------------------
// Joint3D / Fixed3D: impulse clamp in the wrong units; dt = 0
// ---------------------------------------------------------------------------

/// Hanging bob under a Joint3D, stepped in the world's order (gravity, 10
/// solver iterations, integrate). The clamp `0.1 / dt` has units of 1/s, not
/// N s, so a bob can receive at most 0.1/(m dt) m/s per iteration.
#[test]
fn review_joint3d_holds_heavy_bob() {
    for &mass in &[1.0, 1000.0] {
        let anchor = obj3(f64::INFINITY, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0));
        let bob = obj3(mass, (0.0, -1.0, 0.0), (0.0, 0.0, 0.0));
        let mut joint = Joint3D::new(anchor, bob, 1.0).unwrap();
        let dt = 1.0 / 60.0;
        for _ in 0..2000 {
            joint.object2.velocity.y -= G * dt;
            for _ in 0..10 {
                joint.solve(dt).unwrap();
            }
            let v = joint.object2.velocity.clone();
            joint.object2.position.x += v.x * dt;
            joint.object2.position.y += v.y * dt;
        }
        let err = joint.calculate_error();
        let vy = joint.object2.velocity.y;
        assert!(
            err < 0.05 && vy.abs() < 0.5,
            "mass {mass} kg: joint error {err:.3} m, bob vy {vy:.2} m/s after 2000 steps"
        );
    }
}

/// Same clamp in Fixed3D.
#[test]
fn review_fixed3d_holds_heavy_body() {
    for &mass in &[1.0, 1000.0] {
        let anchor = obj3(f64::INFINITY, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0));
        let body = obj3(mass, (0.0, -1.0, 0.0), (0.0, 0.0, 0.0));
        let mut weld = Fixed3D::new(anchor, body).unwrap();
        let dt = 1.0 / 60.0;
        for _ in 0..2000 {
            weld.object2.velocity.y -= G * dt;
            for _ in 0..10 {
                weld.solve(dt).unwrap();
            }
            let v = weld.object2.velocity.clone();
            weld.object2.position.y += v.y * dt;
        }
        let err = weld.calculate_error();
        let vy = weld.object2.velocity.y;
        assert!(
            err < 0.05 && vy.abs() < 0.5,
            "mass {mass} kg: weld error {err:.3} m, body vy {vy:.2} m/s after 2000 steps"
        );
    }
}

/// dt = 0 must not poison a static anchor with NaN.
#[test]
fn review_joint3d_dt_zero_does_not_nan() {
    let anchor = obj3(f64::INFINITY, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0));
    let bob = obj3(1.0, (1.1, 0.0, 0.0), (0.0, 0.0, 0.0));
    let mut joint = Joint3D::new(anchor, bob, 1.0).unwrap();
    let _ = joint.solve(0.0);
    let v1 = joint.object1.velocity.x;
    let v2 = joint.object2.velocity.x;
    assert!(v1.is_finite() && v2.is_finite(), "dt = 0: anchor v = {v1}, bob v = {v2}");
}

// ---------------------------------------------------------------------------
// Contact3D
// ---------------------------------------------------------------------------

/// Head-on elastic impact with friction and zero tangential velocity: friction
/// must not touch the normal velocity, so e = 1 leaves at 1 m/s.
/// (Normal given the way the code actually reads it: object1 -> object2.)
#[test]
fn review_contact3d_friction_does_not_act_along_normal() {
    let floor = obj3(f64::INFINITY, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0));
    let ball = obj3(1.0, (0.0, 0.5, 0.0), (0.0, -1.0, 0.0));
    let cp = ContactPoint3D { position: (0.0, 0.0, 0.0), normal: (0.0, 1.0, 0.0), penetration: 1e-6 };
    let mut c = Contact3D::new(floor, ball, cp, 1.0, 0.5).unwrap().with_baumgarte(0.0);
    c.solve(1.0 / 60.0).unwrap();
    let vy = c.object2.velocity.y;
    assert!((vy - 1.0).abs() < 1e-9, "elastic bounce with mu = 0.5: vy = {vy}, expected 1.0");
}

/// Following the documented convention ("normal points from object1 toward
/// object2"), a ball (object1) falling onto a static floor (object2) must
/// bounce. The docs used to say the opposite, and following them produced no
/// response at all.
#[test]
fn review_contact3d_documented_normal_convention_bounces() {
    let ball = obj3(1.0, (0.0, 0.5, 0.0), (0.0, -1.0, 0.0));
    let floor = obj3(f64::INFINITY, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0));
    let cp = ContactPoint3D {
        position: (0.0, 0.0, 0.0),
        normal: (0.0, -1.0, 0.0), // from object1 (ball) toward object2 (floor), as documented
        penetration: 0.05,
    };
    let mut c = Contact3D::new(ball, floor, cp, 0.5, 0.0).unwrap();
    c.solve(1.0 / 60.0).unwrap();
    let vy = c.object1.velocity.y;
    assert!(vy >= 0.0, "ball approaching the floor must not keep falling: vy = {vy}");
}

/// Inelastic (e = 0) resting contact at 10 cm penetration. The solve both
/// projects the position out *and* adds the Baumgarte bias velocity, so the
/// body leaves the surface. With e = 0 the free-flight apex must not rise
/// above the contact plane.
#[test]
fn review_contact3d_inelastic_contact_does_not_launch_body() {
    let dt = 1.0 / 60.0;
    let pen = 0.1;
    let floor = obj3(f64::INFINITY, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0));
    let ball = obj3(1.0, (0.0, 0.5 - pen, 0.0), (0.0, -G * dt, 0.0));
    let cp = ContactPoint3D { position: (0.0, 0.0, 0.0), normal: (0.0, 1.0, 0.0), penetration: pen };
    let mut c = Contact3D::new(floor, ball, cp, 0.0, 0.0).unwrap();
    c.solve(dt).unwrap();
    let vy = c.object2.velocity.y;
    let separation = -pen + (c.object2.position.y - (0.5 - pen));
    let apex = separation + vy.max(0.0).powi(2) / (2.0 * G);
    assert!(
        apex <= 1e-9,
        "e = 0: after the solve the ball is at {separation:.3} m with vy = {vy:.3} m/s, \
         so it flies to {apex:.3} m above the surface"
    );
}

/// The stored penetration is never reduced after the position correction, so
/// every iteration that does not end separating re-applies the correction.
/// With e = 0 and no bias, ten iterations push the body out ten times.
#[test]
fn review_contact3d_position_correction_not_reapplied_per_iteration() {
    let pen = 0.05;
    let floor = obj3(f64::INFINITY, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0));
    let ball = obj3(1.0, (0.0, 0.5 - pen, 0.0), (0.0, 0.0, 0.0));
    let cp = ContactPoint3D { position: (0.0, 0.0, 0.0), normal: (0.0, 1.0, 0.0), penetration: pen };
    let mut c = Contact3D::new(floor, ball, cp, 0.0, 0.0).unwrap().with_baumgarte(0.0);
    for _ in 0..10 {
        Constraint3D::solve(&mut c, 1.0 / 60.0).unwrap();
    }
    let moved = c.object2.position.y - (0.5 - pen);
    assert!(moved <= pen + 1e-9, "penetration {pen} m, but the body was pushed out {moved:.3} m");
}

// ---------------------------------------------------------------------------
// Springs
// ---------------------------------------------------------------------------

/// 1D Spring: the system is mirror-symmetric, so a bob on the -x side must
/// lose energy exactly as fast as one on the +x side. `(k x + c v_rel) *
/// signum(dx)` flips the sign of the damping term when dx < 0.
#[test]
fn review_spring_1d_damping_is_mirror_symmetric() {
    let run = |side: f64| -> (f64, f64) {
        let anchor = Object::new(f64::INFINITY, 0.0, 0.0).unwrap();
        let bob = Object::new(1.0, 0.0, side * 1.5).unwrap();
        let mut s = Spring::new(anchor, bob, 10.0, 1.0, 1.0).unwrap();
        let energy = |s: &Spring| {
            let x = (s.object2.position - s.object1.position).abs() - s.rest_length;
            0.5 * s.object2.velocity.powi(2) + 0.5 * s.spring_constant * x * x
        };
        let e0 = energy(&s);
        for _ in 0..2000 {
            ConstraintSolver::solve(&mut s, 1e-3).unwrap();
        }
        (e0, energy(&s))
    };
    let (e0, e_pos) = run(1.0);
    let (_, e_neg) = run(-1.0);
    assert!(
        (e_pos - e_neg).abs() < 1e-6 && e_neg < e0,
        "E0 = {e0:.3} J; after 2 s with c = 1: {e_pos:.4} J on the +x side, {e_neg:.4} J on the -x side"
    );
}

/// Critical damping against a static anchor is 2 sqrt(k m).
#[test]
fn review_spring3d_critical_damping_with_static_anchor() {
    let anchor = obj3(f64::INFINITY, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0));
    let bob = obj3(2.0, (1.0, 0.0, 0.0), (0.0, 0.0, 0.0));
    let s = Spring3D::new(anchor, bob, 50.0, 1.0, 0.0).unwrap();
    let c = s.critical_damping();
    let expected = 2.0 * (50.0_f64 * 2.0).sqrt();
    assert!((c - expected).abs() < 1e-9, "critical damping {c}, expected {expected}");
}

// ---------------------------------------------------------------------------
// UnifiedSolver3D warm starting
// ---------------------------------------------------------------------------

/// Warm starting must change what the solver does (it should pre-apply last
/// frame's impulse). `set_lambda` only stores a number that no constraint
/// reads, so turning it on changes nothing.
#[test]
#[ignore = "known defect, not fixed in this change: see docs/reviews/2026-09-29-correctness-performance.md"]
fn review_warm_starting_has_an_effect() {
    let run = |warm: bool| -> (f64, f64) {
        let anchor = obj3(f64::INFINITY, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0));
        let bob = obj3(1.0, (1.0, 0.0, 0.0), (0.0, 0.0, 0.0));
        let j = Rc::new(RefCell::new(Joint3D::new(anchor, bob, 1.0).unwrap()));
        let mut solver = UnifiedSolver3D::new(4, 1e-9).unwrap().with_warm_starting(warm);
        solver.add_constraint(Box::new(Shared(j.clone())));
        let dt = 1.0 / 60.0;
        for _ in 0..120 {
            j.borrow_mut().object2.velocity.y -= G * dt;
            solver.solve(dt).unwrap();
            let mut jb = j.borrow_mut();
            let v = jb.object2.velocity.clone();
            jb.object2.position.x += v.x * dt;
            jb.object2.position.y += v.y * dt;
        }
        let jb = j.borrow();
        (jb.object2.position.x, jb.object2.position.y)
    };
    let cold = run(false);
    let warm = run(true);
    assert!(cold != warm, "warm starting on and off produced bit-identical trajectories: {cold:?}");
}
