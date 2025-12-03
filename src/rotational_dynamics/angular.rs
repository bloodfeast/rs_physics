//! Angular velocity, momentum, and kinetic energy utilities
//!
//! This module provides shared functions for rotational physics calculations
//! that work with both 2D (scalar) and 3D (vector) angular quantities.

use super::inertia::{InertiaScalar, InertiaTensor};

// Re-export vector math from shared utils module
pub use crate::utils::vector3::{
    cross_product, dot_product, magnitude, normalize, scale, add, sub,
    point_velocity, Vec3,
};

//==============================================================================
// 2D ANGULAR DYNAMICS
//==============================================================================

/// Angular state for 2D rotation
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct AngularState2D {
    /// Angular velocity in radians per second (positive = counter-clockwise)
    pub velocity: f64,
    /// Angular position (orientation) in radians
    pub angle: f64,
}

impl AngularState2D {
    /// Create a new angular state
    pub fn new(velocity: f64, angle: f64) -> Self {
        Self { velocity, angle }
    }

    /// Calculate angular momentum: L = I * ω
    #[inline]
    pub fn momentum(&self, inertia: InertiaScalar) -> f64 {
        inertia.value() * self.velocity
    }

    /// Calculate rotational kinetic energy: E = (1/2) * I * ω²
    #[inline]
    pub fn kinetic_energy(&self, inertia: InertiaScalar) -> f64 {
        0.5 * inertia.value() * self.velocity * self.velocity
    }

    /// Apply a torque for a duration: Δω = τ * Δt / I
    #[inline]
    pub fn apply_torque(&mut self, torque: f64, dt: f64, inertia: InertiaScalar) {
        self.velocity += torque * dt * inertia.inverse();
    }

    /// Apply an angular impulse: Δω = J / I
    #[inline]
    pub fn apply_impulse(&mut self, impulse: f64, inertia: InertiaScalar) {
        self.velocity += impulse * inertia.inverse();
    }

    /// Integrate the angular position: θ += ω * Δt
    #[inline]
    pub fn integrate(&mut self, dt: f64) {
        self.angle += self.velocity * dt;
        // Normalize angle to [-π, π]
        self.angle = normalize_angle(self.angle);
    }

    /// Apply angular damping
    #[inline]
    pub fn apply_damping(&mut self, damping: f64, dt: f64) {
        self.velocity *= (1.0 - damping * dt).max(0.0);
    }
}

/// Normalize an angle to the range [-π, π]
#[inline]
pub fn normalize_angle(angle: f64) -> f64 {
    let pi = std::f64::consts::PI;
    let mut a = angle % (2.0 * pi);
    if a > pi {
        a -= 2.0 * pi;
    } else if a < -pi {
        a += 2.0 * pi;
    }
    a
}

//==============================================================================
// 3D ANGULAR DYNAMICS
//==============================================================================

/// Angular state for 3D rotation
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct AngularState3D {
    /// Angular velocity vector (ωx, ωy, ωz) in radians per second
    pub velocity: (f64, f64, f64),
}

impl AngularState3D {
    /// Create a new angular state
    pub fn new(velocity: (f64, f64, f64)) -> Self {
        Self { velocity }
    }

    /// Create from components
    pub fn from_components(wx: f64, wy: f64, wz: f64) -> Self {
        Self {
            velocity: (wx, wy, wz),
        }
    }

    /// Get the magnitude of angular velocity
    #[inline]
    pub fn speed(&self) -> f64 {
        let (wx, wy, wz) = self.velocity;
        (wx * wx + wy * wy + wz * wz).sqrt()
    }

    /// Calculate angular momentum: L = I * ω
    #[inline]
    pub fn momentum(&self, inertia: &InertiaTensor) -> (f64, f64, f64) {
        inertia.multiply_vector(self.velocity)
    }

    /// Calculate rotational kinetic energy: E = (1/2) * ω · L = (1/2) * ω · (I * ω)
    #[inline]
    pub fn kinetic_energy(&self, inertia: &InertiaTensor) -> f64 {
        let l = self.momentum(inertia);
        let (wx, wy, wz) = self.velocity;
        0.5 * (wx * l.0 + wy * l.1 + wz * l.2)
    }

    /// Apply a torque for a duration: Δω = I⁻¹ * τ * Δt
    #[inline]
    pub fn apply_torque(&mut self, torque: (f64, f64, f64), dt: f64, inertia: &InertiaTensor) {
        let alpha = inertia.apply_inverse_to_torque(torque);
        self.velocity.0 += alpha.0 * dt;
        self.velocity.1 += alpha.1 * dt;
        self.velocity.2 += alpha.2 * dt;
    }

    /// Apply an angular impulse: Δω = I⁻¹ * J
    #[inline]
    pub fn apply_impulse(&mut self, impulse: (f64, f64, f64), inertia: &InertiaTensor) {
        let delta = inertia.apply_inverse_to_torque(impulse);
        self.velocity.0 += delta.0;
        self.velocity.1 += delta.1;
        self.velocity.2 += delta.2;
    }

    /// Apply angular damping
    #[inline]
    pub fn apply_damping(&mut self, damping: f64, dt: f64) {
        let factor = (1.0 - damping * dt).max(0.0);
        self.velocity.0 *= factor;
        self.velocity.1 *= factor;
        self.velocity.2 *= factor;
    }

    /// Apply velocity threshold - set near-zero velocities to zero
    #[inline]
    pub fn apply_threshold(&mut self, threshold: f64) {
        if self.velocity.0.abs() < threshold {
            self.velocity.0 = 0.0;
        }
        if self.velocity.1.abs() < threshold {
            self.velocity.1 = 0.0;
        }
        if self.velocity.2.abs() < threshold {
            self.velocity.2 = 0.0;
        }
    }
}

//==============================================================================
// COLLISION ANGULAR IMPULSE UTILITIES
//==============================================================================

/// Calculate the angular impulse from a collision
/// Returns the change in angular velocity that should be applied
///
/// # Arguments
/// * `r` - Vector from center of mass to contact point
/// * `impulse` - Linear impulse vector at contact point
/// * `inertia` - Inertia tensor of the object
///
/// # Returns
/// Angular velocity change (Δω)
pub fn angular_impulse_from_collision(
    r: (f64, f64, f64),
    impulse: (f64, f64, f64),
    inertia: &InertiaTensor,
) -> (f64, f64, f64) {
    // τ = r × F (torque from impulse)
    let torque = cross_product(r, impulse);
    // Δω = I⁻¹ * τ
    inertia.apply_inverse_to_torque(torque)
}

/// Calculate the effective mass at a contact point for collision response
/// This accounts for both linear and rotational inertia
///
/// # Arguments
/// * `inv_mass` - Inverse mass of the object (0 for static)
/// * `r` - Vector from center of mass to contact point
/// * `normal` - Collision normal
/// * `inv_inertia` - Inverse inertia tensor
///
/// # Returns
/// Effective inverse mass at the contact point along the normal
pub fn effective_inverse_mass(
    inv_mass: f64,
    r: (f64, f64, f64),
    normal: (f64, f64, f64),
    inv_inertia: &InertiaTensor,
) -> f64 {
    // Linear contribution
    let linear = inv_mass;

    // Angular contribution: (I⁻¹ * (r × n)) · (r × n) / |n|²
    let r_cross_n = cross_product(r, normal);
    let i_inv_r_cross_n = inv_inertia.multiply_vector(r_cross_n);
    let angular = dot_product(i_inv_r_cross_n, r_cross_n);

    linear + angular
}

// NOTE: point_velocity, cross_product, dot_product, magnitude, normalize, scale, add, sub
// are now re-exported from crate::utils::vector3

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== AngularState2D Tests ====================

    #[test]
    fn test_angular_state_2d() {
        let mut state = AngularState2D::new(2.0, 0.0);
        let inertia = InertiaScalar::from(4.0);

        assert_eq!(state.momentum(inertia), 8.0);
        assert!((state.kinetic_energy(inertia) - 8.0).abs() < 1e-10);

        state.apply_torque(4.0, 1.0, inertia);
        assert!((state.velocity - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_angular_state_2d_default() {
        let state = AngularState2D::default();
        assert_eq!(state.velocity, 0.0);
        assert_eq!(state.angle, 0.0);
    }

    #[test]
    fn test_angular_state_2d_impulse() {
        let mut state = AngularState2D::new(0.0, 0.0);
        let inertia = InertiaScalar::from(2.0);

        state.apply_impulse(4.0, inertia);
        // Δω = J / I = 4 / 2 = 2
        assert!((state.velocity - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_angular_state_2d_integrate() {
        let mut state = AngularState2D::new(1.0, 0.0);

        state.integrate(1.0);
        assert!((state.angle - 1.0).abs() < 1e-10);

        state.integrate(1.0);
        assert!((state.angle - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_angular_state_2d_integrate_wrap() {
        let mut state = AngularState2D::new(std::f64::consts::PI, 0.0);

        // After 2 seconds, angle should be 2π, which normalizes to ~0
        state.integrate(2.0);
        assert!(state.angle.abs() < 1e-10);
    }

    #[test]
    fn test_angular_state_2d_damping() {
        let mut state = AngularState2D::new(10.0, 0.0);

        state.apply_damping(0.5, 1.0);
        // velocity *= (1 - 0.5 * 1) = 0.5
        assert!((state.velocity - 5.0).abs() < 1e-10);

        // Full damping should clamp to 0
        state.apply_damping(2.0, 1.0);
        assert_eq!(state.velocity, 0.0);
    }

    // ==================== AngularState3D Tests ====================

    #[test]
    fn test_angular_state_3d() {
        let mut state = AngularState3D::from_components(1.0, 0.0, 0.0);
        let inertia = InertiaTensor::uniform(2.0);

        let momentum = state.momentum(&inertia);
        assert!((momentum.0 - 2.0).abs() < 1e-10);

        state.apply_torque((2.0, 0.0, 0.0), 1.0, &inertia);
        assert!((state.velocity.0 - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_angular_state_3d_default() {
        let state = AngularState3D::default();
        assert_eq!(state.velocity, (0.0, 0.0, 0.0));
    }

    #[test]
    fn test_angular_state_3d_speed() {
        let state = AngularState3D::from_components(3.0, 4.0, 0.0);
        assert!((state.speed() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_angular_state_3d_kinetic_energy() {
        let state = AngularState3D::from_components(2.0, 0.0, 0.0);
        let inertia = InertiaTensor::uniform(3.0);

        // E = 0.5 * ω · L = 0.5 * ω · (I * ω)
        // L = (6, 0, 0), ω = (2, 0, 0)
        // E = 0.5 * 12 = 6
        assert!((state.kinetic_energy(&inertia) - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_angular_state_3d_impulse() {
        let mut state = AngularState3D::from_components(0.0, 0.0, 0.0);
        let inertia = InertiaTensor::diagonal_only(2.0, 4.0, 8.0);

        state.apply_impulse((2.0, 4.0, 8.0), &inertia);
        // Δω = I^-1 * J = (2/2, 4/4, 8/8) = (1, 1, 1)
        assert!((state.velocity.0 - 1.0).abs() < 1e-10);
        assert!((state.velocity.1 - 1.0).abs() < 1e-10);
        assert!((state.velocity.2 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_angular_state_3d_damping() {
        let mut state = AngularState3D::from_components(10.0, 20.0, 30.0);

        state.apply_damping(0.5, 1.0);
        assert!((state.velocity.0 - 5.0).abs() < 1e-10);
        assert!((state.velocity.1 - 10.0).abs() < 1e-10);
        assert!((state.velocity.2 - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_angular_state_3d_threshold() {
        let mut state = AngularState3D::from_components(0.001, 0.5, 0.0001);

        state.apply_threshold(0.01);
        assert_eq!(state.velocity.0, 0.0); // Below threshold
        assert!((state.velocity.1 - 0.5).abs() < 1e-10); // Above threshold
        assert_eq!(state.velocity.2, 0.0); // Below threshold
    }

    // ==================== Point Velocity Tests ====================

    #[test]
    fn test_point_velocity() {
        let linear = (1.0, 0.0, 0.0);
        let angular = (0.0, 0.0, 1.0); // Rotating around Z
        let r = (0.0, 1.0, 0.0); // Point 1 unit in Y

        let vel = point_velocity(linear, angular, r);
        // Angular contribution: ω × r = (0,0,1) × (0,1,0) = (-1, 0, 0)
        // Total: (1,0,0) + (-1,0,0) = (0,0,0)
        assert!((vel.0).abs() < 1e-10);
        assert!((vel.1).abs() < 1e-10);
        assert!((vel.2).abs() < 1e-10);
    }

    #[test]
    fn test_point_velocity_pure_rotation() {
        let linear = (0.0, 0.0, 0.0);
        let angular = (0.0, 0.0, 1.0); // 1 rad/s around Z
        let r = (1.0, 0.0, 0.0); // Point 1 unit in X

        let vel = point_velocity(linear, angular, r);
        // ω × r = (0,0,1) × (1,0,0) = (0, 1, 0)
        assert!((vel.0).abs() < 1e-10);
        assert!((vel.1 - 1.0).abs() < 1e-10);
        assert!((vel.2).abs() < 1e-10);
    }

    // ==================== Collision Utilities Tests ====================

    #[test]
    fn test_angular_impulse_from_collision() {
        let r = (1.0, 0.0, 0.0); // Contact 1 unit in X
        let impulse = (0.0, 1.0, 0.0); // Impulse in Y direction
        let inertia = InertiaTensor::uniform(2.0);

        let delta_omega = angular_impulse_from_collision(r, impulse, &inertia);
        // τ = r × F = (1,0,0) × (0,1,0) = (0, 0, 1)
        // Δω = I^-1 * τ = (0, 0, 1) / 2 = (0, 0, 0.5)
        assert!((delta_omega.0).abs() < 1e-10);
        assert!((delta_omega.1).abs() < 1e-10);
        assert!((delta_omega.2 - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_effective_inverse_mass() {
        let inv_mass = 1.0;
        let r = (1.0, 0.0, 0.0);
        let normal = (0.0, 1.0, 0.0);
        let inv_inertia = InertiaTensor::uniform(1.0);

        let eff_inv_mass = effective_inverse_mass(inv_mass, r, normal, &inv_inertia);

        // Linear contribution: 1.0
        // Angular contribution: (I^-1 * (r × n)) · (r × n)
        // r × n = (1,0,0) × (0,1,0) = (0,0,1)
        // I^-1 * (0,0,1) = (0,0,1)
        // (0,0,1) · (0,0,1) = 1
        // Total: 1 + 1 = 2
        assert!((eff_inv_mass - 2.0).abs() < 1e-10);
    }

    // ==================== Vector Math Tests ====================

    #[test]
    fn test_cross_product() {
        let a = (1.0, 0.0, 0.0);
        let b = (0.0, 1.0, 0.0);
        let c = cross_product(a, b);
        assert!((c.0).abs() < 1e-10);
        assert!((c.1).abs() < 1e-10);
        assert!((c.2 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_dot_product() {
        let a = (1.0, 2.0, 3.0);
        let b = (4.0, 5.0, 6.0);
        let d = dot_product(a, b);
        // 1*4 + 2*5 + 3*6 = 32
        assert!((d - 32.0).abs() < 1e-10);
    }

    #[test]
    fn test_magnitude() {
        let v = (3.0, 4.0, 0.0);
        assert!((magnitude(v) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalize() {
        let v = (3.0, 4.0, 0.0);
        let n = normalize(v);
        assert!((n.0 - 0.6).abs() < 1e-10);
        assert!((n.1 - 0.8).abs() < 1e-10);
        assert!((n.2).abs() < 1e-10);

        // Zero vector should return zero
        let zero = normalize((0.0, 0.0, 0.0));
        assert_eq!(zero, (0.0, 0.0, 0.0));
    }

    #[test]
    fn test_scale() {
        let v = (1.0, 2.0, 3.0);
        let s = scale(v, 2.0);
        assert_eq!(s, (2.0, 4.0, 6.0));
    }

    #[test]
    fn test_add_sub() {
        let a = (1.0, 2.0, 3.0);
        let b = (4.0, 5.0, 6.0);

        let sum = add(a, b);
        assert_eq!(sum, (5.0, 7.0, 9.0));

        let diff = sub(b, a);
        assert_eq!(diff, (3.0, 3.0, 3.0));
    }

    #[test]
    fn test_normalize_angle() {
        assert!((normalize_angle(0.0)).abs() < 1e-10);
        assert!((normalize_angle(std::f64::consts::PI) - std::f64::consts::PI).abs() < 1e-10);
        assert!((normalize_angle(3.0 * std::f64::consts::PI) - std::f64::consts::PI).abs() < 1e-10);
        assert!((normalize_angle(-3.0 * std::f64::consts::PI) + std::f64::consts::PI).abs() < 1e-10);
    }
}
