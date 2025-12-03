//! Shared 3D vector math utilities
//!
//! This module provides fundamental 3D vector operations used throughout the physics engine.
//! All other modules should use these functions instead of defining their own.

/// A 3D vector represented as a tuple (x, y, z)
pub type Vec3 = (f64, f64, f64);

/// Cross product of two 3D vectors
///
/// # Arguments
/// * `a` - The first vector
/// * `b` - The second vector
///
/// # Returns
/// The cross product vector `a × b`
///
/// # Example
/// ```
/// use rs_physics::utils::vector3::cross_product;
///
/// let a = (1.0, 0.0, 0.0);
/// let b = (0.0, 1.0, 0.0);
/// let result = cross_product(a, b);
/// assert_eq!(result, (0.0, 0.0, 1.0));
/// ```
#[inline]
pub fn cross_product(a: Vec3, b: Vec3) -> Vec3 {
    (
        a.1 * b.2 - a.2 * b.1,
        a.2 * b.0 - a.0 * b.2,
        a.0 * b.1 - a.1 * b.0,
    )
}

/// Dot product of two 3D vectors
///
/// # Arguments
/// * `a` - The first vector
/// * `b` - The second vector
///
/// # Returns
/// The dot product scalar `a · b`
///
/// # Example
/// ```
/// use rs_physics::utils::vector3::dot_product;
///
/// let a = (1.0, 2.0, 3.0);
/// let b = (4.0, 5.0, 6.0);
/// let result = dot_product(a, b);
/// assert_eq!(result, 32.0); // 1*4 + 2*5 + 3*6 = 32
/// ```
#[inline]
pub fn dot_product(a: Vec3, b: Vec3) -> f64 {
    a.0 * b.0 + a.1 * b.1 + a.2 * b.2
}

/// Calculate the magnitude (length) of a 3D vector
///
/// # Arguments
/// * `v` - The vector
///
/// # Returns
/// The magnitude `|v|`
///
/// # Example
/// ```
/// use rs_physics::utils::vector3::magnitude;
///
/// let v = (3.0, 4.0, 0.0);
/// assert!((magnitude(v) - 5.0).abs() < 1e-10);
/// ```
#[inline]
pub fn magnitude(v: Vec3) -> f64 {
    (v.0 * v.0 + v.1 * v.1 + v.2 * v.2).sqrt()
}

/// Calculate the squared magnitude of a 3D vector (faster, avoids sqrt)
///
/// # Arguments
/// * `v` - The vector
///
/// # Returns
/// The squared magnitude `|v|²`
#[inline]
pub fn magnitude_squared(v: Vec3) -> f64 {
    v.0 * v.0 + v.1 * v.1 + v.2 * v.2
}

/// Normalize a 3D vector to unit length
///
/// # Arguments
/// * `v` - The vector to normalize
///
/// # Returns
/// The unit vector in the same direction, or (0,0,0) if the input is too small
///
/// # Example
/// ```
/// use rs_physics::utils::vector3::normalize;
///
/// let v = (3.0, 4.0, 0.0);
/// let n = normalize(v);
/// assert!((n.0 - 0.6).abs() < 1e-10);
/// assert!((n.1 - 0.8).abs() < 1e-10);
/// ```
#[inline]
pub fn normalize(v: Vec3) -> Vec3 {
    let mag = magnitude(v);
    if mag > 1e-10 {
        (v.0 / mag, v.1 / mag, v.2 / mag)
    } else {
        (0.0, 0.0, 0.0)
    }
}

/// Scale a vector by a scalar
///
/// # Arguments
/// * `v` - The vector
/// * `s` - The scalar multiplier
///
/// # Returns
/// The scaled vector `s * v`
#[inline]
pub fn scale(v: Vec3, s: f64) -> Vec3 {
    (v.0 * s, v.1 * s, v.2 * s)
}

/// Add two vectors
///
/// # Arguments
/// * `a` - The first vector
/// * `b` - The second vector
///
/// # Returns
/// The sum `a + b`
#[inline]
pub fn add(a: Vec3, b: Vec3) -> Vec3 {
    (a.0 + b.0, a.1 + b.1, a.2 + b.2)
}

/// Subtract two vectors
///
/// # Arguments
/// * `a` - The first vector
/// * `b` - The second vector
///
/// # Returns
/// The difference `a - b`
#[inline]
pub fn sub(a: Vec3, b: Vec3) -> Vec3 {
    (a.0 - b.0, a.1 - b.1, a.2 - b.2)
}

/// Negate a vector
///
/// # Arguments
/// * `v` - The vector
///
/// # Returns
/// The negated vector `-v`
#[inline]
pub fn negate(v: Vec3) -> Vec3 {
    (-v.0, -v.1, -v.2)
}

/// Calculate the velocity at a point on a rigid body
///
/// The velocity at a point is the sum of the linear velocity and the
/// rotational contribution: `v_point = v_linear + ω × r`
///
/// # Arguments
/// * `linear_vel` - Linear velocity of the center of mass
/// * `angular_vel` - Angular velocity vector
/// * `r` - Position vector from center of mass to the point
///
/// # Returns
/// Total velocity at the point
#[inline]
pub fn point_velocity(linear_vel: Vec3, angular_vel: Vec3, r: Vec3) -> Vec3 {
    let angular_contribution = cross_product(angular_vel, r);
    add(linear_vel, angular_contribution)
}

//==============================================================================
// COLLISION ANGULAR DYNAMICS UTILITIES
//==============================================================================
// These functions work with diagonal inertia tensors stored as [Ixx, Iyy, Izz, ...]
// arrays as returned by Shape3D::moment_of_inertia()

/// Calculate the angular contribution to effective inverse mass at a contact point
///
/// For diagonal inertia tensors, this computes: (r×n)²ₓ/Ixx + (r×n)²ᵧ/Iyy + (r×n)²ᵤ/Izz
///
/// # Arguments
/// * `r` - Vector from center of mass to contact point
/// * `normal` - Collision normal
/// * `inertia` - Diagonal inertia tensor as [Ixx, Iyy, Izz, ...]
///
/// # Returns
/// The angular contribution to effective inverse mass (0 if inertia is infinite/zero)
#[inline]
pub fn angular_effective_inv_mass(r: Vec3, normal: Vec3, inertia: &[f64; 6]) -> f64 {
    let r_cross_n = cross_product(r, normal);

    // For diagonal tensors: sum of (r×n)²ᵢ / Iᵢ
    r_cross_n.0 * r_cross_n.0 / inertia[0]
        + r_cross_n.1 * r_cross_n.1 / inertia[1]
        + r_cross_n.2 * r_cross_n.2 / inertia[2]
}

/// Calculate the change in angular velocity from a collision impulse
///
/// Computes Δω = I⁻¹ * (r × J) for diagonal inertia tensors
///
/// # Arguments
/// * `r` - Vector from center of mass to contact point
/// * `impulse` - Linear impulse vector at contact point
/// * `inertia` - Diagonal inertia tensor as [Ixx, Iyy, Izz, ...]
/// * `scale` - Optional scaling factor (e.g., for energy dissipation)
///
/// # Returns
/// Change in angular velocity (Δωx, Δωy, Δωz)
#[inline]
pub fn angular_velocity_delta(r: Vec3, impulse: Vec3, inertia: &[f64; 6], scale: f64) -> Vec3 {
    let angular_impulse = cross_product(r, impulse);
    (
        angular_impulse.0 / inertia[0] * scale,
        angular_impulse.1 / inertia[1] * scale,
        angular_impulse.2 / inertia[2] * scale,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_angular_effective_inv_mass() {
        // Uniform inertia (like a sphere)
        let inertia = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0];
        let r = (1.0, 0.0, 0.0);
        let normal = (0.0, 1.0, 0.0);

        // r × n = (1,0,0) × (0,1,0) = (0,0,1)
        // angular_inv_mass = 0²/1 + 0²/1 + 1²/1 = 1.0
        let result = angular_effective_inv_mass(r, normal, &inertia);
        assert!((result - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_angular_effective_inv_mass_non_uniform() {
        // Non-uniform inertia (like a box)
        let inertia = [2.0, 4.0, 8.0, 0.0, 0.0, 0.0];
        let r = (1.0, 1.0, 1.0);
        let normal = (1.0, 0.0, 0.0);

        // r × n = (1,1,1) × (1,0,0) = (0,1,-1)
        // angular_inv_mass = 0²/2 + 1²/4 + (-1)²/8 = 0 + 0.25 + 0.125 = 0.375
        let result = angular_effective_inv_mass(r, normal, &inertia);
        assert!((result - 0.375).abs() < 1e-10);
    }

    #[test]
    fn test_angular_velocity_delta() {
        let inertia = [1.0, 2.0, 4.0, 0.0, 0.0, 0.0];
        let r = (1.0, 0.0, 0.0);
        let impulse = (0.0, 1.0, 0.0);

        // r × impulse = (1,0,0) × (0,1,0) = (0,0,1)
        // Δω = (0/1, 0/2, 1/4) * scale = (0, 0, 0.25) for scale=1.0
        let delta = angular_velocity_delta(r, impulse, &inertia, 1.0);
        assert!((delta.0).abs() < 1e-10);
        assert!((delta.1).abs() < 1e-10);
        assert!((delta.2 - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_angular_velocity_delta_with_scale() {
        let inertia = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0];
        let r = (0.0, 1.0, 0.0);
        let impulse = (1.0, 0.0, 0.0);

        // r × impulse = (0,1,0) × (1,0,0) = (0,0,-1)
        // Δω = (0, 0, -1) * 0.8 = (0, 0, -0.8)
        let delta = angular_velocity_delta(r, impulse, &inertia, 0.8);
        assert!((delta.0).abs() < 1e-10);
        assert!((delta.1).abs() < 1e-10);
        assert!((delta.2 + 0.8).abs() < 1e-10);
    }

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
        assert!((d - 32.0).abs() < 1e-10);
    }

    #[test]
    fn test_magnitude() {
        let v = (3.0, 4.0, 0.0);
        assert!((magnitude(v) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_magnitude_squared() {
        let v = (3.0, 4.0, 0.0);
        assert!((magnitude_squared(v) - 25.0).abs() < 1e-10);
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
    fn test_negate() {
        let v = (1.0, -2.0, 3.0);
        let n = negate(v);
        assert_eq!(n, (-1.0, 2.0, -3.0));
    }

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
}
