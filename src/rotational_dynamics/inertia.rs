//! Inertia traits and types for 2D and 3D rotational dynamics
//!
//! This module provides unified interfaces for working with moments of inertia
//! in both 2D (scalar) and 3D (tensor) contexts.

use crate::utils::PhysicsError;

//==============================================================================
// INERTIA TYPES
//==============================================================================

/// A scalar moment of inertia for 2D rotation (rotation around a single axis)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InertiaScalar(pub f64);

impl InertiaScalar {
    /// Create a new scalar inertia value
    pub fn new(value: f64) -> Result<Self, PhysicsError> {
        if value <= 0.0 {
            return Err(PhysicsError::InvalidMass);
        }
        Ok(Self(value))
    }

    /// Get the inertia value
    #[inline]
    pub fn value(&self) -> f64 {
        self.0
    }

    /// Get the inverse inertia (1/I), returns 0 for infinite inertia
    #[inline]
    pub fn inverse(&self) -> f64 {
        if self.0 <= 0.0 || self.0.is_infinite() {
            0.0
        } else {
            1.0 / self.0
        }
    }
}

impl Default for InertiaScalar {
    fn default() -> Self {
        Self(1.0)
    }
}

impl From<f64> for InertiaScalar {
    fn from(value: f64) -> Self {
        Self(value.max(0.0))
    }
}

/// A 3x3 symmetric inertia tensor for 3D rotation
/// Stored as [Ixx, Iyy, Izz, Ixy, Ixz, Iyz]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InertiaTensor {
    /// Diagonal elements: Ixx, Iyy, Izz
    pub diagonal: (f64, f64, f64),
    /// Off-diagonal elements: Ixy, Ixz, Iyz (symmetric, so only 3 values needed)
    pub off_diagonal: (f64, f64, f64),
}

impl InertiaTensor {
    /// Create a new inertia tensor from all 6 components
    pub fn new(ixx: f64, iyy: f64, izz: f64, ixy: f64, ixz: f64, iyz: f64) -> Self {
        Self {
            diagonal: (ixx, iyy, izz),
            off_diagonal: (ixy, ixz, iyz),
        }
    }

    /// Create a diagonal inertia tensor (no off-diagonal terms)
    pub fn diagonal_only(ixx: f64, iyy: f64, izz: f64) -> Self {
        Self {
            diagonal: (ixx, iyy, izz),
            off_diagonal: (0.0, 0.0, 0.0),
        }
    }

    /// Create a uniform/isotropic inertia tensor (same value on all diagonals)
    pub fn uniform(i: f64) -> Self {
        Self::diagonal_only(i, i, i)
    }

    /// Create from a 6-element array [Ixx, Iyy, Izz, Ixy, Ixz, Iyz]
    pub fn from_array(arr: [f64; 6]) -> Self {
        Self::new(arr[0], arr[1], arr[2], arr[3], arr[4], arr[5])
    }

    /// Convert to a 6-element array [Ixx, Iyy, Izz, Ixy, Ixz, Iyz]
    pub fn to_array(&self) -> [f64; 6] {
        [
            self.diagonal.0,
            self.diagonal.1,
            self.diagonal.2,
            self.off_diagonal.0,
            self.off_diagonal.1,
            self.off_diagonal.2,
        ]
    }

    /// Get the inverse inertia tensor for angular acceleration calculations
    /// For a diagonal tensor, this is simply 1/Ixx, 1/Iyy, 1/Izz
    /// For non-diagonal tensors, this computes the full matrix inverse
    pub fn inverse(&self) -> InertiaTensor {
        let (ixx, iyy, izz) = self.diagonal;
        let (ixy, ixz, iyz) = self.off_diagonal;

        // For diagonal tensors, simple inversion
        if ixy.abs() < 1e-10 && ixz.abs() < 1e-10 && iyz.abs() < 1e-10 {
            return InertiaTensor::diagonal_only(
                if ixx > 1e-10 { 1.0 / ixx } else { 0.0 },
                if iyy > 1e-10 { 1.0 / iyy } else { 0.0 },
                if izz > 1e-10 { 1.0 / izz } else { 0.0 },
            );
        }

        // Full 3x3 symmetric matrix inversion
        // | ixx  ixy  ixz |
        // | ixy  iyy  iyz |
        // | ixz  iyz  izz |
        let det = ixx * (iyy * izz - iyz * iyz)
            - ixy * (ixy * izz - iyz * ixz)
            + ixz * (ixy * iyz - iyy * ixz);

        if det.abs() < 1e-10 {
            // Singular matrix, return zero inverse
            return InertiaTensor::diagonal_only(0.0, 0.0, 0.0);
        }

        let inv_det = 1.0 / det;

        // Cofactor matrix (transposed for inverse)
        let inv_ixx = (iyy * izz - iyz * iyz) * inv_det;
        let inv_iyy = (ixx * izz - ixz * ixz) * inv_det;
        let inv_izz = (ixx * iyy - ixy * ixy) * inv_det;
        let inv_ixy = (ixz * iyz - ixy * izz) * inv_det;
        let inv_ixz = (ixy * iyz - iyy * ixz) * inv_det;
        let inv_iyz = (ixy * ixz - ixx * iyz) * inv_det;

        InertiaTensor::new(inv_ixx, inv_iyy, inv_izz, inv_ixy, inv_ixz, inv_iyz)
    }

    /// Multiply the inertia tensor by an angular velocity vector to get angular momentum
    /// L = I * ω
    pub fn multiply_vector(&self, omega: (f64, f64, f64)) -> (f64, f64, f64) {
        let (ixx, iyy, izz) = self.diagonal;
        let (ixy, ixz, iyz) = self.off_diagonal;
        let (wx, wy, wz) = omega;

        (
            ixx * wx + ixy * wy + ixz * wz,
            ixy * wx + iyy * wy + iyz * wz,
            ixz * wx + iyz * wy + izz * wz,
        )
    }

    /// Apply the inverse inertia tensor to a torque to get angular acceleration
    /// α = I⁻¹ * τ
    ///
    /// # Cost
    ///
    /// **This recomputes [`InertiaTensor::inverse`] on every call** — a determinant and
    /// six cofactors for a general tensor, three divisions for a diagonal one. The
    /// inverse is a property of the body, not of the call, so anything invoking this
    /// per body per step should hold the inverse instead. [`RigidBodyRotation`] does
    /// exactly that: it computes the inverse once in its constructor and never again.
    ///
    /// [`RigidBodyRotation`]: crate::rotational_dynamics::RigidBodyRotation
    pub fn apply_inverse_to_torque(&self, torque: (f64, f64, f64)) -> (f64, f64, f64) {
        self.inverse().multiply_vector(torque)
    }

    /// Every component is finite — no `NaN`, no `±∞`.
    #[inline]
    pub fn is_finite(&self) -> bool {
        self.diagonal.0.is_finite()
            && self.diagonal.1.is_finite()
            && self.diagonal.2.is_finite()
            && self.off_diagonal.0.is_finite()
            && self.off_diagonal.1.is_finite()
            && self.off_diagonal.2.is_finite()
    }

    /// **The inverse exists and every principal moment is strictly positive.**
    ///
    /// A symmetric matrix is positive definite iff its three leading principal minors
    /// are positive (Sylvester's criterion), which is checked directly here rather than
    /// by extracting eigenvalues. Positive definiteness is what makes `I⁻¹` finite; a
    /// tensor with a zero moment about some axis — `inertia_3d::thin_rod_center` returns
    /// one, deliberately, as an idealisation — gives that axis infinite angular
    /// acceleration and cannot be integrated.
    ///
    /// The tolerance is relative to the trace, so the answer does not depend on whether
    /// the body is measured in kg·m² or g·cm².
    pub fn is_positive_definite(&self) -> bool {
        if !self.is_finite() {
            return false;
        }
        let (ixx, iyy, izz) = self.diagonal;
        let (ixy, ixz, iyz) = self.off_diagonal;

        let scale = (ixx.abs() + iyy.abs() + izz.abs()) / 3.0;
        if scale <= 0.0 {
            return false;
        }
        let eps = 1e-12;

        // Leading principal minors of
        //   | ixx  ixy  ixz |
        //   | ixy  iyy  iyz |
        //   | ixz  iyz  izz |
        let m1 = ixx;
        let m2 = ixx * iyy - ixy * ixy;
        let m3 = ixx * (iyy * izz - iyz * iyz) - ixy * (ixy * izz - iyz * ixz)
            + ixz * (ixy * iyz - iyy * ixz);

        m1 > eps * scale && m2 > eps * scale * scale && m3 > eps * scale * scale * scale
    }

    /// **The principal moments obey `I₁ + I₂ ≥ I₃`** — the tensor could have come from
    /// a real mass distribution.
    ///
    /// No arrangement of positive mass can violate this: each moment is an integral of
    /// the squared distance from one axis, and the three integrands satisfy the
    /// inequality pointwise. A tensor that violates it is a data error, and it is worth
    /// catching because **the integrator's error bound depends on it**: in principal
    /// axes `ω̇ₓ = (I_y − I_z)/I_x · ω_y ω_z`, and the triangle inequality is exactly
    /// what bounds that coefficient by 1, hence `|ω̇| ≲ |ω|²` and hence the substep rule
    /// in [`RigidBodyRotation::step`]. Without it there is no bound on how fast the
    /// gyroscopic term can move `ω`, and a fixed substep count means nothing.
    ///
    /// Checked without an eigendecomposition: the inequality on the eigenvalues of `I`
    /// holds iff `C = (tr I / 2)·Id − I` is positive semi-definite, because the
    /// eigenvalues of `C` are exactly `(Iⱼ + I_k − Iᵢ)/2`. Positive semi-definiteness
    /// of a symmetric 3×3 needs *all* seven principal minors to be non-negative, not
    /// just the leading three, so all seven are tested.
    ///
    /// [`RigidBodyRotation::step`]: crate::rotational_dynamics::RigidBodyRotation::step
    pub fn satisfies_triangle_inequality(&self) -> bool {
        if !self.is_finite() {
            return false;
        }
        let (ixx, iyy, izz) = self.diagonal;
        let (ixy, ixz, iyz) = self.off_diagonal;

        let half_trace = 0.5 * (ixx + iyy + izz);
        // C = (tr/2) Id - I
        let cxx = half_trace - ixx;
        let cyy = half_trace - iyy;
        let czz = half_trace - izz;
        let cxy = -ixy;
        let cxz = -ixz;
        let cyz = -iyz;

        let scale = (ixx.abs() + iyy.abs() + izz.abs()) / 3.0;
        if scale <= 0.0 {
            return false;
        }
        let eps = 1e-12;
        let (t1, t2, t3) = (eps * scale, eps * scale * scale, eps * scale * scale * scale);

        // 1x1 principal minors
        if cxx < -t1 || cyy < -t1 || czz < -t1 {
            return false;
        }
        // 2x2 principal minors
        if cxx * cyy - cxy * cxy < -t2
            || cxx * czz - cxz * cxz < -t2
            || cyy * czz - cyz * cyz < -t2
        {
            return false;
        }
        // 3x3
        let det = cxx * (cyy * czz - cyz * cyz) - cxy * (cxy * czz - cyz * cxz)
            + cxz * (cxy * cyz - cyy * cxz);
        det >= -t3
    }
}

impl Default for InertiaTensor {
    fn default() -> Self {
        Self::uniform(1.0)
    }
}

impl From<[f64; 6]> for InertiaTensor {
    fn from(arr: [f64; 6]) -> Self {
        Self::from_array(arr)
    }
}

impl From<InertiaTensor> for [f64; 6] {
    fn from(tensor: InertiaTensor) -> Self {
        tensor.to_array()
    }
}

//==============================================================================
// MOMENT OF INERTIA CALCULATIONS
//==============================================================================

/// Shape types for 2D moment of inertia calculations
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Shape2D {
    /// A solid disk with the given radius
    Disk(f64),
    /// A hollow ring with inner and outer radii
    Ring(f64, f64),
    /// A solid rectangle with width and height
    Rectangle(f64, f64),
    /// A thin rod with length (rotating about center)
    Rod(f64),
    /// A thin rod rotating about one end
    RodEnd(f64),
}

impl Shape2D {
    /// Calculate the moment of inertia for this 2D shape
    pub fn moment_of_inertia(&self, mass: f64) -> InertiaScalar {
        let i = match self {
            Shape2D::Disk(r) => 0.5 * mass * r * r,
            Shape2D::Ring(r_inner, r_outer) => {
                0.5 * mass * (r_inner * r_inner + r_outer * r_outer)
            }
            Shape2D::Rectangle(w, h) => (1.0 / 12.0) * mass * (w * w + h * h),
            Shape2D::Rod(length) => (1.0 / 12.0) * mass * length * length,
            Shape2D::RodEnd(length) => (1.0 / 3.0) * mass * length * length,
        };
        InertiaScalar(i)
    }
}

/// Common 3D shape inertia calculations
/// Note: Shape3D in models/shape_3d.rs already has moment_of_inertia(),
/// these are additional utility functions
pub mod inertia_3d {
    use super::InertiaTensor;

    /// Solid sphere: I = (2/5) * m * r²
    pub fn solid_sphere(mass: f64, radius: f64) -> InertiaTensor {
        let i = (2.0 / 5.0) * mass * radius * radius;
        InertiaTensor::uniform(i)
    }

    /// Hollow sphere (thin shell): I = (2/3) * m * r²
    pub fn hollow_sphere(mass: f64, radius: f64) -> InertiaTensor {
        let i = (2.0 / 3.0) * mass * radius * radius;
        InertiaTensor::uniform(i)
    }

    /// Solid cylinder (about axis of symmetry)
    /// Ixx = Iyy = (1/12) * m * (3r² + h²)
    /// Izz = (1/2) * m * r²
    pub fn solid_cylinder(mass: f64, radius: f64, height: f64) -> InertiaTensor {
        let ixx_iyy = (1.0 / 12.0) * mass * (3.0 * radius * radius + height * height);
        let izz = 0.5 * mass * radius * radius;
        InertiaTensor::diagonal_only(ixx_iyy, ixx_iyy, izz)
    }

    /// Solid cuboid (box)
    /// Ixx = (1/12) * m * (h² + d²)
    /// Iyy = (1/12) * m * (w² + d²)
    /// Izz = (1/12) * m * (w² + h²)
    pub fn solid_cuboid(mass: f64, width: f64, height: f64, depth: f64) -> InertiaTensor {
        let ixx = (1.0 / 12.0) * mass * (height * height + depth * depth);
        let iyy = (1.0 / 12.0) * mass * (width * width + depth * depth);
        let izz = (1.0 / 12.0) * mass * (width * width + height * height);
        InertiaTensor::diagonal_only(ixx, iyy, izz)
    }

    /// Thin rod (about center, perpendicular to rod)
    /// I = (1/12) * m * L²
    pub fn thin_rod_center(mass: f64, length: f64) -> InertiaTensor {
        let i = (1.0 / 12.0) * mass * length * length;
        // Rod along z-axis, rotation about x or y
        InertiaTensor::diagonal_only(i, i, 0.0)
    }

    /// Thin rod (about end, perpendicular to rod)
    /// I = (1/3) * m * L²
    pub fn thin_rod_end(mass: f64, length: f64) -> InertiaTensor {
        let i = (1.0 / 3.0) * mass * length * length;
        InertiaTensor::diagonal_only(i, i, 0.0)
    }

    /// Apply parallel axis theorem to shift inertia tensor
    /// I' = I + m * d²
    /// where d is the distance from the center of mass
    pub fn parallel_axis(
        base: InertiaTensor,
        mass: f64,
        offset: (f64, f64, f64),
    ) -> InertiaTensor {
        let (dx, dy, dz) = offset;
        let _d_sq = dx * dx + dy * dy + dz * dz;

        // Parallel axis additions
        let add_ixx = mass * (dy * dy + dz * dz);
        let add_iyy = mass * (dx * dx + dz * dz);
        let add_izz = mass * (dx * dx + dy * dy);
        let add_ixy = -mass * dx * dy;
        let add_ixz = -mass * dx * dz;
        let add_iyz = -mass * dy * dz;

        InertiaTensor::new(
            base.diagonal.0 + add_ixx,
            base.diagonal.1 + add_iyy,
            base.diagonal.2 + add_izz,
            base.off_diagonal.0 + add_ixy,
            base.off_diagonal.1 + add_ixz,
            base.off_diagonal.2 + add_iyz,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== InertiaScalar Tests ====================

    #[test]
    fn test_inertia_scalar() {
        let i = InertiaScalar::new(2.0).unwrap();
        assert_eq!(i.value(), 2.0);
        assert!((i.inverse() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_inertia_scalar_invalid() {
        assert!(InertiaScalar::new(0.0).is_err());
        assert!(InertiaScalar::new(-1.0).is_err());
    }

    #[test]
    fn test_inertia_scalar_from_f64() {
        let i = InertiaScalar::from(5.0);
        assert_eq!(i.value(), 5.0);

        // Negative values should clamp to 0
        let i_neg = InertiaScalar::from(-5.0);
        assert_eq!(i_neg.value(), 0.0);
    }

    #[test]
    fn test_inertia_scalar_inverse_edge_cases() {
        // Zero inertia should return 0 inverse (infinite mass behavior)
        let zero = InertiaScalar::from(0.0);
        assert_eq!(zero.inverse(), 0.0);

        // Infinite inertia should return 0 inverse
        let inf = InertiaScalar::from(f64::INFINITY);
        assert_eq!(inf.inverse(), 0.0);
    }

    #[test]
    fn test_inertia_scalar_default() {
        let i = InertiaScalar::default();
        assert_eq!(i.value(), 1.0);
    }

    // ==================== InertiaTensor Tests ====================

    #[test]
    fn test_inertia_tensor_diagonal() {
        let t = InertiaTensor::diagonal_only(1.0, 2.0, 3.0);
        let inv = t.inverse();
        assert!((inv.diagonal.0 - 1.0).abs() < 1e-10);
        assert!((inv.diagonal.1 - 0.5).abs() < 1e-10);
        assert!((inv.diagonal.2 - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_inertia_tensor_uniform() {
        let t = InertiaTensor::uniform(5.0);
        assert_eq!(t.diagonal.0, 5.0);
        assert_eq!(t.diagonal.1, 5.0);
        assert_eq!(t.diagonal.2, 5.0);
        assert_eq!(t.off_diagonal.0, 0.0);
        assert_eq!(t.off_diagonal.1, 0.0);
        assert_eq!(t.off_diagonal.2, 0.0);
    }

    #[test]
    fn test_inertia_tensor_from_array() {
        let arr = [1.0, 2.0, 3.0, 0.1, 0.2, 0.3];
        let t = InertiaTensor::from_array(arr);
        assert_eq!(t.diagonal, (1.0, 2.0, 3.0));
        assert_eq!(t.off_diagonal, (0.1, 0.2, 0.3));

        let back: [f64; 6] = t.into();
        assert_eq!(back, arr);
    }

    #[test]
    fn test_inertia_tensor_full_inverse() {
        // Non-diagonal tensor
        let t = InertiaTensor::new(4.0, 5.0, 6.0, 0.5, 0.3, 0.2);
        let inv = t.inverse();

        // Verify I * I^-1 ≈ Identity by checking multiply_vector
        // Apply to a test vector and verify result
        let test_vec = (1.0, 2.0, 3.0);
        let temp = t.multiply_vector(test_vec);
        let result = inv.multiply_vector(temp);

        assert!((result.0 - test_vec.0).abs() < 1e-8);
        assert!((result.1 - test_vec.1).abs() < 1e-8);
        assert!((result.2 - test_vec.2).abs() < 1e-8);
    }

    #[test]
    fn test_tensor_multiply_vector() {
        let t = InertiaTensor::diagonal_only(2.0, 3.0, 4.0);
        let omega = (1.0, 2.0, 3.0);
        let l = t.multiply_vector(omega);
        assert!((l.0 - 2.0).abs() < 1e-10);
        assert!((l.1 - 6.0).abs() < 1e-10);
        assert!((l.2 - 12.0).abs() < 1e-10);
    }

    #[test]
    fn test_tensor_apply_inverse_to_torque() {
        let t = InertiaTensor::diagonal_only(2.0, 4.0, 8.0);
        let torque = (4.0, 8.0, 16.0);
        let alpha = t.apply_inverse_to_torque(torque);
        // α = I^-1 * τ = (4/2, 8/4, 16/8) = (2, 2, 2)
        assert!((alpha.0 - 2.0).abs() < 1e-10);
        assert!((alpha.1 - 2.0).abs() < 1e-10);
        assert!((alpha.2 - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_inertia_tensor_default() {
        let t = InertiaTensor::default();
        assert_eq!(t.diagonal, (1.0, 1.0, 1.0));
        assert_eq!(t.off_diagonal, (0.0, 0.0, 0.0));
    }

    // ==================== Shape2D Tests ====================

    #[test]
    fn test_shape2d_disk() {
        let disk = Shape2D::Disk(2.0);
        let i = disk.moment_of_inertia(1.0);
        // I = 0.5 * m * r² = 0.5 * 1 * 4 = 2
        assert!((i.value() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_shape2d_ring() {
        let ring = Shape2D::Ring(1.0, 2.0);
        let i = ring.moment_of_inertia(1.0);
        // I = 0.5 * m * (r_inner² + r_outer²) = 0.5 * 1 * (1 + 4) = 2.5
        assert!((i.value() - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_shape2d_rectangle() {
        let rect = Shape2D::Rectangle(2.0, 4.0);
        let i = rect.moment_of_inertia(1.0);
        // I = (1/12) * m * (w² + h²) = (1/12) * 1 * (4 + 16) = 20/12 ≈ 1.667
        assert!((i.value() - 20.0 / 12.0).abs() < 1e-10);
    }

    #[test]
    fn test_shape2d_rod() {
        let rod = Shape2D::Rod(3.0);
        let i = rod.moment_of_inertia(1.0);
        // I = (1/12) * m * L² = (1/12) * 1 * 9 = 0.75
        assert!((i.value() - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_shape2d_rod_end() {
        let rod = Shape2D::RodEnd(3.0);
        let i = rod.moment_of_inertia(1.0);
        // I = (1/3) * m * L² = (1/3) * 1 * 9 = 3.0
        assert!((i.value() - 3.0).abs() < 1e-10);
    }

    // ==================== 3D Inertia Utilities ====================

    #[test]
    fn test_solid_sphere() {
        let t = inertia_3d::solid_sphere(1.0, 1.0);
        // I = (2/5) * m * r² = 0.4
        assert!((t.diagonal.0 - 0.4).abs() < 1e-10);
        assert!((t.diagonal.1 - 0.4).abs() < 1e-10);
        assert!((t.diagonal.2 - 0.4).abs() < 1e-10);
    }

    #[test]
    fn test_hollow_sphere() {
        let t = inertia_3d::hollow_sphere(1.0, 1.0);
        // I = (2/3) * m * r² ≈ 0.667
        let expected = 2.0 / 3.0;
        assert!((t.diagonal.0 - expected).abs() < 1e-10);
        assert!((t.diagonal.1 - expected).abs() < 1e-10);
        assert!((t.diagonal.2 - expected).abs() < 1e-10);
    }

    #[test]
    fn test_solid_cylinder() {
        let t = inertia_3d::solid_cylinder(1.0, 1.0, 2.0);
        // Izz = 0.5 * m * r² = 0.5
        // Ixx = Iyy = (1/12) * m * (3r² + h²) = (1/12) * (3 + 4) ≈ 0.583
        let expected_xx = (1.0 / 12.0) * (3.0 + 4.0);
        assert!((t.diagonal.0 - expected_xx).abs() < 1e-10);
        assert!((t.diagonal.1 - expected_xx).abs() < 1e-10);
        assert!((t.diagonal.2 - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_solid_cuboid() {
        let t = inertia_3d::solid_cuboid(1.0, 2.0, 3.0, 4.0);
        // Ixx = (1/12) * m * (h² + d²) = (1/12) * (9 + 16) ≈ 2.083
        // Iyy = (1/12) * m * (w² + d²) = (1/12) * (4 + 16) ≈ 1.667
        // Izz = (1/12) * m * (w² + h²) = (1/12) * (4 + 9) ≈ 1.083
        let expected_xx = (1.0 / 12.0) * (9.0 + 16.0);
        let expected_yy = (1.0 / 12.0) * (4.0 + 16.0);
        let expected_zz = (1.0 / 12.0) * (4.0 + 9.0);
        assert!((t.diagonal.0 - expected_xx).abs() < 1e-10);
        assert!((t.diagonal.1 - expected_yy).abs() < 1e-10);
        assert!((t.diagonal.2 - expected_zz).abs() < 1e-10);
    }

    #[test]
    fn test_thin_rod_center() {
        let t = inertia_3d::thin_rod_center(1.0, 2.0);
        // I = (1/12) * m * L² = (1/12) * 4 ≈ 0.333
        let expected = (1.0 / 12.0) * 4.0;
        assert!((t.diagonal.0 - expected).abs() < 1e-10);
        assert!((t.diagonal.1 - expected).abs() < 1e-10);
        assert!((t.diagonal.2).abs() < 1e-10); // No rotation about rod axis
    }

    #[test]
    fn test_thin_rod_end() {
        let t = inertia_3d::thin_rod_end(1.0, 2.0);
        // I = (1/3) * m * L² = (1/3) * 4 ≈ 1.333
        let expected = (1.0 / 3.0) * 4.0;
        assert!((t.diagonal.0 - expected).abs() < 1e-10);
        assert!((t.diagonal.1 - expected).abs() < 1e-10);
    }

    #[test]
    fn test_parallel_axis_theorem() {
        // Shift a uniform tensor
        let base = InertiaTensor::uniform(1.0);
        let shifted = inertia_3d::parallel_axis(base, 1.0, (1.0, 0.0, 0.0));

        // Shifting by (1,0,0) adds to Iyy and Izz
        // Ixx unchanged (offset perpendicular to x-axis doesn't affect x rotation)
        assert!((shifted.diagonal.0 - 1.0).abs() < 1e-10);
        // Iyy += m * (dx² + dz²) = 1 * (1 + 0) = 1, so Iyy = 2
        assert!((shifted.diagonal.1 - 2.0).abs() < 1e-10);
        // Izz += m * (dx² + dy²) = 1 * (1 + 0) = 1, so Izz = 2
        assert!((shifted.diagonal.2 - 2.0).abs() < 1e-10);
    }
}
