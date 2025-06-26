use std::f64::consts::PI;

/// Quaternion representation for 3D rotations to avoid gimbal lock
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quaternion {
    pub w: f64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Quaternion {
    /// Creates a new identity quaternion (no rotation)
    pub fn identity() -> Self {
        Self {
            w: 1.0,
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }
    
    pub fn is_near_identity(&self, epsilon: f64) -> bool {
        (self.w - 1.0).abs() < epsilon &&
            self.x.abs() < epsilon &&
            self.y.abs() < epsilon &&
            self.z.abs() < epsilon
    }

    /// Creates a quaternion from axis-angle representation
    pub fn from_axis_angle(axis: (f64, f64, f64), angle: f64) -> Self {
        let half_angle = angle / 2.0;
        let sin_half = half_angle.sin();
        let (ax, ay, az) = axis;
        let magnitude = (ax * ax + ay * ay + az * az).sqrt();

        if magnitude < 1e-10 {
            return Quaternion::identity();
        }

        let nx = ax / magnitude;
        let ny = ay / magnitude;
        let nz = az / magnitude;

        Quaternion {
            w: half_angle.cos(),
            x: nx * sin_half,
            y: ny * sin_half,
            z: nz * sin_half,
        }
    }

    /// Creates a quaternion from Euler angles (roll, pitch, yaw)
    pub fn from_euler(roll: f64, pitch: f64, yaw: f64) -> Self {
        // Convert Euler angles to quaternion using the ZYX convention
        let cy = (yaw * 0.5).cos();
        let sy = (yaw * 0.5).sin();
        let cp = (pitch * 0.5).cos();
        let sp = (pitch * 0.5).sin();
        let cr = (roll * 0.5).cos();
        let sr = (roll * 0.5).sin();

        Quaternion {
            w: cr * cp * cy + sr * sp * sy,
            x: sr * cp * cy - cr * sp * sy,
            y: cr * sp * cy + sr * cp * sy,
            z: cr * cp * sy - sr * sp * cy,
        }
    }

    /// Converts quaternion to Euler angles (roll, pitch, yaw)
    pub fn to_euler(&self) -> (f64, f64, f64) {
        // Normalize quaternion
        let q = self.normalized();

        // Roll (x-axis rotation)
        let sinr_cosp = 2.0 * (q.w * q.x + q.y * q.z);
        let cosr_cosp = 1.0 - 2.0 * (q.x * q.x + q.y * q.y);
        let roll = sinr_cosp.atan2(cosr_cosp);

        // Pitch (y-axis rotation)
        let sinp = 2.0 * (q.w * q.y - q.z * q.x);
        let pitch = if sinp.abs() >= 1.0 {
            (PI / 2.0).copysign(sinp) // Use 90 degrees if out of range
        } else {
            sinp.asin()
        };

        // Yaw (z-axis rotation)
        let siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
        let cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
        let yaw = siny_cosp.atan2(cosy_cosp);

        (roll, pitch, yaw)
    }

    /// Returns the length/magnitude of the quaternion
    pub fn magnitude(&self) -> f64 {
        (self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    /// Returns a normalized version of the quaternion
    pub fn normalized(&self) -> Self {
        let mag = self.magnitude();
        if mag < 1e-10 {
            return Quaternion::identity();
        }
        Quaternion {
            w: self.w / mag,
            x: self.x / mag,
            y: self.y / mag,
            z: self.z / mag,
        }
    }

    /// Multiplies two quaternions (composition of rotations)
    pub fn multiply(&self, other: &Quaternion) -> Quaternion {
        Quaternion {
            w: self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            x: self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            y: self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            z: self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
        }
    }

    /// Returns the conjugate of the quaternion
    pub fn conjugate(&self) -> Quaternion {
        Quaternion {
            w: self.w,
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }

    /// Returns the inverse of the quaternion
    pub fn inverse(&self) -> Quaternion {
        let mag_squared = self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z;
        if mag_squared < 1e-10 {
            return Quaternion::identity();
        }

        let conj = self.conjugate();
        Quaternion {
            w: conj.w / mag_squared,
            x: conj.x / mag_squared,
            y: conj.y / mag_squared,
            z: conj.z / mag_squared,
        }
    }

    /// Rotates a point using the quaternion
    pub fn rotate_point(&self, point: (f64, f64, f64)) -> (f64, f64, f64) {
        // Convert the point to a quaternion with w=0
        let p = Quaternion {
            w: 0.0,
            x: point.0,
            y: point.1,
            z: point.2,
        };

        // Apply rotation: q * p * q^-1
        let q_normalized = self.normalized();
        let q_inv = q_normalized.inverse();
        let rotated = q_normalized.multiply(&p).multiply(&q_inv);

        (rotated.x, rotated.y, rotated.z)
    }

    /// Spherical linear interpolation between two quaternions
    pub fn slerp(&self, other: &Quaternion, t: f64) -> Quaternion {
        let q1 = self.normalized();
        let mut q2 = other.normalized();

        // Calculate cosine of angle between quaternions
        let mut dot = q1.w * q2.w + q1.x * q2.x + q1.y * q2.y + q1.z * q2.z;

        // If the dot product is negative, we need to invert one quaternion to take the shorter path
        if dot < 0.0 {
            q2 = Quaternion { w: -q2.w, x: -q2.x, y: -q2.y, z: -q2.z };
            dot = -dot;
        }

        // If the inputs are too close, linearly interpolate and normalize the result
        const DOT_THRESHOLD: f64 = 0.9995;
        if dot > DOT_THRESHOLD {
            let result = Quaternion {
                w: q1.w + t * (q2.w - q1.w),
                x: q1.x + t * (q2.x - q1.x),
                y: q1.y + t * (q2.y - q1.y),
                z: q1.z + t * (q2.z - q1.z),
            };
            return result.normalized();
        }

        // Calculate the angle between the quaternions
        let theta_0 = dot.acos();
        let theta = theta_0 * t;

        // Calculate the interpolated quaternion
        let sin_theta = theta.sin();
        let sin_theta_0 = theta_0.sin();

        // Scale the quaternions
        let s0 = ((1.0 - t) * theta_0).cos() / sin_theta_0;
        let s1 = sin_theta / sin_theta_0;

        Quaternion {
            w: s0 * q1.w + s1 * q2.w,
            x: s0 * q1.x + s1 * q2.x,
            y: s0 * q1.y + s1 * q2.y,
            z: s0 * q1.z + s1 * q2.z,
        }
    }
}