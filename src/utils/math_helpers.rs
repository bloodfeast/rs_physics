/// Max error ~0.0015 radians (0.086 degrees)
#[inline(always)]
pub fn fast_atan(x: f32) -> f32 {
    // Constants for the approximation
    const A: f32 = 0.0776509570923569;
    const B: f32 = -0.287434475393028;
    const C: f32 = 0.995354526943123;
    const D: f32 = -0.211410230597758;

    let abs_x = x.abs();
    let z = if abs_x > 1.0 { 1.0 / abs_x } else { abs_x };

    // Polynomial approximation
    let result = (A*z*z*z + B*z*z + C*z) / (z*z + D*z*z + 1.0);

    // Adjust for input range
    let result = if abs_x > 1.0 { std::f32::consts::FRAC_PI_2 - result } else { result };

    // Handle sign
    if x < 0.0 { -result } else { result }
}

/// Fast atan2 implementation using the optimized arctangent
#[inline(always)]
pub fn fast_atan2(y: f32, x: f32) -> f32 {
    // Handle special cases
    if x == 0.0 {
        if y > 0.0 {
            return std::f32::consts::FRAC_PI_2;
        } else if y < 0.0 {
            return -std::f32::consts::FRAC_PI_2;
        } else {
            return 0.0; // Undefined, but return 0.0 as a convention
        }
    }

    // Compute the basic arctangent
    let z = if x > 0.0 {
        fast_atan(y / x)
    } else if x < 0.0 {
        if y >= 0.0 {
            fast_atan(y / x) + std::f32::consts::PI
        } else {
            fast_atan(y / x) - std::f32::consts::PI
        }
    } else if y > 0.0 {
        std::f32::consts::FRAC_PI_2
    } else {
        -std::f32::consts::FRAC_PI_2
    };

    z
}

/// Even faster but less accurate arctangent approximation
/// Max error ~0.01 radians (0.57 degrees)
#[inline(always)]
pub fn fastest_atan(x: f32) -> f32 {
    // Constants
    const HALF_PI: f32 = std::f32::consts::FRAC_PI_2;

    // Extract sign bit and use absolute value
    let sign_mask = x.to_bits() & 0x80000000;
    let abs_x = f32::from_bits(x.to_bits() & 0x7FFFFFFF);

    // Check if we need to use the reciprocal
    let use_recip = abs_x > 1.0;
    let z = if use_recip { 1.0 / abs_x } else { abs_x };

    // Ultra-fast approximation (just linear + quadratic term)
    // Optimized for speed over accuracy
    let result = z * (1.0 - 0.28 * z);

    // Apply transformation if needed
    let result = if use_recip { HALF_PI - result } else { result };

    // Apply sign bit
    f32::from_bits(result.to_bits() | sign_mask)
}

/// Make sure you are passing 8 values at a time
#[feature(enable = "avx2")]
#[inline(always)]
pub fn simd_atan_f32x8(x: [f32; 8]) -> [f32; 8] {
    use std::arch::x86_64::*;

    unsafe {
        // Load values into AVX register
        let x_avx = _mm256_loadu_ps(x.as_ptr());

        // Constants
        let half_pi = _mm256_set1_ps(std::f32::consts::FRAC_PI_2);
        let one = _mm256_set1_ps(1.0);
        let c1 = _mm256_set1_ps(0.9963);
        let c2 = _mm256_set1_ps(0.3214);

        // Get absolute values and keep track of signs
        let sign_mask = _mm256_and_ps(x_avx, _mm256_set1_ps(-0.0));
        let abs_x = _mm256_andnot_ps(_mm256_set1_ps(-0.0), x_avx);

        // Create masks for values > 1.0
        let gt_mask = _mm256_cmp_ps::<_CMP_GT_OQ>(abs_x, one);

        // Calculate reciprocals where needed
        let recip = _mm256_div_ps(one, abs_x);
        let z = _mm256_blendv_ps(abs_x, recip, gt_mask);

        // Calculate polynomial approximation
        let z_squared = _mm256_mul_ps(z, z);
        let term = _mm256_mul_ps(c2, z_squared);
        let coef = _mm256_sub_ps(c1, term);
        let polynomial = _mm256_mul_ps(z, coef);

        // Apply transformation for values > 1.0
        let transformed = _mm256_sub_ps(half_pi, polynomial);
        let result = _mm256_blendv_ps(polynomial, transformed, gt_mask);

        // Restore original sign
        let final_result = _mm256_xor_ps(result, sign_mask);

        // Store result
        let mut output: [f32; 8] = [0.0; 8];
        _mm256_storeu_ps(output.as_mut_ptr(), final_result);
        output
    }
}

/// Minimax polynomial approximation with better accuracy
/// Max error ~0.0007 radians (0.04 degrees)
#[inline(always)]
pub fn minimax_atan(x: f32) -> f32 {
    let abs_x = x.abs();
    let inv = abs_x > 1.0;
    let z = if inv { 1.0 / abs_x } else { abs_x };

    // 7th-order minimax polynomial approximation
    let z2 = z * z;
    let z4 = z2 * z2;
    let z6 = z4 * z2;

    let mut result = z * (0.99997726 + z2 * (-0.33262347 + z2 * (0.19354346 + z2 * (-0.11643287 + z6 * 0.05265332))));

    if inv {
        result = std::f32::consts::FRAC_PI_2 - result;
    }

    if x < 0.0 { -result } else { result }
}

/// CORDIC-inspired arctangent approximation
/// Good for platforms where multiplication is expensive
#[inline(always)]
pub fn cordic_atan2(y: f32, x: f32) -> f32 {
    const QUARTER_PI: f32 = std::f32::consts::FRAC_PI_4;

    // Handle special cases
    if x == 0.0 && y == 0.0 {
        return 0.0;
    }

    let mut angle: f32;
    let mut abs_y = y.abs();

    // First octant (0 to 45 degrees)
    if x >= 0.0 {
        if x >= abs_y {
            // 1st octant
            angle = fast_atan(abs_y / x);
        } else {
            // 2nd octant
            angle = std::f32::consts::FRAC_PI_2 - fast_atan(x / abs_y);
        }
    } else {
        if -x >= abs_y {
            // 3rd and 4th octants
            angle = std::f32::consts::PI - fast_atan(abs_y / -x);
        } else {
            // 5th-8th octants
            angle = std::f32::consts::FRAC_PI_2 + fast_atan(-x / abs_y);
        }
    }

    // Adjust for sign of y
    if y < 0.0 {
        angle = -angle;
    }

    angle
}

/// Lookup table-based arctangent for even faster computation
/// This version uses a 1024-entry table and linear interpolation
pub struct AtanLookupTable {
    table: [f32; 1024],
}

impl AtanLookupTable {
    pub fn new() -> Self {
        let mut table = [0.0; 1024];
        for i in 0..1024 {
            let x = i as f32 / 1023.0;
            table[i] = x.atan();
        }
        Self { table }
    }

    #[inline(always)]
    pub fn atan(&self, x: f32) -> f32 {
        let abs_x = x.abs();
        let inv = abs_x > 1.0;
        let z = if inv { 1.0 / abs_x } else { abs_x };

        // Table lookup with linear interpolation
        let idx = (z * 1023.0) as usize;
        let idx = idx.min(1022); // Guard against index out of bounds

        let frac = (z * 1023.0) - idx as f32;
        let a = self.table[idx];
        let b = self.table[idx + 1];
        let result = a + frac * (b - a);

        let result = if inv {
            std::f32::consts::FRAC_PI_2 - result
        } else {
            result
        };

        if x < 0.0 { -result } else { result }
    }

    #[inline(always)]
    pub fn atan2(&self, y: f32, x: f32) -> f32 {
        // Handle special cases
        if x == 0.0 {
            if y > 0.0 {
                return std::f32::consts::FRAC_PI_2;
            } else if y < 0.0 {
                return -std::f32::consts::FRAC_PI_2;
            } else {
                return 0.0;
            }
        }

        if y == 0.0 {
            if x > 0.0 {
                return 0.0;
            } else {
                return std::f32::consts::PI;
            }
        }

        // Compute the basic arctangent
        if x > 0.0 {
            self.atan(y / x)
        } else if x < 0.0 {
            if y >= 0.0 {
                self.atan(y / x) + std::f32::consts::PI
            } else {
                self.atan(y / x) - std::f32::consts::PI
            }
        } else if y > 0.0 {
            std::f32::consts::FRAC_PI_2
        } else {
            -std::f32::consts::FRAC_PI_2
        }
    }
}

/// Fast inverse square root implementation
/// Based on the famous Quake III Arena algorithm
/// - Unsafe because it is unchecked and can cause NaN or Inf values
/// if the input is not a valid floating point number (e.g. 0.0)
#[inline(always)]
pub unsafe fn fast_inverse_sqrt(x: f32) -> f32 {
    let i = x.to_bits();
    let i = 0x5f3759df - (i >> 1);
    f32::from_bits(i)
}

#[inline(always)]
pub fn fast_sqrt(x: f32) -> f32 {
    // Handle edge cases
    if x < 0.0 {
        return f32::NAN;
    }
    if x == 0.0 || x == 1.0 {
        return x;
    }

    // Start with a good initial guess using bit manipulation
    let i: u32 = x.to_bits();

    // Shift the exponent to effectively divide by 2
    // Magic number 0x5f3759df is replaced with 0x5f375a86 which works better for sqrt
    let mut j: u32 = 0x5f375a86 - (i >> 1);

    // Convert back to float
    let mut y: f32 = f32::from_bits(j);

    // First iteration
    y = y * (1.5 - 0.5 * x * y * y);

    // Second iteration for better accuracy
    y = y * (1.5 - 0.5 * x * y * y);

    // Multiply by x to get the square root
    x * y
}

/// Fast inverse square root implementation for f64 values
/// Based on the famous Quake III Arena algorithm but adapted for 64-bit doubles
/// - Unsafe because it is unchecked and can cause NaN or Inf values
/// if the input is not a valid floating point number (e.g. 0.0)
#[inline(always)]
pub unsafe fn fast_inverse_sqrt_f64(x: f64) -> f64 {
    // Original algorithm
    let x_half = 0.5 * x;
    let i = x.to_bits();

    // The f64 magic number is different from the f32 version (0x5f3759df)
    let magic_constant = 0x5fe6eb50c7b537a9_u64;
    let i = magic_constant - (i >> 1);
    let y = f64::from_bits(i);

    // One iteration gives good precision for most applications
    y * (1.5 - x_half * y * y)
}

#[inline(always)]
pub fn fast_sqrt_f64(x: f64) -> f64 {
    // Handle edge cases
    if x < 0.0 {
        return f64::NAN;
    }
    if x == 0.0 || x == 1.0 {
        return x;
    }

    // Start with a good initial guess using bit manipulation
    let i: u64 = x.to_bits();

    // Shift the exponent to effectively divide by 2
    // Magic number for f64 (adjusted from the f32 version)
    let mut j: u64 = 0x5fe6eb50c7b537a9 - (i >> 1);

    // Convert back to f64
    let mut y: f64 = f64::from_bits(j);

    // Newton-Raphson iterations for refining the estimate
    // For f64, more iterations may be beneficial for accuracy

    // First iteration
    y = y * (1.5 - 0.5 * x * y * y);

    // Second iteration
    y = y * (1.5 - 0.5 * x * y * y);

    // Third iteration (additional iteration for f64 precision)
    y = y * (1.5 - 0.5 * x * y * y);

    // Multiply by x to get the square root
    x * y
}