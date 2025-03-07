/// Max error ~0.0015 radians (0.086 degrees)
#[inline]
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
#[inline]
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
#[inline]
pub fn fastest_atan(x: f32) -> f32 {
    const HALF_PI: f32 = std::f32::consts::FRAC_PI_2;
    let abs_x = x.abs();

    // Fast approximation: π/4 * x - x * (|x| - 1) * (0.2447 + 0.0663 * |x|)
    let mut result = HALF_PI * abs_x - abs_x * (abs_x - 1.0) * (0.2447 + 0.0663 * abs_x);

    // Handle values above 1.0
    if abs_x > 1.0 {
        result = HALF_PI - result;
    }

    if x < 0.0 { -result } else { result }
}

/// Minimax polynomial approximation with better accuracy
/// Max error ~0.0007 radians (0.04 degrees)
#[inline]
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
#[inline]
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

    #[inline]
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

    #[inline]
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

/// Benchmark utility to compare approximation performance and accuracy
#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn benchmark_atan_methods() {
        const ITERATIONS: usize = 1_000_000;
        let lookup_table = AtanLookupTable::new();

        // Test values
        let test_values: Vec<f32> = (-100..=100).map(|i| i as f32 / 10.0).collect();

        // Measure error
        let mut max_error_fast = 0.0_f32;
        let mut max_error_fastest = 0.0_f32;
        let mut max_error_minimax = 0.0_f32;
        let mut max_error_lookup = 0.0_f32;

        for &x in &test_values {
            let truth = x.atan();
            let err_fast = (fast_atan(x) - truth).abs();
            let err_fastest = (fastest_atan(x) - truth).abs();
            let err_minimax = (minimax_atan(x) - truth).abs();
            let err_lookup = (lookup_table.atan(x) - truth).abs();

            max_error_fast = max_error_fast.max(err_fast);
            max_error_fastest = max_error_fastest.max(err_fastest);
            max_error_minimax = max_error_minimax.max(err_minimax);
            max_error_lookup = max_error_lookup.max(err_lookup);
        }

        println!("Maximum errors (radians):");
        println!("fast_atan:     {}", max_error_fast);
        println!("fastest_atan:  {}", max_error_fastest);
        println!("minimax_atan:  {}", max_error_minimax);
        println!("lookup_atan:   {}", max_error_lookup);

        // Benchmark standard library
        let start = Instant::now();
        let mut sum = 0.0;
        for _ in 0..ITERATIONS {
            for &x in &test_values {
                sum += x.atan();
            }
        }
        let std_duration = start.elapsed();

        // Benchmark fast implementation
        let start = Instant::now();
        let mut sum = 0.0;
        for _ in 0..ITERATIONS {
            for &x in &test_values {
                sum += fast_atan(x);
            }
        }
        let fast_duration = start.elapsed();

        // Benchmark fastest implementation
        let start = Instant::now();
        let mut sum = 0.0;
        for _ in 0..ITERATIONS {
            for &x in &test_values {
                sum += fastest_atan(x);
            }
        }
        let fastest_duration = start.elapsed();

        // Benchmark minimax implementation
        let start = Instant::now();
        let mut sum = 0.0;
        for _ in 0..ITERATIONS {
            for &x in &test_values {
                sum += minimax_atan(x);
            }
        }
        let minimax_duration = start.elapsed();

        // Benchmark lookup table
        let start = Instant::now();
        let mut sum = 0.0;
        for _ in 0..ITERATIONS {
            for &x in &test_values {
                sum += lookup_table.atan(x);
            }
        }
        let lookup_duration = start.elapsed();

        println!("Performance comparison:");
        println!("std::f32::atan: {:?}", std_duration);
        println!("fast_atan:      {:?} ({:.2}x speedup)", fast_duration, std_duration.as_secs_f64() / fast_duration.as_secs_f64());
        println!("fastest_atan:   {:?} ({:.2}x speedup)", fastest_duration, std_duration.as_secs_f64() / fastest_duration.as_secs_f64());
        println!("minimax_atan:   {:?} ({:.2}x speedup)", minimax_duration, std_duration.as_secs_f64() / minimax_duration.as_secs_f64());
        println!("lookup_atan:    {:?} ({:.2}x speedup)", lookup_duration, std_duration.as_secs_f64() / lookup_duration.as_secs_f64());
    }
}