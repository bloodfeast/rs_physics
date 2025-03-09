use crate::particles::particle_interactions_barnes_hut_cosmological::{ApproxNode, compute_forces_scalar, Particle, ParticleCollection, Quad};
use crate::utils::{fast_sqrt, fast_sqrt_f64};

/// SIMD-optimized force calculation for a batch of nodes using AVX-512
#[target_feature(enable = "avx512f")]
#[target_feature(enable = "avx512vl")]
pub unsafe fn compute_forces_simd_avx512(
    p: &Particle,
    nodes: &[ApproxNode],
    g: f64,
    time: f64
) -> (f64, f64) {
    use std::arch::x86_64::*;

    let mut total_fx = 0.0;
    let mut total_fy = 0.0;
    let n = nodes.len();
    let mut i = 0;

    // Process 8 nodes at a time using AVX-512
    while i + 8 <= n {
        // Load particle data
        let p_x = _mm512_set1_pd(p.position.0);
        let p_y = _mm512_set1_pd(p.position.1);
        let p_mass = _mm512_set1_pd(p.mass);
        let g_val = _mm512_set1_pd(g);
        let softening = _mm512_set1_pd(1e-4);
        let time_val = _mm512_set1_pd(time);
        let rotation_factor = _mm512_set1_pd(0.01);

        // Load node data
        let mut masses = [0.0; 8];
        let mut com_xs = [0.0; 8];
        let mut com_ys = [0.0; 8];
        let mut spins = [0.0; 8];

        for j in 0..8 {
            masses[j] = nodes[i + j].mass;
            com_xs[j] = nodes[i + j].com_x;
            com_ys[j] = nodes[i + j].com_y;
            spins[j] = nodes[i + j].spin;
        }

        let node_masses = _mm512_loadu_pd(masses.as_ptr());
        let node_com_xs = _mm512_loadu_pd(com_xs.as_ptr());
        let node_com_ys = _mm512_loadu_pd(com_ys.as_ptr());
        let node_spins = _mm512_loadu_pd(spins.as_ptr());

        // Calculate displacement vectors
        let dx = _mm512_sub_pd(node_com_xs, p_x);
        let dy = _mm512_sub_pd(node_com_ys, p_y);

        // Calculate distance squared
        let dx2 = _mm512_mul_pd(dx, dx);
        let dy2 = _mm512_mul_pd(dy, dy);
        let dist_sq = _mm512_add_pd(_mm512_add_pd(dx2, dy2), softening);

        // Calculate inverse distance for normalization
        let dist = _mm512_sqrt_pd(dist_sq);
        let inv_dist = _mm512_div_pd(_mm512_set1_pd(1.0), dist);

        // Calculate basic gravitational force
        let m1m2 = _mm512_mul_pd(p_mass, node_masses);
        let force_magnitudes = _mm512_div_pd(_mm512_mul_pd(g_val, m1m2), dist_sq);

        // Calculate force components
        let fx_grav = _mm512_mul_pd(_mm512_mul_pd(force_magnitudes, dx), inv_dist);
        let fy_grav = _mm512_mul_pd(_mm512_mul_pd(force_magnitudes, dy), inv_dist);

        // Calculate rotational effects
        let spin_strength = _mm512_mul_pd(node_spins, rotation_factor);
        let neg_one = _mm512_set1_pd(-1.0);
        let fx_rot = _mm512_mul_pd(_mm512_mul_pd(_mm512_mul_pd(dy, spin_strength), neg_one), inv_dist);
        let fy_rot = _mm512_mul_pd(_mm512_mul_pd(dx, spin_strength), inv_dist);

        // Calculate Hubble flow
        let hubble_base = _mm512_set1_pd(70.0);
        let expansion_factor = _mm512_set1_pd(0.1);
        let hubble_denom = _mm512_add_pd(_mm512_set1_pd(1.0), _mm512_mul_pd(time_val, expansion_factor));
        let hubble_param = _mm512_div_pd(hubble_base, hubble_denom);
        let expansion_scale = _mm512_mul_pd(hubble_param, _mm512_set1_pd(1e-6));

        let fx_exp = _mm512_mul_pd(dx, expansion_scale);
        let fy_exp = _mm512_mul_pd(dy, expansion_scale);

        // Combine all forces
        let fx_total = _mm512_add_pd(_mm512_add_pd(fx_grav, fx_rot), fx_exp);
        let fy_total = _mm512_add_pd(_mm512_add_pd(fy_grav, fy_rot), fy_exp);

        // Sum results horizontally - AVX-512 specific horizontal add
        total_fx += _mm512_reduce_add_pd(fx_total);
        total_fy += _mm512_reduce_add_pd(fy_total);

        i += 8;
    }

    // Handle remaining nodes with AVX2 if available (4 at a time)
    #[cfg(target_feature = "avx")]
    {
        while i + 4 <= n {
            // Load particle data
            let p_x = _mm256_set1_pd(p.position.0);
            let p_y = _mm256_set1_pd(p.position.1);
            let p_mass = _mm256_set1_pd(p.mass);
            let g_val = _mm256_set1_pd(g);
            let softening = _mm256_set1_pd(1e-4);
            let time_val = _mm256_set1_pd(time);
            let rotation_factor = _mm256_set1_pd(0.01);

            // Load node data
            let mut masses = [0.0; 4];
            let mut com_xs = [0.0; 4];
            let mut com_ys = [0.0; 4];
            let mut spins = [0.0; 4];

            for j in 0..4 {
                masses[j] = nodes[i + j].mass;
                com_xs[j] = nodes[i + j].com_x;
                com_ys[j] = nodes[i + j].com_y;
                spins[j] = nodes[i + j].spin;
            }

            let node_masses = _mm256_loadu_pd(masses.as_ptr());
            let node_com_xs = _mm256_loadu_pd(com_xs.as_ptr());
            let node_com_ys = _mm256_loadu_pd(com_ys.as_ptr());
            let node_spins = _mm256_loadu_pd(spins.as_ptr());

            // Calculate displacement vectors
            let dx = _mm256_sub_pd(node_com_xs, p_x);
            let dy = _mm256_sub_pd(node_com_ys, p_y);

            // Calculate distance squared
            let dx2 = _mm256_mul_pd(dx, dx);
            let dy2 = _mm256_mul_pd(dy, dy);
            let dist_sq = _mm256_add_pd(_mm256_add_pd(dx2, dy2), softening);

            // Calculate inverse distance for normalization
            let dist = _mm256_sqrt_pd(dist_sq);
            let inv_dist = _mm256_div_pd(_mm256_set1_pd(1.0), dist);

            // Calculate basic gravitational force
            let m1m2 = _mm256_mul_pd(p_mass, node_masses);
            let force_magnitudes = _mm256_div_pd(_mm256_mul_pd(g_val, m1m2), dist_sq);

            // Calculate force components
            let fx_grav = _mm256_mul_pd(_mm256_mul_pd(force_magnitudes, dx), inv_dist);
            let fy_grav = _mm256_mul_pd(_mm256_mul_pd(force_magnitudes, dy), inv_dist);

            // Calculate rotational effects
            let spin_strength = _mm256_mul_pd(node_spins, rotation_factor);
            let fx_rot = _mm256_mul_pd(_mm256_mul_pd(_mm256_mul_pd(dy, spin_strength), _mm256_set1_pd(-1.0)), inv_dist);
            let fy_rot = _mm256_mul_pd(_mm256_mul_pd(dx, spin_strength), inv_dist);

            // Calculate Hubble flow
            let hubble_base = _mm256_set1_pd(70.0);
            let expansion_factor = _mm256_set1_pd(0.1);
            let hubble_denom = _mm256_add_pd(_mm256_set1_pd(1.0), _mm256_mul_pd(time_val, expansion_factor));
            let hubble_param = _mm256_div_pd(hubble_base, hubble_denom);
            let expansion_scale = _mm256_mul_pd(hubble_param, _mm256_set1_pd(1e-6));

            let fx_exp = _mm256_mul_pd(dx, expansion_scale);
            let fy_exp = _mm256_mul_pd(dy, expansion_scale);

            // Combine all forces
            let fx_total = _mm256_add_pd(_mm256_add_pd(fx_grav, fx_rot), fx_exp);
            let fy_total = _mm256_add_pd(_mm256_add_pd(fy_grav, fy_rot), fy_exp);

            // Sum results horizontally
            let mut fx_arr = [0.0; 4];
            let mut fy_arr = [0.0; 4];
            _mm256_storeu_pd(fx_arr.as_mut_ptr(), fx_total);
            _mm256_storeu_pd(fy_arr.as_mut_ptr(), fy_total);

            for j in 0..4 {
                total_fx += fx_arr[j];
                total_fy += fy_arr[j];
            }

            i += 4;
        }
    }

    // Handle remaining nodes using scalar code
    for j in i..n {
        let node = &nodes[j];
        let dx = node.com_x - p.position.0;
        let dy = node.com_y - p.position.1;

        // Use a softening parameter to avoid numerical instability
        let softening = 1e-4;
        let dist_sq = dx * dx + dy * dy + softening;
        let dist = fast_sqrt_f64(dist_sq);

        // Basic gravitational force
        let basic_force = g * p.mass * node.mass / dist_sq;
        let force_x = basic_force * dx / dist;
        let force_y = basic_force * dy / dist;

        // Rotational effect
        let rotation_strength = node.spin * 0.01;
        let tangential_x = -dy / dist * rotation_strength;
        let tangential_y = dx / dist * rotation_strength;

        // Expansion term
        let hubble_param = 70.0 * (1.0 / (1.0 + time * 0.1));
        let expansion_scale = hubble_param * 1e-6;
        let expansion_x = dx * expansion_scale;
        let expansion_y = dy * expansion_scale;

        total_fx += force_x + tangential_x + expansion_x;
        total_fy += force_y + tangential_y + expansion_y;
    }

    (total_fx, total_fy)
}

/// Try to use AVX-512 if available with 32-bit single precision for even greater throughput
/// Process 16 particles at once
#[target_feature(enable = "avx512f")]
#[target_feature(enable = "avx512vl")]
pub unsafe fn compute_forces_simd_avx512_f32(
    p: &Particle,
    nodes: &[ApproxNode],
    g: f64,
    time: f64
) -> (f64, f64) {
    use std::arch::x86_64::*;

    let mut total_fx = 0.0;
    let mut total_fy = 0.0;
    let n = nodes.len();
    let mut i = 0;

    // Convert inputs to f32 for faster processing
    let p_x_f32 = p.position.0 as f32;
    let p_y_f32 = p.position.1 as f32;
    let p_mass_f32 = p.mass as f32;
    let g_f32 = g as f32;
    let time_f32 = time as f32;

    // Process 16 nodes at a time using AVX-512 with f32
    while i + 16 <= n {
        // Load particle data
        let p_x = _mm512_set1_ps(p_x_f32);
        let p_y = _mm512_set1_ps(p_y_f32);
        let p_mass = _mm512_set1_ps(p_mass_f32);
        let g_val = _mm512_set1_ps(g_f32);
        let softening = _mm512_set1_ps(1e-4);
        let time_val = _mm512_set1_ps(time_f32);
        let rotation_factor = _mm512_set1_ps(0.01);

        // Load node data
        let mut masses = [0.0f32; 16];
        let mut com_xs = [0.0f32; 16];
        let mut com_ys = [0.0f32; 16];
        let mut spins = [0.0f32; 16];

        for j in 0..16 {
            masses[j] = nodes[i + j].mass as f32;
            com_xs[j] = nodes[i + j].com_x as f32;
            com_ys[j] = nodes[i + j].com_y as f32;
            spins[j] = nodes[i + j].spin as f32;
        }

        let node_masses = _mm512_loadu_ps(masses.as_ptr());
        let node_com_xs = _mm512_loadu_ps(com_xs.as_ptr());
        let node_com_ys = _mm512_loadu_ps(com_ys.as_ptr());
        let node_spins = _mm512_loadu_ps(spins.as_ptr());

        // Calculate displacement vectors
        let dx = _mm512_sub_ps(node_com_xs, p_x);
        let dy = _mm512_sub_ps(node_com_ys, p_y);

        // Calculate distance squared
        let dx2 = _mm512_mul_ps(dx, dx);
        let dy2 = _mm512_mul_ps(dy, dy);
        let dist_sq = _mm512_add_ps(_mm512_add_ps(dx2, dy2), softening);

        // Calculate inverse distance for normalization
        let dist = _mm512_sqrt_ps(dist_sq);
        let inv_dist = _mm512_div_ps(_mm512_set1_ps(1.0), dist);

        // Calculate basic gravitational force
        let m1m2 = _mm512_mul_ps(p_mass, node_masses);
        let force_magnitudes = _mm512_div_ps(_mm512_mul_ps(g_val, m1m2), dist_sq);

        // Calculate force components
        let fx_grav = _mm512_mul_ps(_mm512_mul_ps(force_magnitudes, dx), inv_dist);
        let fy_grav = _mm512_mul_ps(_mm512_mul_ps(force_magnitudes, dy), inv_dist);

        // Calculate rotational effects
        let spin_strength = _mm512_mul_ps(node_spins, rotation_factor);
        let neg_one = _mm512_set1_ps(-1.0);
        let fx_rot = _mm512_mul_ps(_mm512_mul_ps(_mm512_mul_ps(dy, spin_strength), neg_one), inv_dist);
        let fy_rot = _mm512_mul_ps(_mm512_mul_ps(dx, spin_strength), inv_dist);

        // Calculate Hubble flow
        let hubble_base = _mm512_set1_ps(70.0);
        let expansion_factor = _mm512_set1_ps(0.1);
        let hubble_denom = _mm512_add_ps(_mm512_set1_ps(1.0), _mm512_mul_ps(time_val, expansion_factor));
        let hubble_param = _mm512_div_ps(hubble_base, hubble_denom);
        let expansion_scale = _mm512_mul_ps(hubble_param, _mm512_set1_ps(1e-6));

        let fx_exp = _mm512_mul_ps(dx, expansion_scale);
        let fy_exp = _mm512_mul_ps(dy, expansion_scale);

        // Combine all forces
        let fx_total = _mm512_add_ps(_mm512_add_ps(fx_grav, fx_rot), fx_exp);
        let fy_total = _mm512_add_ps(_mm512_add_ps(fy_grav, fy_rot), fy_exp);

        // Sum results horizontally - AVX-512 specific horizontal add for floats
        total_fx += _mm512_reduce_add_ps(fx_total) as f64;
        total_fy += _mm512_reduce_add_ps(fy_total) as f64;

        i += 16;
    }

    // Handle remaining nodes with AVX-512 double precision
    if i + 8 <= n {
        let fx_fy = compute_forces_simd_avx512(&Particle {
            position: p.position,
            velocity: p.velocity.clone(),
            mass: p.mass,
            spin: p.spin,
            age: p.age,
            density: p.density,
        }, &nodes[i..i+8], g, time);

        total_fx += fx_fy.0;
        total_fy += fx_fy.1;
        i += 8;
    }

    // Handle remaining nodes with AVX2 or scalar
    if i < n {
        // Use existing scalar implementation for the rest
        let fx_fy = compute_forces_scalar(p, &nodes[i..], g, time);
        total_fx += fx_fy.0;
        total_fy += fx_fy.1;
    }

    (total_fx, total_fy)
}

/// SIMD-optimized force calculation for a batch of nodes using AVX2
#[target_feature(enable = "avx2")]
pub unsafe fn compute_forces_simd_avx2(
    p: &Particle,
    nodes: &[ApproxNode],
    g: f64,
    time: f64
) -> (f64, f64) {
    use std::arch::x86_64::*;

    let mut total_fx = 0.0;
    let mut total_fy = 0.0;
    let n = nodes.len();
    let mut i = 0;

    // Process 4 nodes at a time using AVX2
    while i + 4 <= n {
        // Load particle data
        let p_x = _mm256_set1_pd(p.position.0);
        let p_y = _mm256_set1_pd(p.position.1);
        let p_mass = _mm256_set1_pd(p.mass);
        let g_val = _mm256_set1_pd(g);
        let softening = _mm256_set1_pd(1e-4);
        let time_val = _mm256_set1_pd(time);
        let rotation_factor = _mm256_set1_pd(0.01);

        // Load node data
        let mut masses = [0.0; 4];
        let mut com_xs = [0.0; 4];
        let mut com_ys = [0.0; 4];
        let mut spins = [0.0; 4];

        for j in 0..4 {
            masses[j] = nodes[i + j].mass;
            com_xs[j] = nodes[i + j].com_x;
            com_ys[j] = nodes[i + j].com_y;
            spins[j] = nodes[i + j].spin;
        }

        let node_masses = _mm256_loadu_pd(masses.as_ptr());
        let node_com_xs = _mm256_loadu_pd(com_xs.as_ptr());
        let node_com_ys = _mm256_loadu_pd(com_ys.as_ptr());
        let node_spins = _mm256_loadu_pd(spins.as_ptr());

        // Calculate displacement vectors
        let dx = _mm256_sub_pd(node_com_xs, p_x);
        let dy = _mm256_sub_pd(node_com_ys, p_y);

        // Calculate distance squared
        let dx2 = _mm256_mul_pd(dx, dx);
        let dy2 = _mm256_mul_pd(dy, dy);
        let dist_sq = _mm256_add_pd(_mm256_add_pd(dx2, dy2), softening);

        // Calculate inverse distance for normalization
        let dist = _mm256_sqrt_pd(dist_sq);
        let inv_dist = _mm256_div_pd(_mm256_set1_pd(1.0), dist);

        // Calculate basic gravitational force
        let m1m2 = _mm256_mul_pd(p_mass, node_masses);
        let force_magnitudes = _mm256_div_pd(_mm256_mul_pd(g_val, m1m2), dist_sq);

        // Calculate force components
        let fx_grav = _mm256_mul_pd(_mm256_mul_pd(force_magnitudes, dx), inv_dist);
        let fy_grav = _mm256_mul_pd(_mm256_mul_pd(force_magnitudes, dy), inv_dist);

        // Calculate rotational effects
        let spin_strength = _mm256_mul_pd(node_spins, rotation_factor);
        let fx_rot = _mm256_mul_pd(_mm256_mul_pd(_mm256_mul_pd(dy, spin_strength), _mm256_set1_pd(-1.0)), inv_dist);
        let fy_rot = _mm256_mul_pd(_mm256_mul_pd(dx, spin_strength), inv_dist);

        // Calculate Hubble flow
        let hubble_base = _mm256_set1_pd(70.0);
        let expansion_factor = _mm256_set1_pd(0.1);
        let hubble_denom = _mm256_add_pd(_mm256_set1_pd(1.0), _mm256_mul_pd(time_val, expansion_factor));
        let hubble_param = _mm256_div_pd(hubble_base, hubble_denom);
        let expansion_scale = _mm256_mul_pd(hubble_param, _mm256_set1_pd(1e-6));

        let fx_exp = _mm256_mul_pd(dx, expansion_scale);
        let fy_exp = _mm256_mul_pd(dy, expansion_scale);

        // Combine all forces
        let fx_total = _mm256_add_pd(_mm256_add_pd(fx_grav, fx_rot), fx_exp);
        let fy_total = _mm256_add_pd(_mm256_add_pd(fy_grav, fy_rot), fy_exp);

        // Sum results horizontally
        let mut fx_arr = [0.0; 4];
        let mut fy_arr = [0.0; 4];
        _mm256_storeu_pd(fx_arr.as_mut_ptr(), fx_total);
        _mm256_storeu_pd(fy_arr.as_mut_ptr(), fy_total);

        for j in 0..4 {
            total_fx += fx_arr[j];
            total_fy += fy_arr[j];
        }

        i += 4;
    }

    // Handle remaining nodes using scalar code
    for j in i..n {
        let node = &nodes[j];
        let dx = node.com_x - p.position.0;
        let dy = node.com_y - p.position.1;

        // Use a softening parameter to avoid numerical instability
        let softening = 1e-4;
        let dist_sq = dx * dx + dy * dy + softening;
        let dist = fast_sqrt_f64(dist_sq);

        // Basic gravitational force
        let basic_force = g * p.mass * node.mass / dist_sq;
        let force_x = basic_force * dx / dist;
        let force_y = basic_force * dy / dist;

        // Rotational effect
        let rotation_strength = node.spin * 0.01;
        let tangential_x = -dy / dist * rotation_strength;
        let tangential_y = dx / dist * rotation_strength;

        // Expansion term
        let hubble_param = 70.0 * (1.0 / (1.0 + time * 0.1));
        let expansion_scale = hubble_param * 1e-6;
        let expansion_x = dx * expansion_scale;
        let expansion_y = dy * expansion_scale;

        total_fx += force_x + tangential_x + expansion_x;
        total_fy += force_y + tangential_y + expansion_y;
    }

    (total_fx, total_fy)
}

/// SIMD-optimized force calculation for SoA particle data using SSE4.1
#[target_feature(enable = "sse4.1")]
pub unsafe fn compute_forces_simd_soa_sse41(
    particles: &ParticleCollection,
    index: usize,
    nodes: &[ApproxNode],
    g: f32,
    time: f32
) -> (f32, f32) {
    use std::arch::x86_64::*;

    let mut total_fx = 0.0;
    let mut total_fy = 0.0;
    let n = nodes.len();
    let mut i = 0;

    // Get particle data
    let p_x = particles.positions_x[index];
    let p_y = particles.positions_y[index];
    let p_mass = particles.masses[index];

    // Process 4 nodes at a time using SSE4.1 (for f32)
    while i + 4 <= n {
        // Load particle data into SIMD registers
        let p_x_v = _mm_set1_ps(p_x);
        let p_y_v = _mm_set1_ps(p_y);
        let p_mass_v = _mm_set1_ps(p_mass);
        let g_v = _mm_set1_ps(g);
        let softening = _mm_set1_ps(1e-4);
        let rotation_factor = _mm_set1_ps(0.01);

        // Load node data - 4 at a time
        let mut masses = [0.0f32; 4];
        let mut com_xs = [0.0f32; 4];
        let mut com_ys = [0.0f32; 4];
        let mut spins = [0.0f32; 4];

        for j in 0..4 {
            masses[j] = nodes[i + j].mass as f32;
            com_xs[j] = nodes[i + j].com_x as f32;
            com_ys[j] = nodes[i + j].com_y as f32;
            spins[j] = nodes[i + j].spin as f32;
        }

        let node_masses = _mm_loadu_ps(masses.as_ptr());
        let node_com_xs = _mm_loadu_ps(com_xs.as_ptr());
        let node_com_ys = _mm_loadu_ps(com_ys.as_ptr());
        let node_spins = _mm_loadu_ps(spins.as_ptr());

        // Calculate displacement vectors
        let dx = _mm_sub_ps(node_com_xs, p_x_v);
        let dy = _mm_sub_ps(node_com_ys, p_y_v);

        // Calculate distance squared
        let dx2 = _mm_mul_ps(dx, dx);
        let dy2 = _mm_mul_ps(dy, dy);
        let dist_sq = _mm_add_ps(_mm_add_ps(dx2, dy2), softening);

        // Distance and inverse distance
        let dist = _mm_sqrt_ps(dist_sq);
        let inv_dist = _mm_div_ps(_mm_set1_ps(1.0), dist);

        // Calculate basic gravitational force
        let m1m2 = _mm_mul_ps(p_mass_v, node_masses);
        let force_magnitudes = _mm_div_ps(_mm_mul_ps(g_v, m1m2), dist_sq);

        // Force components
        let fx_grav = _mm_mul_ps(_mm_mul_ps(force_magnitudes, dx), inv_dist);
        let fy_grav = _mm_mul_ps(_mm_mul_ps(force_magnitudes, dy), inv_dist);

        // Rotational effects
        let spin_strength = _mm_mul_ps(node_spins, rotation_factor);
        let neg_one = _mm_set1_ps(-1.0);
        let fx_rot = _mm_mul_ps(_mm_mul_ps(_mm_mul_ps(dy, spin_strength), neg_one), inv_dist);
        let fy_rot = _mm_mul_ps(_mm_mul_ps(dx, spin_strength), inv_dist);

        // Combine forces
        let fx_total = _mm_add_ps(fx_grav, fx_rot);
        let fy_total = _mm_add_ps(fy_grav, fy_rot);

        // Sum results using SSE4.1 horizontal add
        let mut fx_arr = [0.0f32; 4];
        let mut fy_arr = [0.0f32; 4];
        _mm_storeu_ps(fx_arr.as_mut_ptr(), fx_total);
        _mm_storeu_ps(fy_arr.as_mut_ptr(), fy_total);

        // Sum the values manually
        for j in 0..4 {
            total_fx += fx_arr[j];
            total_fy += fy_arr[j];
        }

        i += 4;
    }

    // Handle remaining nodes
    for j in i..n {
        let node = &nodes[j];
        let dx = node.com_x as f32 - p_x;
        let dy = node.com_y as f32 - p_y;
        let dist_sq = dx * dx + dy * dy + 1e-4;
        let dist = dist_sq.sqrt();
        let force = g * p_mass * node.mass as f32 / dist_sq;
        total_fx += force * dx / dist;
        total_fy += force * dy / dist;
    }

    (total_fx, total_fy)
}

/// SIMD-optimized force calculation for a batch of nodes using SSE4.1
#[cfg(target_feature = "sse4.1")]
pub unsafe fn compute_forces_simd_sse41(
    p: &Particle,
    nodes: &[ApproxNode],
    g: f64,
    time: f64
) -> (f64, f64) {
    use std::arch::x86_64::*;

    let mut total_fx = 0.0;
    let mut total_fy = 0.0;
    let n = nodes.len();
    let mut i = 0;

    // Process 2 nodes at a time using SSE4.1
    while i + 2 <= n {
        // Load particle data
        let p_x = _mm_set1_pd(p.position.0);
        let p_y = _mm_set1_pd(p.position.1);
        let p_mass = _mm_set1_pd(p.mass);
        let g_val = _mm_set1_pd(g);
        let softening = _mm_set1_pd(1e-4);
        let time_val = _mm_set1_pd(time);
        let rotation_factor = _mm_set1_pd(0.01);

        // Load node data
        let mut masses = [0.0; 2];
        let mut com_xs = [0.0; 2];
        let mut com_ys = [0.0; 2];
        let mut spins = [0.0; 2];

        for j in 0..2 {
            masses[j] = nodes[i + j].mass;
            com_xs[j] = nodes[i + j].com_x;
            com_ys[j] = nodes[i + j].com_y;
            spins[j] = nodes[i + j].spin;
        }

        let node_masses = _mm_loadu_pd(masses.as_ptr());
        let node_com_xs = _mm_loadu_pd(com_xs.as_ptr());
        let node_com_ys = _mm_loadu_pd(com_ys.as_ptr());
        let node_spins = _mm_loadu_pd(spins.as_ptr());

        // Calculate displacement vectors
        let dx = _mm_sub_pd(node_com_xs, p_x);
        let dy = _mm_sub_pd(node_com_ys, p_y);

        // Calculate distance squared
        let dx2 = _mm_mul_pd(dx, dx);
        let dy2 = _mm_mul_pd(dy, dy);
        let dist_sq = _mm_add_pd(_mm_add_pd(dx2, dy2), softening);

        // Calculate inverse distance for normalization
        let dist = _mm_sqrt_pd(dist_sq);
        let inv_dist = _mm_div_pd(_mm_set1_pd(1.0), dist);

        // Calculate basic gravitational force
        let m1m2 = _mm_mul_pd(p_mass, node_masses);
        let force_magnitudes = _mm_div_pd(_mm_mul_pd(g_val, m1m2), dist_sq);

        // Calculate force components
        let fx_grav = _mm_mul_pd(_mm_mul_pd(force_magnitudes, dx), inv_dist);
        let fy_grav = _mm_mul_pd(_mm_mul_pd(force_magnitudes, dy), inv_dist);

        // Calculate rotational effects
        let spin_strength = _mm_mul_pd(node_spins, rotation_factor);
        let fx_rot = _mm_mul_pd(_mm_mul_pd(_mm_mul_pd(dy, spin_strength), _mm_set1_pd(-1.0)), inv_dist);
        let fy_rot = _mm_mul_pd(_mm_mul_pd(dx, spin_strength), inv_dist);

        // Calculate Hubble flow
        let hubble_base = _mm_set1_pd(70.0);
        let expansion_factor = _mm_set1_pd(0.1);
        let hubble_denom = _mm_add_pd(_mm_set1_pd(1.0), _mm_mul_pd(time_val, expansion_factor));
        let hubble_param = _mm_div_pd(hubble_base, hubble_denom);
        let expansion_scale = _mm_mul_pd(hubble_param, _mm_set1_pd(1e-6));

        let fx_exp = _mm_mul_pd(dx, expansion_scale);
        let fy_exp = _mm_mul_pd(dy, expansion_scale);

        // Combine all forces
        let fx_total = _mm_add_pd(_mm_add_pd(fx_grav, fx_rot), fx_exp);
        let fy_total = _mm_add_pd(_mm_add_pd(fy_grav, fy_rot), fy_exp);

        // Sum results horizontally
        let mut fx_arr = [0.0; 2];
        let mut fy_arr = [0.0; 2];
        _mm_storeu_pd(fx_arr.as_mut_ptr(), fx_total);
        _mm_storeu_pd(fy_arr.as_mut_ptr(), fy_total);

        for j in 0..2 {
            total_fx += fx_arr[j];
            total_fy += fy_arr[j];
        }

        i += 2;
    }

    // Handle remaining nodes using scalar code
    for j in i..n {
        let node = &nodes[j];
        let dx = node.com_x - p.position.0;
        let dy = node.com_y - p.position.1;

        // Use a softening parameter to avoid numerical instability
        let softening = 1e-4;
        let dist_sq = dx * dx + dy * dy + softening;
        let dist = fast_sqrt_f64(dist_sq);

        // Basic gravitational force
        let basic_force = g * p.mass * node.mass / dist_sq;
        let force_x = basic_force * dx / dist;
        let force_y = basic_force * dy / dist;

        // Rotational effect
        let rotation_strength = node.spin * 0.01;
        let tangential_x = -dy / dist * rotation_strength;
        let tangential_y = dx / dist * rotation_strength;

        // Expansion term
        let hubble_param = 70.0 * (1.0 / (1.0 + time * 0.1));
        let expansion_scale = hubble_param * 1e-6;
        let expansion_x = dx * expansion_scale;
        let expansion_y = dy * expansion_scale;

        total_fx += force_x + tangential_x + expansion_x;
        total_fy += force_y + tangential_y + expansion_y;
    }

    (total_fx, total_fy)
}

/// SIMD-optimized force calculation for SoA particle data using AVX-512
#[target_feature(enable = "avx512f")]
pub unsafe fn compute_forces_simd_soa_avx512(
    particles: &ParticleCollection,
    index: usize,
    nodes: &[ApproxNode],
    g: f32,
    time: f32
) -> (f32, f32) {
    use std::arch::x86_64::*;

    let mut total_fx = 0.0;
    let mut total_fy = 0.0;
    let n = nodes.len();
    let mut i = 0;

    // Get particle data
    let p_x = particles.positions_x[index];
    let p_y = particles.positions_y[index];
    let p_mass = particles.masses[index];

    // Process 16 nodes at a time using AVX-512 (for f32)
    while i + 16 <= n {
        // Load particle data into SIMD registers
        let p_x_v = _mm512_set1_ps(p_x);
        let p_y_v = _mm512_set1_ps(p_y);
        let p_mass_v = _mm512_set1_ps(p_mass);
        let g_v = _mm512_set1_ps(g);
        let softening = _mm512_set1_ps(1e-4);
        let time_v = _mm512_set1_ps(time);
        let rotation_factor = _mm512_set1_ps(0.01);

        // Load node data - 16 at a time
        let mut masses = [0.0f32; 16];
        let mut com_xs = [0.0f32; 16];
        let mut com_ys = [0.0f32; 16];
        let mut spins = [0.0f32; 16];

        for j in 0..16 {
            masses[j] = nodes[i + j].mass as f32;
            com_xs[j] = nodes[i + j].com_x as f32;
            com_ys[j] = nodes[i + j].com_y as f32;
            spins[j] = nodes[i + j].spin as f32;
        }

        let node_masses = _mm512_loadu_ps(masses.as_ptr());
        let node_com_xs = _mm512_loadu_ps(com_xs.as_ptr());
        let node_com_ys = _mm512_loadu_ps(com_ys.as_ptr());
        let node_spins = _mm512_loadu_ps(spins.as_ptr());

        // Calculate displacement vectors
        let dx = _mm512_sub_ps(node_com_xs, p_x_v);
        let dy = _mm512_sub_ps(node_com_ys, p_y_v);

        // Calculate distance squared
        let dx2 = _mm512_mul_ps(dx, dx);
        let dy2 = _mm512_mul_ps(dy, dy);
        let dist_sq = _mm512_add_ps(_mm512_add_ps(dx2, dy2), softening);

        // Distance and inverse distance
        let dist = _mm512_sqrt_ps(dist_sq);
        let inv_dist = _mm512_div_ps(_mm512_set1_ps(1.0), dist);

        // Calculate basic gravitational force
        let m1m2 = _mm512_mul_ps(p_mass_v, node_masses);
        let force_magnitudes = _mm512_div_ps(_mm512_mul_ps(g_v, m1m2), dist_sq);

        // Force components
        let fx_grav = _mm512_mul_ps(_mm512_mul_ps(force_magnitudes, dx), inv_dist);
        let fy_grav = _mm512_mul_ps(_mm512_mul_ps(force_magnitudes, dy), inv_dist);

        // Rotational effects
        let spin_strength = _mm512_mul_ps(node_spins, rotation_factor);
        let neg_one = _mm512_set1_ps(-1.0);
        let fx_rot = _mm512_mul_ps(_mm512_mul_ps(_mm512_mul_ps(dy, spin_strength), neg_one), inv_dist);
        let fy_rot = _mm512_mul_ps(_mm512_mul_ps(dx, spin_strength), inv_dist);

        // Calculate Hubble flow
        let hubble_base = _mm512_set1_ps(70.0);
        let expansion_factor = _mm512_set1_ps(0.1);
        let hubble_denom = _mm512_add_ps(_mm512_set1_ps(1.0), _mm512_mul_ps(time_v, expansion_factor));
        let hubble_param = _mm512_div_ps(hubble_base, hubble_denom);
        let expansion_scale = _mm512_mul_ps(hubble_param, _mm512_set1_ps(1e-6));

        let fx_exp = _mm512_mul_ps(dx, expansion_scale);
        let fy_exp = _mm512_mul_ps(dy, expansion_scale);

        // Combine forces
        let fx_total = _mm512_add_ps(_mm512_add_ps(fx_grav, fx_rot), fx_exp);
        let fy_total = _mm512_add_ps(_mm512_add_ps(fy_grav, fy_rot), fy_exp);

        // Sum results - AVX-512 has direct reduction operations
        total_fx += _mm512_reduce_add_ps(fx_total);
        total_fy += _mm512_reduce_add_ps(fy_total);

        i += 16;
    }

    // Process any remaining nodes with AVX2 if available
    if i + 8 <= n {
        // Call AVX2 implementation for remaining 8-15 nodes
        let (fx8, fy8) = compute_forces_simd_soa_avx2_partial(
            p_x, p_y, p_mass,
            &nodes[i..i+8],
            g, time
        );
        total_fx += fx8;
        total_fy += fy8;
        i += 8;
    }

    // Handle remaining nodes
    for j in i..n {
        let node = &nodes[j];
        let dx = node.com_x as f32 - p_x;
        let dy = node.com_y as f32 - p_y;
        let dist_sq = dx * dx + dy * dy + 1e-4;
        let dist = dist_sq.sqrt();
        let force = g * p_mass * node.mass as f32 / dist_sq;
        total_fx += force * dx / dist;
        total_fy += force * dy / dist;
    }

    (total_fx, total_fy)
}

// Helper function to process exactly 8 nodes with AVX2
#[target_feature(enable = "avx2")]
unsafe fn compute_forces_simd_soa_avx2_partial(
    p_x: f32,
    p_y: f32,
    p_mass: f32,
    nodes: &[ApproxNode],
    g: f32,
    time: f32
) -> (f32, f32) {
    use std::arch::x86_64::*;

    debug_assert!(nodes.len() >= 8);

    // Load particle data into SIMD registers
    let p_x_v = _mm256_set1_ps(p_x);
    let p_y_v = _mm256_set1_ps(p_y);
    let p_mass_v = _mm256_set1_ps(p_mass);
    let g_v = _mm256_set1_ps(g);
    let softening = _mm256_set1_ps(1e-4);
    let rotation_factor = _mm256_set1_ps(0.01);
    let time_v = _mm256_set1_ps(time);

    // Load node data - 8 at a time
    let mut masses = [0.0f32; 8];
    let mut com_xs = [0.0f32; 8];
    let mut com_ys = [0.0f32; 8];
    let mut spins = [0.0f32; 8];

    for j in 0..8 {
        masses[j] = nodes[j].mass as f32;
        com_xs[j] = nodes[j].com_x as f32;
        com_ys[j] = nodes[j].com_y as f32;
        spins[j] = nodes[j].spin as f32;
    }

    let node_masses = _mm256_loadu_ps(masses.as_ptr());
    let node_com_xs = _mm256_loadu_ps(com_xs.as_ptr());
    let node_com_ys = _mm256_loadu_ps(com_ys.as_ptr());
    let node_spins = _mm256_loadu_ps(spins.as_ptr());

    // Calculate displacement vectors
    let dx = _mm256_sub_ps(node_com_xs, p_x_v);
    let dy = _mm256_sub_ps(node_com_ys, p_y_v);

    // Calculate distance squared
    let dx2 = _mm256_mul_ps(dx, dx);
    let dy2 = _mm256_mul_ps(dy, dy);
    let dist_sq = _mm256_add_ps(_mm256_add_ps(dx2, dy2), softening);

    // Distance and inverse distance
    let dist = _mm256_sqrt_ps(dist_sq);
    let inv_dist = _mm256_div_ps(_mm256_set1_ps(1.0), dist);

    // Calculate basic gravitational force
    let m1m2 = _mm256_mul_ps(p_mass_v, node_masses);
    let force_magnitudes = _mm256_div_ps(_mm256_mul_ps(g_v, m1m2), dist_sq);

    // Force components
    let fx_grav = _mm256_mul_ps(_mm256_mul_ps(force_magnitudes, dx), inv_dist);
    let fy_grav = _mm256_mul_ps(_mm256_mul_ps(force_magnitudes, dy), inv_dist);

    // Rotational effects
    let spin_strength = _mm256_mul_ps(node_spins, rotation_factor);
    let fx_rot = _mm256_mul_ps(_mm256_mul_ps(_mm256_mul_ps(dy, spin_strength), _mm256_set1_ps(-1.0)), inv_dist);
    let fy_rot = _mm256_mul_ps(_mm256_mul_ps(dx, spin_strength), inv_dist);

    // Calculate Hubble flow
    let hubble_base = _mm256_set1_ps(70.0);
    let expansion_factor = _mm256_set1_ps(0.1);
    let hubble_denom = _mm256_add_ps(_mm256_set1_ps(1.0), _mm256_mul_ps(time_v, expansion_factor));
    let hubble_param = _mm256_div_ps(hubble_base, hubble_denom);
    let expansion_scale = _mm256_mul_ps(hubble_param, _mm256_set1_ps(1e-6));

    let fx_exp = _mm256_mul_ps(dx, expansion_scale);
    let fy_exp = _mm256_mul_ps(dy, expansion_scale);

    // Combine forces
    let fx_total = _mm256_add_ps(_mm256_add_ps(fx_grav, fx_rot), fx_exp);
    let fy_total = _mm256_add_ps(_mm256_add_ps(fy_grav, fy_rot), fy_exp);

    // Horizontal sum
    let mut fx_arr = [0.0f32; 8];
    let mut fy_arr = [0.0f32; 8];
    _mm256_storeu_ps(fx_arr.as_mut_ptr(), fx_total);
    _mm256_storeu_ps(fy_arr.as_mut_ptr(), fy_total);

    let mut sum_fx = 0.0;
    let mut sum_fy = 0.0;
    for j in 0..8 {
        sum_fx += fx_arr[j];
        sum_fy += fy_arr[j];
    }

    (sum_fx, sum_fy)
}

/// SIMD-optimized force calculation for SoA particle data
#[target_feature(enable = "avx2")]
pub unsafe fn compute_forces_simd_soa_avx2(
    particles: &ParticleCollection,
    index: usize,
    nodes: &[ApproxNode],
    g: f32,
    time: f32
) -> (f32, f32) {
    use std::arch::x86_64::*;

    let mut total_fx = 0.0;
    let mut total_fy = 0.0;
    let n = nodes.len();
    let mut i = 0;

    // Get particle data
    let p_x = particles.positions_x[index];
    let p_y = particles.positions_y[index];
    let p_mass = particles.masses[index];

    // Process 8 nodes at a time using AVX2 (for f32)
    while i + 8 <= n {
        // Load particle data into SIMD registers
        let p_x_v = _mm256_set1_ps(p_x);
        let p_y_v = _mm256_set1_ps(p_y);
        let p_mass_v = _mm256_set1_ps(p_mass);
        let g_v = _mm256_set1_ps(g);
        let softening = _mm256_set1_ps(1e-4);
        let time_v = _mm256_set1_ps(time);
        let rotation_factor = _mm256_set1_ps(0.01);

        // Load node data - 8 at a time
        let mut masses = [0.0f32; 8];
        let mut com_xs = [0.0f32; 8];
        let mut com_ys = [0.0f32; 8];
        let mut spins = [0.0f32; 8];

        for j in 0..8 {
            masses[j] = nodes[i + j].mass as f32;
            com_xs[j] = nodes[i + j].com_x as f32;
            com_ys[j] = nodes[i + j].com_y as f32;
            spins[j] = nodes[i + j].spin as f32;
        }

        let node_masses = _mm256_loadu_ps(masses.as_ptr());
        let node_com_xs = _mm256_loadu_ps(com_xs.as_ptr());
        let node_com_ys = _mm256_loadu_ps(com_ys.as_ptr());
        let node_spins = _mm256_loadu_ps(spins.as_ptr());

        // Calculate displacement vectors
        let dx = _mm256_sub_ps(node_com_xs, p_x_v);
        let dy = _mm256_sub_ps(node_com_ys, p_y_v);

        // Calculate distance squared
        let dx2 = _mm256_mul_ps(dx, dx);
        let dy2 = _mm256_mul_ps(dy, dy);
        let dist_sq = _mm256_add_ps(_mm256_add_ps(dx2, dy2), softening);

        // Distance and inverse distance
        let dist = _mm256_sqrt_ps(dist_sq);
        let inv_dist = _mm256_div_ps(_mm256_set1_ps(1.0), dist);

        // Calculate basic gravitational force
        let m1m2 = _mm256_mul_ps(p_mass_v, node_masses);
        let force_magnitudes = _mm256_div_ps(_mm256_mul_ps(g_v, m1m2), dist_sq);

        // Force components
        let fx_grav = _mm256_mul_ps(_mm256_mul_ps(force_magnitudes, dx), inv_dist);
        let fy_grav = _mm256_mul_ps(_mm256_mul_ps(force_magnitudes, dy), inv_dist);

        // Rotational effects
        let spin_strength = _mm256_mul_ps(node_spins, rotation_factor);
        let fx_rot = _mm256_mul_ps(_mm256_mul_ps(_mm256_mul_ps(dy, spin_strength), _mm256_set1_ps(-1.0)), inv_dist);
        let fy_rot = _mm256_mul_ps(_mm256_mul_ps(dx, spin_strength), inv_dist);

        // Calculate Hubble flow
        let hubble_base = _mm256_set1_ps(70.0);
        let expansion_factor = _mm256_set1_ps(0.1);
        let hubble_denom = _mm256_add_ps(_mm256_set1_ps(1.0), _mm256_mul_ps(time_v, expansion_factor));
        let hubble_param = _mm256_div_ps(hubble_base, hubble_denom);
        let expansion_scale = _mm256_mul_ps(hubble_param, _mm256_set1_ps(1e-6));

        let fx_exp = _mm256_mul_ps(dx, expansion_scale);
        let fy_exp = _mm256_mul_ps(dy, expansion_scale);

        // Combine forces
        let fx_total = _mm256_add_ps(_mm256_add_ps(fx_grav, fx_rot), fx_exp);
        let fy_total = _mm256_add_ps(_mm256_add_ps(fy_grav, fy_rot), fy_exp);

        // Sum results
        let mut fx_arr = [0.0f32; 8];
        let mut fy_arr = [0.0f32; 8];
        _mm256_storeu_ps(fx_arr.as_mut_ptr(), fx_total);
        _mm256_storeu_ps(fy_arr.as_mut_ptr(), fy_total);

        for j in 0..8 {
            total_fx += fx_arr[j];
            total_fy += fy_arr[j];
        }

        i += 8;
    }

    // Handle remaining nodes
    for j in i..n {
        let node = &nodes[j];
        let dx = node.com_x as f32 - p_x;
        let dy = node.com_y as f32 - p_y;
        let dist_sq = dx * dx + dy * dy + 1e-4;
        let dist = fast_sqrt(dist_sq);
        let force = g * p_mass * node.mass as f32 / dist_sq;
        total_fx += force * dx / dist;
        total_fy += force * dy / dist;
    }

    (total_fx, total_fy)
}

/// Update velocities in batches using AVX-512
#[target_feature(enable = "avx512f")]
pub unsafe fn update_velocities_simd_avx512(
    particles: &mut ParticleCollection,
    forces_x: &[f32],
    forces_y: &[f32],
    dt: f32
) {
    use std::arch::x86_64::*;

    let n = particles.count;
    let mut i = 0;
    let dt_v = _mm512_set1_ps(dt);

    // Process 16 particles at a time
    while i + 16 <= n {
        // Load masses and calculate inverse mass
        let masses = _mm512_loadu_ps(&particles.masses[i]);
        let inv_masses = _mm512_div_ps(_mm512_set1_ps(1.0), masses);

        // Load forces
        let fx = _mm512_loadu_ps(&forces_x[i]);
        let fy = _mm512_loadu_ps(&forces_y[i]);

        // Calculate acceleration
        let ax = _mm512_mul_ps(fx, inv_masses);
        let ay = _mm512_mul_ps(fy, inv_masses);

        // Load current velocities
        let vx = _mm512_loadu_ps(&particles.velocities_x[i]);
        let vy = _mm512_loadu_ps(&particles.velocities_y[i]);

        // Update velocities: v = v + a*dt
        let new_vx = _mm512_add_ps(vx, _mm512_mul_ps(ax, dt_v));
        let new_vy = _mm512_add_ps(vy, _mm512_mul_ps(ay, dt_v));

        // Store updated velocities
        _mm512_storeu_ps(&mut particles.velocities_x[i], new_vx);
        _mm512_storeu_ps(&mut particles.velocities_y[i], new_vy);

        i += 16;
    }

    // Process remaining particles with AVX2 if at least 8 are left
    if i + 8 <= n {
        update_velocities_simd_avx2_partial(
            &mut particles.velocities_x[i..],
            &mut particles.velocities_y[i..],
            &particles.masses[i..],
            &forces_x[i..],
            &forces_y[i..],
            dt
        );
        i += 8;
    }

    // Handle remaining particles
    for j in i..n {
        let inv_mass = 1.0 / particles.masses[j];
        particles.velocities_x[j] += forces_x[j] * inv_mass * dt;
        particles.velocities_y[j] += forces_y[j] * inv_mass * dt;
    }
}

/// Helper for remaining 8-15 particles using AVX2
#[target_feature(enable = "avx2")]
unsafe fn update_velocities_simd_avx2_partial(
    velocities_x: &mut [f32],
    velocities_y: &mut [f32],
    masses: &[f32],
    forces_x: &[f32],
    forces_y: &[f32],
    dt: f32
) {
    use std::arch::x86_64::*;

    debug_assert!(velocities_x.len() >= 8);
    debug_assert!(velocities_y.len() >= 8);
    debug_assert!(masses.len() >= 8);
    debug_assert!(forces_x.len() >= 8);
    debug_assert!(forces_y.len() >= 8);

    let dt_v = _mm256_set1_ps(dt);

    // Load masses and calculate inverse mass
    let masses_v = _mm256_loadu_ps(masses.as_ptr());
    let inv_masses = _mm256_div_ps(_mm256_set1_ps(1.0), masses_v);

    // Load forces
    let fx = _mm256_loadu_ps(forces_x.as_ptr());
    let fy = _mm256_loadu_ps(forces_y.as_ptr());

    // Calculate acceleration
    let ax = _mm256_mul_ps(fx, inv_masses);
    let ay = _mm256_mul_ps(fy, inv_masses);

    // Load current velocities
    let vx = _mm256_loadu_ps(velocities_x.as_ptr());
    let vy = _mm256_loadu_ps(velocities_y.as_ptr());

    // Update velocities: v = v + a*dt
    let new_vx = _mm256_add_ps(vx, _mm256_mul_ps(ax, dt_v));
    let new_vy = _mm256_add_ps(vy, _mm256_mul_ps(ay, dt_v));

    // Store updated velocities
    _mm256_storeu_ps(velocities_x.as_mut_ptr(), new_vx);
    _mm256_storeu_ps(velocities_y.as_mut_ptr(), new_vy);
}

/// Process multiple particles in a batch with SIMD
#[target_feature(enable = "avx2")]
pub unsafe fn update_velocities_simd_avx2(
    particles: &mut ParticleCollection,
    forces_x: &[f32],
    forces_y: &[f32],
    dt: f32
) {
    use std::arch::x86_64::*;

    let n = particles.count;
    let mut i = 0;
    let dt_v = _mm256_set1_ps(dt);

    // Process 8 particles at a time
    while i + 8 <= n {
        // Load masses and calculate inverse mass
        let masses = _mm256_loadu_ps(&particles.masses[i]);
        let inv_masses = _mm256_div_ps(_mm256_set1_ps(1.0), masses);

        // Load forces
        let fx = _mm256_loadu_ps(&forces_x[i]);
        let fy = _mm256_loadu_ps(&forces_y[i]);

        // Calculate acceleration
        let ax = _mm256_mul_ps(fx, inv_masses);
        let ay = _mm256_mul_ps(fy, inv_masses);

        // Load current velocities
        let vx = _mm256_loadu_ps(&particles.velocities_x[i]);
        let vy = _mm256_loadu_ps(&particles.velocities_y[i]);

        // Update velocities: v = v + a*dt
        let new_vx = _mm256_add_ps(vx, _mm256_mul_ps(ax, dt_v));
        let new_vy = _mm256_add_ps(vy, _mm256_mul_ps(ay, dt_v));

        // Store updated velocities
        _mm256_storeu_ps(&mut particles.velocities_x[i], new_vx);
        _mm256_storeu_ps(&mut particles.velocities_y[i], new_vy);

        i += 8;
    }

    // Handle remaining particles
    for j in i..n {
        let inv_mass = 1.0 / particles.masses[j];
        particles.velocities_x[j] += forces_x[j] * inv_mass * dt;
        particles.velocities_y[j] += forces_y[j] * inv_mass * dt;
    }
}

/// Update velocities in batches using SSE4.1
#[target_feature(enable = "sse4.1")]
pub unsafe fn update_velocities_simd_sse41(
    particles: &mut ParticleCollection,
    forces_x: &[f32],
    forces_y: &[f32],
    dt: f32
) {
    use std::arch::x86_64::*;

    let n = particles.count;
    let mut i = 0;
    let dt_v = _mm_set1_ps(dt);

    // Process 4 particles at a time
    while i + 4 <= n {
        // Load masses and calculate inverse mass
        let masses = _mm_loadu_ps(&particles.masses[i]);
        let inv_masses = _mm_div_ps(_mm_set1_ps(1.0), masses);

        // Load forces
        let fx = _mm_loadu_ps(&forces_x[i]);
        let fy = _mm_loadu_ps(&forces_y[i]);

        // Calculate acceleration
        let ax = _mm_mul_ps(fx, inv_masses);
        let ay = _mm_mul_ps(fy, inv_masses);

        // Load current velocities
        let vx = _mm_loadu_ps(&particles.velocities_x[i]);
        let vy = _mm_loadu_ps(&particles.velocities_y[i]);

        // Update velocities: v = v + a*dt
        let new_vx = _mm_add_ps(vx, _mm_mul_ps(ax, dt_v));
        let new_vy = _mm_add_ps(vy, _mm_mul_ps(ay, dt_v));

        // Store updated velocities
        _mm_storeu_ps(&mut particles.velocities_x[i], new_vx);
        _mm_storeu_ps(&mut particles.velocities_y[i], new_vy);

        i += 4;
    }

    // Handle remaining particles
    for j in i..n {
        let inv_mass = 1.0 / particles.masses[j];
        particles.velocities_x[j] += forces_x[j] * inv_mass * dt;
        particles.velocities_y[j] += forces_y[j] * inv_mass * dt;
    }
}

/// Update positions in batches using AVX-512
#[target_feature(enable = "avx512f")]
pub unsafe fn update_positions_simd_avx512(
    particles: &mut ParticleCollection,
    dt: f32
) {
    use std::arch::x86_64::*;

    let n = particles.count;
    let mut i = 0;
    let dt_v = _mm512_set1_ps(dt);

    // Process 16 particles at a time
    while i + 16 <= n {
        // Load current positions and velocities
        let px = _mm512_loadu_ps(&particles.positions_x[i]);
        let py = _mm512_loadu_ps(&particles.positions_y[i]);
        let vx = _mm512_loadu_ps(&particles.velocities_x[i]);
        let vy = _mm512_loadu_ps(&particles.velocities_y[i]);

        // Update positions: p = p + v*dt
        let new_px = _mm512_add_ps(px, _mm512_mul_ps(vx, dt_v));
        let new_py = _mm512_add_ps(py, _mm512_mul_ps(vy, dt_v));

        // Store updated positions
        _mm512_storeu_ps(&mut particles.positions_x[i], new_px);
        _mm512_storeu_ps(&mut particles.positions_y[i], new_py);

        // Update ages
        let ages = _mm512_loadu_ps(&particles.ages[i]);
        let new_ages = _mm512_add_ps(ages, dt_v);
        _mm512_storeu_ps(&mut particles.ages[i], new_ages);

        i += 16;
    }

    // Process remaining particles with AVX2 if at least 8 are left
    if i + 8 <= n {
        update_positions_simd_avx2_partial(
            &mut particles.positions_x[i..],
            &mut particles.positions_y[i..],
            &mut particles.ages[i..],
            &particles.velocities_x[i..],
            &particles.velocities_y[i..],
            dt
        );
        i += 8;
    }

    // Handle remaining particles
    for j in i..n {
        particles.positions_x[j] += particles.velocities_x[j] * dt;
        particles.positions_y[j] += particles.velocities_y[j] * dt;
        particles.ages[j] += dt;
    }
}

/// Helper for remaining 8-15 particles using AVX2
#[target_feature(enable = "avx2")]
unsafe fn update_positions_simd_avx2_partial(
    positions_x: &mut [f32],
    positions_y: &mut [f32],
    ages: &mut [f32],
    velocities_x: &[f32],
    velocities_y: &[f32],
    dt: f32
) {
    use std::arch::x86_64::*;

    debug_assert!(positions_x.len() >= 8);
    debug_assert!(positions_y.len() >= 8);
    debug_assert!(ages.len() >= 8);
    debug_assert!(velocities_x.len() >= 8);
    debug_assert!(velocities_y.len() >= 8);

    let dt_v = _mm256_set1_ps(dt);

    // Load current positions and velocities
    let px = _mm256_loadu_ps(positions_x.as_ptr());
    let py = _mm256_loadu_ps(positions_y.as_ptr());
    let vx = _mm256_loadu_ps(velocities_x.as_ptr());
    let vy = _mm256_loadu_ps(velocities_y.as_ptr());

    // Update positions: p = p + v*dt
    let new_px = _mm256_add_ps(px, _mm256_mul_ps(vx, dt_v));
    let new_py = _mm256_add_ps(py, _mm256_mul_ps(vy, dt_v));

    // Store updated positions
    _mm256_storeu_ps(positions_x.as_mut_ptr(), new_px);
    _mm256_storeu_ps(positions_y.as_mut_ptr(), new_py);

    // Update ages
    let ages_v = _mm256_loadu_ps(ages.as_ptr());
    let new_ages = _mm256_add_ps(ages_v, dt_v);
    _mm256_storeu_ps(ages.as_mut_ptr(), new_ages);
}

/// Update positions in batches using SIMD
#[target_feature(enable = "avx2")]
pub unsafe fn update_positions_simd_avx2(
    particles: &mut ParticleCollection,
    dt: f32
) {
    use std::arch::x86_64::*;

    let n = particles.count;
    let mut i = 0;
    let dt_v = _mm256_set1_ps(dt);

    // Process 8 particles at a time
    while i + 8 <= n {
        // Load current positions and velocities
        let px = _mm256_loadu_ps(&particles.positions_x[i]);
        let py = _mm256_loadu_ps(&particles.positions_y[i]);
        let vx = _mm256_loadu_ps(&particles.velocities_x[i]);
        let vy = _mm256_loadu_ps(&particles.velocities_y[i]);

        // Update positions: p = p + v*dt
        let new_px = _mm256_add_ps(px, _mm256_mul_ps(vx, dt_v));
        let new_py = _mm256_add_ps(py, _mm256_mul_ps(vy, dt_v));

        // Store updated positions
        _mm256_storeu_ps(&mut particles.positions_x[i], new_px);
        _mm256_storeu_ps(&mut particles.positions_y[i], new_py);

        // Update ages
        let ages = _mm256_loadu_ps(&particles.ages[i]);
        let new_ages = _mm256_add_ps(ages, dt_v);
        _mm256_storeu_ps(&mut particles.ages[i], new_ages);

        i += 8;
    }

    // Handle remaining particles
    for j in i..n {
        particles.positions_x[j] += particles.velocities_x[j] * dt;
        particles.positions_y[j] += particles.velocities_y[j] * dt;
        particles.ages[j] += dt;
    }
}

/// Update positions in batches using SSE4.1
#[target_feature(enable = "sse4.1")]
pub unsafe fn update_positions_simd_sse41(
    particles: &mut ParticleCollection,
    dt: f32
) {
    use std::arch::x86_64::*;

    let n = particles.count;
    let mut i = 0;
    let dt_v = _mm_set1_ps(dt);

    // Process 4 particles at a time
    while i + 4 <= n {
        // Load current positions and velocities
        let px = _mm_loadu_ps(&particles.positions_x[i]);
        let py = _mm_loadu_ps(&particles.positions_y[i]);
        let vx = _mm_loadu_ps(&particles.velocities_x[i]);
        let vy = _mm_loadu_ps(&particles.velocities_y[i]);

        // Update positions: p = p + v*dt
        let new_px = _mm_add_ps(px, _mm_mul_ps(vx, dt_v));
        let new_py = _mm_add_ps(py, _mm_mul_ps(vy, dt_v));

        // Store updated positions
        _mm_storeu_ps(&mut particles.positions_x[i], new_px);
        _mm_storeu_ps(&mut particles.positions_y[i], new_py);

        // Update ages
        let ages = _mm_loadu_ps(&particles.ages[i]);
        let new_ages = _mm_add_ps(ages, dt_v);
        _mm_storeu_ps(&mut particles.ages[i], new_ages);

        i += 4;
    }

    // Handle remaining particles
    for j in i..n {
        particles.positions_x[j] += particles.velocities_x[j] * dt;
        particles.positions_y[j] += particles.velocities_y[j] * dt;
        particles.ages[j] += dt;
    }
}

/// Compute center of mass for a node with AVX-512 acceleration
#[target_feature(enable = "avx512f")]
pub unsafe fn compute_center_of_mass_simd_avx512(
    particles: &ParticleCollection,
    indices: &[usize]
) -> (f32, f32, f32) { // Returns (mass, com_x, com_y)
    use std::arch::x86_64::*;

    let n = indices.len();
    let mut i = 0;

    // SIMD accumulators
    let mut total_mass_v = _mm512_setzero_ps();
    let mut weighted_x_v = _mm512_setzero_ps();
    let mut weighted_y_v = _mm512_setzero_ps();

    // Process 16 particles at a time
    while i + 16 <= n {
        // Load data for 16 particles
        let mut positions_x = [0.0f32; 16];
        let mut positions_y = [0.0f32; 16];
        let mut masses = [0.0f32; 16];

        for j in 0..16 {
            let idx = indices[i + j];
            positions_x[j] = particles.positions_x[idx];
            positions_y[j] = particles.positions_y[idx];
            masses[j] = particles.masses[idx];
        }

        let px = _mm512_loadu_ps(positions_x.as_ptr());
        let py = _mm512_loadu_ps(positions_y.as_ptr());
        let m = _mm512_loadu_ps(masses.as_ptr());

        // Calculate weighted positions
        let wx = _mm512_mul_ps(px, m);
        let wy = _mm512_mul_ps(py, m);

        // Accumulate
        total_mass_v = _mm512_add_ps(total_mass_v, m);
        weighted_x_v = _mm512_add_ps(weighted_x_v, wx);
        weighted_y_v = _mm512_add_ps(weighted_y_v, wy);

        i += 16;
    }

    // Direct reduction with AVX-512
    let mut total_mass = _mm512_reduce_add_ps(total_mass_v);
    let mut total_wx = _mm512_reduce_add_ps(weighted_x_v);
    let mut total_wy = _mm512_reduce_add_ps(weighted_y_v);

    // Process remaining particles
    for j in i..n {
        let idx = indices[j];
        let mass = particles.masses[idx];
        total_mass += mass;
        total_wx += particles.positions_x[idx] * mass;
        total_wy += particles.positions_y[idx] * mass;
    }

    // Calculate center of mass
    let com_x = if total_mass > 0.0 { total_wx / total_mass } else { 0.0 };
    let com_y = if total_mass > 0.0 { total_wy / total_mass } else { 0.0 };

    (total_mass, com_x, com_y)
}

/// Batch testing to identify which particles are inside a quadrant - AVX-512 version
#[target_feature(enable = "avx512f")]
pub unsafe fn batch_test_particles_in_quadrant_avx512(
    particles: &ParticleCollection,
    indices: &[usize],
    quad: &Quad,
    result_mask: &mut [bool]
) {
    use std::arch::x86_64::*;

    let n = indices.len();
    debug_assert_eq!(n, result_mask.len());

    let mut i = 0;

    // Quad boundaries
    let min_x = _mm512_set1_ps((quad.cx - quad.half_size) as f32);
    let max_x = _mm512_set1_ps((quad.cx + quad.half_size) as f32);
    let min_y = _mm512_set1_ps((quad.cy - quad.half_size) as f32);
    let max_y = _mm512_set1_ps((quad.cy + quad.half_size) as f32);

    // Process 16 particles at a time
    while i + 16 <= n {
        // Load positions for 16 particles
        let mut positions_x = [0.0f32; 16];
        let mut positions_y = [0.0f32; 16];

        for j in 0..16 {
            let idx = indices[i + j];
            positions_x[j] = particles.positions_x[idx];
            positions_y[j] = particles.positions_y[idx];
        }

        let px = _mm512_loadu_ps(positions_x.as_ptr());
        let py = _mm512_loadu_ps(positions_y.as_ptr());

        // Test if inside quad boundaries
        let x_ge_min = _mm512_cmp_ps_mask::<_CMP_GE_OQ>(px, min_x);
        let x_lt_max = _mm512_cmp_ps_mask::<_CMP_LT_OQ>(px, max_x);
        let y_ge_min = _mm512_cmp_ps_mask::<_CMP_GE_OQ>(py, min_y);
        let y_lt_max = _mm512_cmp_ps_mask::<_CMP_LT_OQ>(py, max_y);

        // Combine tests
        let inside_mask = x_ge_min & x_lt_max & y_ge_min & y_lt_max;

        // Store results
        for j in 0..16 {
            result_mask[i + j] = ((inside_mask >> j) & 1) != 0;
        }

        i += 16;
    }

    // Process remaining particles
    for j in i..n {
        let idx = indices[j];
        let x = particles.positions_x[idx];
        let y = particles.positions_y[idx];

        result_mask[j] = x >= (quad.cx - quad.half_size) as f32 &&
            x < (quad.cx + quad.half_size) as f32 &&
            y >= (quad.cy - quad.half_size) as f32 &&
            y < (quad.cy + quad.half_size) as f32;
    }
}

/// Compute center of mass for a node with SIMD acceleration
#[target_feature(enable = "avx2")]
pub unsafe fn compute_center_of_mass_simd(
    particles: &ParticleCollection,
    indices: &[usize]
) -> (f32, f32, f32) { // Returns (mass, com_x, com_y)
    use std::arch::x86_64::*;

    let n = indices.len();
    let mut i = 0;

    // SIMD accumulators
    let mut total_mass_v = _mm256_setzero_ps();
    let mut weighted_x_v = _mm256_setzero_ps();
    let mut weighted_y_v = _mm256_setzero_ps();

    // Process 8 particles at a time
    while i + 8 <= n {
        // Load data for 8 particles
        let mut positions_x = [0.0f32; 8];
        let mut positions_y = [0.0f32; 8];
        let mut masses = [0.0f32; 8];

        for j in 0..8 {
            let idx = indices[i + j];
            positions_x[j] = particles.positions_x[idx];
            positions_y[j] = particles.positions_y[idx];
            masses[j] = particles.masses[idx];
        }

        let px = _mm256_loadu_ps(positions_x.as_ptr());
        let py = _mm256_loadu_ps(positions_y.as_ptr());
        let m = _mm256_loadu_ps(masses.as_ptr());

        // Calculate weighted positions
        let wx = _mm256_mul_ps(px, m);
        let wy = _mm256_mul_ps(py, m);

        // Accumulate
        total_mass_v = _mm256_add_ps(total_mass_v, m);
        weighted_x_v = _mm256_add_ps(weighted_x_v, wx);
        weighted_y_v = _mm256_add_ps(weighted_y_v, wy);

        i += 8;
    }

    // Horizontal sum of SIMD registers
    let mut mass_arr = [0.0f32; 8];
    let mut wx_arr = [0.0f32; 8];
    let mut wy_arr = [0.0f32; 8];

    _mm256_storeu_ps(mass_arr.as_mut_ptr(), total_mass_v);
    _mm256_storeu_ps(wx_arr.as_mut_ptr(), weighted_x_v);
    _mm256_storeu_ps(wy_arr.as_mut_ptr(), weighted_y_v);

    let mut total_mass = 0.0;
    let mut total_wx = 0.0;
    let mut total_wy = 0.0;

    for j in 0..8 {
        total_mass += mass_arr[j];
        total_wx += wx_arr[j];
        total_wy += wy_arr[j];
    }

    // Process remaining particles
    for j in i..n {
        let idx = indices[j];
        let mass = particles.masses[idx];
        total_mass += mass;
        total_wx += particles.positions_x[idx] * mass;
        total_wy += particles.positions_y[idx] * mass;
    }

    // Calculate center of mass
    let com_x = if total_mass > 0.0 { total_wx / total_mass } else { 0.0 };
    let com_y = if total_mass > 0.0 { total_wy / total_mass } else { 0.0 };

    (total_mass, com_x, com_y)
}

/// Compute center of mass for a node with SSE4.1 acceleration
#[target_feature(enable = "sse4.1")]
pub unsafe fn compute_center_of_mass_simd_sse41(
    particles: &ParticleCollection,
    indices: &[usize]
) -> (f32, f32, f32) { // Returns (mass, com_x, com_y)
    use std::arch::x86_64::*;

    let n = indices.len();
    let mut i = 0;

    // SIMD accumulators
    let mut total_mass_v = _mm_setzero_ps();
    let mut weighted_x_v = _mm_setzero_ps();
    let mut weighted_y_v = _mm_setzero_ps();

    // Process 4 particles at a time
    while i + 4 <= n {
        // Load data for 4 particles
        let mut positions_x = [0.0f32; 4];
        let mut positions_y = [0.0f32; 4];
        let mut masses = [0.0f32; 4];

        for j in 0..4 {
            let idx = indices[i + j];
            positions_x[j] = particles.positions_x[idx];
            positions_y[j] = particles.positions_y[idx];
            masses[j] = particles.masses[idx];
        }

        let px = _mm_loadu_ps(positions_x.as_ptr());
        let py = _mm_loadu_ps(positions_y.as_ptr());
        let m = _mm_loadu_ps(masses.as_ptr());

        // Calculate weighted positions
        let wx = _mm_mul_ps(px, m);
        let wy = _mm_mul_ps(py, m);

        // Accumulate
        total_mass_v = _mm_add_ps(total_mass_v, m);
        weighted_x_v = _mm_add_ps(weighted_x_v, wx);
        weighted_y_v = _mm_add_ps(weighted_y_v, wy);

        i += 4;
    }

    // Horizontal sum - need to extract values for SSE4.1
    let mut mass_arr = [0.0f32; 4];
    let mut wx_arr = [0.0f32; 4];
    let mut wy_arr = [0.0f32; 4];

    _mm_storeu_ps(mass_arr.as_mut_ptr(), total_mass_v);
    _mm_storeu_ps(wx_arr.as_mut_ptr(), weighted_x_v);
    _mm_storeu_ps(wy_arr.as_mut_ptr(), weighted_y_v);

    let mut total_mass = 0.0;
    let mut total_wx = 0.0;
    let mut total_wy = 0.0;

    for j in 0..4 {
        total_mass += mass_arr[j];
        total_wx += wx_arr[j];
        total_wy += wy_arr[j];
    }

    // Process remaining particles
    for j in i..n {
        let idx = indices[j];
        let mass = particles.masses[idx];
        total_mass += mass;
        total_wx += particles.positions_x[idx] * mass;
        total_wy += particles.positions_y[idx] * mass;
    }

    // Calculate center of mass
    let com_x = if total_mass > 0.0 { total_wx / total_mass } else { 0.0 };
    let com_y = if total_mass > 0.0 { total_wy / total_mass } else { 0.0 };

    (total_mass, com_x, com_y)
}

/// Batch testing to identify which particles are inside a quadrant - SSE4.1 version
#[target_feature(enable = "sse4.1")]
pub unsafe fn batch_test_particles_in_quadrant_sse41(
    particles: &ParticleCollection,
    indices: &[usize],
    quad: &Quad,
    result_mask: &mut [bool]
) {
    use std::arch::x86_64::*;

    let n = indices.len();
    debug_assert_eq!(n, result_mask.len());

    let mut i = 0;

    // Quad boundaries
    let min_x = _mm_set1_ps((quad.cx - quad.half_size) as f32);
    let max_x = _mm_set1_ps((quad.cx + quad.half_size) as f32);
    let min_y = _mm_set1_ps((quad.cy - quad.half_size) as f32);
    let max_y = _mm_set1_ps((quad.cy + quad.half_size) as f32);

    // Process 4 particles at a time
    while i + 4 <= n {
        // Load positions for 4 particles
        let mut positions_x = [0.0f32; 4];
        let mut positions_y = [0.0f32; 4];

        for j in 0..4 {
            let idx = indices[i + j];
            positions_x[j] = particles.positions_x[idx];
            positions_y[j] = particles.positions_y[idx];
        }

        let px = _mm_loadu_ps(positions_x.as_ptr());
        let py = _mm_loadu_ps(positions_y.as_ptr());

        // Test if inside quad boundaries
        // SSE4.1 doesn't have mask registers like AVX-512, so we use comparison results
        let x_ge_min = _mm_cmpge_ps(px, min_x);
        let x_lt_max = _mm_cmplt_ps(px, max_x);
        let y_ge_min = _mm_cmpge_ps(py, min_y);
        let y_lt_max = _mm_cmplt_ps(py, max_y);

        // Combine tests with AND
        let inside_x = _mm_and_ps(x_ge_min, x_lt_max);
        let inside_y = _mm_and_ps(y_ge_min, y_lt_max);
        let inside = _mm_and_ps(inside_x, inside_y);

        // Store results
        let mut mask_arr = [0; 4];
        _mm_storeu_ps(mask_arr.as_mut_ptr() as *mut f32, inside);

        for j in 0..4 {
            // In SSE, the comparison results are all 1s (true) or all 0s (false)
            result_mask[i + j] = mask_arr[j] != 0;
        }

        i += 4;
    }

    // Process remaining particles
    for j in i..n {
        let idx = indices[j];
        let x = particles.positions_x[idx];
        let y = particles.positions_y[idx];

        result_mask[j] = x >= (quad.cx - quad.half_size) as f32 &&
            x < (quad.cx + quad.half_size) as f32 &&
            y >= (quad.cy - quad.half_size) as f32 &&
            y < (quad.cy + quad.half_size) as f32;
    }
}