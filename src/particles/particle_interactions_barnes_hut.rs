/// Represents a square region in 2D space.
#[derive(Clone, Copy, Debug)]
pub struct Quad {
    pub cx: f64,        // center x-coordinate
    pub cy: f64,        // center y-coordinate
    pub half_size: f64, // half the length of one side
}

impl Quad {
    /// Returns true if the point (x, y) is inside this quad.
    /// Uses a half-open interval for the upper bound to reduce boundary ambiguity.
    ///
    /// # Example
    ///
    /// ```
    /// use rs_physics::particles::Quad;
    /// let quad = Quad { cx: 0.0, cy: 0.0, half_size: 1.0 };
    /// assert!(quad.contains(0.0, 0.0));
    /// assert!(!quad.contains(1.0, 0.0)); // 1.0 is not included
    /// ```
    pub fn contains(&self, x: f64, y: f64) -> bool {
        x >= self.cx - self.half_size &&
            x <  self.cx + self.half_size &&
            y >= self.cy - self.half_size &&
            y <  self.cy + self.half_size
    }

    /// Subdivides the quad into four smaller quads (NW, NE, SW, SE).
    ///
    /// # Example
    ///
    /// ```
    /// use rs_physics::particles::Quad;
    /// let quad = Quad { cx: 0.0, cy: 0.0, half_size: 1.0 };
    /// let (nw, ne, sw, se) = quad.subdivide();
    /// // The NW quadrant should have a center at (-0.5, 0.5)
    /// assert_eq!(nw.cx, -0.5);
    /// assert_eq!(nw.cy, 0.5);
    /// ```
    pub fn subdivide(&self) -> (Quad, Quad, Quad, Quad) {
        let hs = self.half_size / 2.0;
        (
            Quad { cx: self.cx - hs, cy: self.cy + hs, half_size: hs }, // NW
            Quad { cx: self.cx + hs, cy: self.cy + hs, half_size: hs }, // NE
            Quad { cx: self.cx - hs, cy: self.cy - hs, half_size: hs }, // SW
            Quad { cx: self.cx + hs, cy: self.cy - hs, half_size: hs }, // SE
        )
    }
}

/// A simple particle representation.
#[derive(Clone, Copy, Debug)]
pub struct ParticleData {
    pub x: f64,
    pub y: f64,
    pub mass: f64,
}

/// Barnes–Hut tree node for 2D space.
pub enum BarnesHutNode {
    /// The node is empty; it stores the quad representing its region.
    Empty(Quad),
    /// The node is a leaf and contains one particle.
    Leaf(Quad, ParticleData),
    /// The node is internal and contains aggregated data along with four children.
    Internal {
        quad: Quad,
        mass: f64,
        com_x: f64, // center of mass x
        com_y: f64, // center of mass y
        nw: Box<BarnesHutNode>,
        ne: Box<BarnesHutNode>,
        sw: Box<BarnesHutNode>,
        se: Box<BarnesHutNode>,
    },
}

impl BarnesHutNode {
    /// Creates a new empty node with the given quad.
    ///
    /// # Example
    ///
    /// ```
    /// use rs_physics::particles::{BarnesHutNode, Quad};
    /// let quad = Quad { cx: 0.0, cy: 0.0, half_size: 1.0 };
    /// let node = BarnesHutNode::new(quad);
    /// // A new node is empty.
    /// if let BarnesHutNode::Empty(q) = node {
    ///     assert_eq!(q.cx, 0.0);
    /// }
    /// ```
    pub fn new(quad: Quad) -> Self {
        BarnesHutNode::Empty(quad)
    }

    /// Inserts a particle into the Barnes–Hut tree.
    ///
    /// # Example
    ///
    /// ```
    /// use rs_physics::particles::{BarnesHutNode, Quad, ParticleData};
    /// let quad = Quad { cx: 0.0, cy: 0.0, half_size: 1.0 };
    /// let mut node = BarnesHutNode::new(quad);
    /// let particle = ParticleData { x: 0.1, y: 0.1, mass: 1.0 };
    /// node.insert(particle);
    /// // After insertion, the node is no longer empty.
    /// match node {
    ///     BarnesHutNode::Leaf(_, p) => assert_eq!(p.x, 0.1),
    ///     _ => panic!("Expected a Leaf node"),
    /// }
    /// ```
    pub fn insert(&mut self, p: ParticleData) {
        match self {
            BarnesHutNode::Empty(quad) => {
                *self = BarnesHutNode::Leaf(*quad, p);
            }
            BarnesHutNode::Leaf(quad, existing) => {
                let (nw_quad, ne_quad, sw_quad, se_quad) = quad.subdivide();
                let mut internal = BarnesHutNode::Internal {
                    quad: *quad,
                    mass: 0.0,
                    com_x: 0.0,
                    com_y: 0.0,
                    nw: Box::new(BarnesHutNode::new(nw_quad)),
                    ne: Box::new(BarnesHutNode::new(ne_quad)),
                    sw: Box::new(BarnesHutNode::new(sw_quad)),
                    se: Box::new(BarnesHutNode::new(se_quad)),
                };

                let existing_particle = *existing;
                if nw_quad.contains(existing_particle.x, existing_particle.y) {
                    internal.insert(existing_particle);
                } else if ne_quad.contains(existing_particle.x, existing_particle.y) {
                    internal.insert(existing_particle);
                } else if sw_quad.contains(existing_particle.x, existing_particle.y) {
                    internal.insert(existing_particle);
                } else if se_quad.contains(existing_particle.x, existing_particle.y) {
                    internal.insert(existing_particle);
                }
                internal.insert(p);
                *self = internal;
            }
            BarnesHutNode::Internal { quad: _, mass, com_x, com_y, nw, ne, sw, se } => {
                let total_mass = *mass + p.mass;
                *com_x = (*com_x * *mass + p.x * p.mass) / total_mass;
                *com_y = (*com_y * *mass + p.y * p.mass) / total_mass;
                *mass = total_mass;
                if nw.as_ref().quad().contains(p.x, p.y) {
                    nw.insert(p);
                } else if ne.as_ref().quad().contains(p.x, p.y) {
                    ne.insert(p);
                } else if sw.as_ref().quad().contains(p.x, p.y) {
                    sw.insert(p);
                } else if se.as_ref().quad().contains(p.x, p.y) {
                    se.insert(p);
                }
            }
        }
    }

    /// Helper method to retrieve the quad for a node.
    ///
    /// # Example
    ///
    /// ```
    /// use rs_physics::particles::{BarnesHutNode, Quad};
    /// let quad = Quad { cx: 0.0, cy: 0.0, half_size: 1.0 };
    /// let node = BarnesHutNode::new(quad);
    /// assert_eq!(node.quad().cx, 0.0);
    /// ```
    pub fn quad(&self) -> Quad {
        match self {
            BarnesHutNode::Empty(q) => *q,
            BarnesHutNode::Leaf(q, _) => *q,
            BarnesHutNode::Internal { quad, .. } => *quad,
        }
    }

    /// Computes the force exerted on particle `p` by the mass in this node.
    ///
    /// If the node is distant enough (as determined by the threshold `theta`), it uses the node’s
    /// aggregated mass and center-of-mass; otherwise, it recurses into its children.
    ///
    /// # Example
    ///
    /// ```
    /// use rs_physics::particles::{BarnesHutNode, Quad, ParticleData};
    /// let quad = Quad { cx: 0.0, cy: 0.0, half_size: 1.0 };
    /// let p1 = ParticleData { x: 0.1, y: 0.1, mass: 1.0 };
    /// let p2 = ParticleData { x: -0.1, y: -0.1, mass: 1.0 };
    /// let mut node = BarnesHutNode::new(quad);
    /// node.insert(p1);
    /// node.insert(p2);
    /// // Using a theta that forces recursion.
    /// let (fx, fy) = node.compute_force(p1, 0.3, 6.67430e-11);
    /// // Expect non-zero force (details depend on your implementation and theta).
    /// assert!(fx.abs() > 0.0 || fy.abs() > 0.0);
    /// ```
    pub fn compute_force(&self, p: ParticleData, theta: f64, g: f64) -> (f64, f64) {
        match self {
            BarnesHutNode::Empty(_) => (0.0, 0.0),
            BarnesHutNode::Leaf(_, q) => {
                if (q.x - p.x).abs() < 1e-12 && (q.y - p.y).abs() < 1e-12 {
                    (0.0, 0.0)
                } else {
                    let dx = q.x - p.x;
                    let dy = q.y - p.y;
                    let dist_sq = dx * dx + dy * dy + 1e-12;
                    let dist = dist_sq.sqrt();
                    let force = g * p.mass * q.mass / dist_sq;
                    (force * dx / dist, force * dy / dist)
                }
            }
            BarnesHutNode::Internal { quad, mass, com_x, com_y, nw, ne, sw, se } => {
                let dx = *com_x - p.x;
                let dy = *com_y - p.y;
                let dist_sq = dx * dx + dy * dy + 1e-12;
                let dist = dist_sq.sqrt();
                if (quad.half_size * 2.0 / dist) < theta {
                    let force = g * p.mass * (*mass) / dist_sq;
                    (force * dx / dist, force * dy / dist)
                } else {
                    let (fx1, fy1) = nw.compute_force(p, theta, g);
                    let (fx2, fy2) = ne.compute_force(p, theta, g);
                    let (fx3, fy3) = sw.compute_force(p, theta, g);
                    let (fx4, fy4) = se.compute_force(p, theta, g);
                    (fx1 + fx2 + fx3 + fx4, fy1 + fy2 + fy3 + fy4)
                }
            }
        }
    }
}

/// Constructs a Barnes–Hut tree from a slice of particles within a given quad.
/// Particles are partitioned into quadrants and subtrees are built in parallel.
///
/// # Example
///
/// ```
/// use rs_physics::particles::{build_tree, BarnesHutNode, Quad, ParticleData};
/// let particles = [
///     ParticleData { x: -0.5, y: -0.5, mass: 1.0 },
///     ParticleData { x: 0.5, y: 0.5, mass: 1.0 },
///     ParticleData { x: -0.5, y: 0.5, mass: 1.0 },
///     ParticleData { x: 0.5, y: -0.5, mass: 1.0 },
/// ];
/// let quad = Quad { cx: 0.0, cy: 0.0, half_size: 1.0 };
/// let tree = build_tree(&particles, quad);
/// // The resulting tree should have aggregated mass and center-of-mass computed.
/// if let BarnesHutNode::Internal { mass, com_x, com_y, .. } = tree {
///     assert!(mass >= 4.0);
/// }
/// ```
pub fn build_tree(particles: &[ParticleData], quad: Quad) -> BarnesHutNode {
    if particles.is_empty() {
        return BarnesHutNode::Empty(quad);
    }
    if particles.len() == 1 {
        return BarnesHutNode::Leaf(quad, particles[0]);
    }

    let (nw_quad, ne_quad, sw_quad, se_quad) = quad.subdivide();

    // Preallocate with estimated capacities
    let estimated_capacity = particles.len().div_ceil(4);
    let mut nw_particles = Vec::with_capacity(estimated_capacity);
    let mut ne_particles = Vec::with_capacity(estimated_capacity);
    let mut sw_particles = Vec::with_capacity(estimated_capacity);
    let mut se_particles = Vec::with_capacity(estimated_capacity);

    for &p in particles {
        if nw_quad.contains(p.x, p.y) {
            nw_particles.push(p);
        } else if ne_quad.contains(p.x, p.y) {
            ne_particles.push(p);
        } else if sw_quad.contains(p.x, p.y) {
            sw_particles.push(p);
        } else if se_quad.contains(p.x, p.y) {
            se_particles.push(p);
        }
    }

    let (nw_tree, ne_tree) = rayon::join(
        || build_tree(&nw_particles, nw_quad),
        || build_tree(&ne_particles, ne_quad)
    );
    let (sw_tree, se_tree) = rayon::join(
        || build_tree(&sw_particles, sw_quad),
        || build_tree(&se_particles, se_quad)
    );

    let mut total_mass = 0.0;
    let mut com_x = 0.0;
    let mut com_y = 0.0;
    let mut update_mass_com = |node: &BarnesHutNode| {
        if let Some((m, cx, cy)) = get_mass_com(node) {
            total_mass += m;
            com_x += cx * m;
            com_y += cy * m;
        }
    };
    update_mass_com(&nw_tree);
    update_mass_com(&ne_tree);
    update_mass_com(&sw_tree);
    update_mass_com(&se_tree);
    if total_mass > 0.0 {
        com_x /= total_mass;
        com_y /= total_mass;
    }

    BarnesHutNode::Internal {
        quad,
        mass: total_mass,
        com_x,
        com_y,
        nw: Box::new(nw_tree),
        ne: Box::new(ne_tree),
        sw: Box::new(sw_tree),
        se: Box::new(se_tree),
    }
}

/// Helper function that extracts the mass and center-of-mass from a BarnesHutNode.
/// Returns Some((mass, com_x, com_y)) for Leaf or Internal nodes, and None for Empty.
///
/// # Example
///
/// ```
/// use rs_physics::particles::{BarnesHutNode, Quad, ParticleData, get_mass_com};
/// let quad = Quad { cx: 0.0, cy: 0.0, half_size: 1.0 };
/// let particle = ParticleData { x: 0.2, y: 0.2, mass: 1.0 };
/// let node = BarnesHutNode::Leaf(quad, particle);
/// let res = get_mass_com(&node);
/// assert!(res.is_some());
/// ```
pub fn get_mass_com(node: &BarnesHutNode) -> Option<(f64, f64, f64)> {
    match node {
        BarnesHutNode::Leaf(_, p) => Some((p.mass, p.x, p.y)),
        BarnesHutNode::Internal { mass, com_x, com_y, .. } => Some((*mass, *com_x, *com_y)),
        BarnesHutNode::Empty(_) => None,
    }
}

/// Helper structure representing an approximated node.
#[derive(Debug)]
pub struct ApproxNode {
    pub mass: f64,
    pub com_x: f64, // center of mass x-coordinate
    pub com_y: f64, // center of mass y-coordinate
}

/// Recursively collects nodes that satisfy the Barnes–Hut criterion for particle `p`.
///
/// # Example
///
/// ```
/// use rs_physics::particles::{ApproxNode, BarnesHutNode, Quad, ParticleData, collect_approx_nodes, build_tree};
/// let particles = [
///     ParticleData { x: 0.1, y: 0.1, mass: 1.0 },
///     ParticleData { x: -0.1, y: -0.1, mass: 1.0 },
/// ];
/// let quad = Quad { cx: 0.0, cy: 0.0, half_size: 1.0 };
/// let tree = build_tree(&particles, quad);
/// let target = ParticleData { x: 0.0, y: 0.0, mass: 1.0 };
/// let mut worklist = Vec::new();
/// collect_approx_nodes(&tree, target, 0.5, &mut worklist);
/// assert!(!worklist.is_empty());
/// ```
pub fn collect_approx_nodes(node: &BarnesHutNode, p: ParticleData, theta: f64, worklist: &mut Vec<ApproxNode>) {
    match node {
        BarnesHutNode::Empty(_) => {},
        BarnesHutNode::Leaf(_, q) => {
            if (q.x - p.x).abs() < 1e-12 && (q.y - p.y).abs() < 1e-12 {
                return;
            }
            worklist.push(ApproxNode { mass: q.mass, com_x: q.x, com_y: q.y });
        },
        BarnesHutNode::Internal { quad, mass, com_x, com_y, nw, ne, sw, se } => {
            let dx = *com_x - p.x;
            let dy = *com_y - p.y;
            let dist = (dx * dx + dy * dy).sqrt();
            if (quad.half_size * 2.0 / dist) < theta {
                worklist.push(ApproxNode { mass: *mass, com_x: *com_x, com_y: *com_y });
            } else {
                collect_approx_nodes(nw, p, theta, worklist);
                collect_approx_nodes(ne, p, theta, worklist);
                collect_approx_nodes(sw, p, theta, worklist);
                collect_approx_nodes(se, p, theta, worklist);
            }
        }
    }
}


/// Computes net force on particle `p` from a worklist of approximated nodes using AVX intrinsics.
/// Processes the worklist in batches of 4.
///
/// # Safety
///
/// Must be called only when AVX support is available.
///
/// # Example
///
/// ```
///
///
/// # #[cfg(target_feature = "avx")]
/// {
///     use rs_physics::particles::{ApproxNode, ParticleData, compute_force_simd_avx, DGirth};
///     // This example assumes AVX is available.
///     let worklist = vec![
///         ApproxNode { mass: 1.0, com_x: 0.5, com_y: 0.5 },
///         ApproxNode { mass: 1.0, com_x: -0.5, com_y: -0.5 },
///         ApproxNode { mass: 1.0, com_x: 0.5, com_y: -0.5 },
///         ApproxNode { mass: 1.0, com_x: -0.5, com_y: 0.5 },
///     ];
///     let p = ParticleData { x: 0.0, y: 0.0, mass: 1.0 };
///     let g = 6.67430e-11;
///     unsafe {
///         let (fx, fy) = compute_force_simd_avx(p, &worklist, g);
///         assert!(fx.abs() > 0.0 || fy.abs() > 0.0);
///     }
/// }
/// ```
#[target_feature(enable = "avx")]
pub unsafe fn compute_force_simd_avx(p: ParticleData, worklist: &[ApproxNode], g: f64) -> (f64, f64) {
    use std::arch::x86_64::*;
    let mut force_x = 0.0;
    let mut force_y = 0.0;
    let n = worklist.len();
    let mut i = 0;
    while i + 4 <= n {
        let mut mass_arr = [0.0; 4];
        let mut com_x_arr = [0.0; 4];
        let mut com_y_arr = [0.0; 4];
        for j in 0..4 {
            mass_arr[j] = worklist[i + j].mass;
            com_x_arr[j] = worklist[i + j].com_x;
            com_y_arr[j] = worklist[i + j].com_y;
        }
        let mass_v = _mm256_loadu_pd(mass_arr.as_ptr());
        let com_x_v = _mm256_loadu_pd(com_x_arr.as_ptr());
        let com_y_v = _mm256_loadu_pd(com_y_arr.as_ptr());
        let p_x_v = _mm256_set1_pd(p.x);
        let p_y_v = _mm256_set1_pd(p.y);
        let dx_v = _mm256_sub_pd(com_x_v, p_x_v);
        let dy_v = _mm256_sub_pd(com_y_v, p_y_v);
        let dx2 = _mm256_mul_pd(dx_v, dx_v);
        let dy2 = _mm256_mul_pd(dy_v, dy_v);
        let sum_v = _mm256_add_pd(dx2, dy2);
        let eps = _mm256_set1_pd(1e-12);
        let dist_sq_v = _mm256_add_pd(sum_v, eps);
        let dist_v = _mm256_sqrt_pd(dist_sq_v);
        let p_mass_v = _mm256_set1_pd(p.mass);
        let g_v = _mm256_set1_pd(g);
        let numerator = _mm256_mul_pd(g_v, _mm256_mul_pd(p_mass_v, mass_v));
        let force_mag_v = _mm256_div_pd(numerator, dist_sq_v);
        let force_x_v = _mm256_div_pd(_mm256_mul_pd(force_mag_v, dx_v), dist_v);
        let force_y_v = _mm256_div_pd(_mm256_mul_pd(force_mag_v, dy_v), dist_v);

        let sum_low = _mm256_extractf128_pd::<0>(force_x_v);  // Extract lower 128 bits (2 doubles)
        let sum_high = _mm256_extractf128_pd::<1>(force_x_v); // Extract upper 128 bits (2 doubles)
        let sum = _mm_add_pd(sum_low, sum_high);            // Add them together -> (sum[0], sum[1])
        let sum_halves = _mm_hadd_pd(sum, sum);             // Horizontal add -> (sum[0]+sum[1], garbage)
        force_x += _mm_cvtsd_f64(sum_halves);

        let sum_low = _mm256_extractf128_pd::<0>(force_y_v);
        let sum_high = _mm256_extractf128_pd::<1>(force_y_v);
        let sum = _mm_add_pd(sum_low, sum_high);
        let sum_halves = _mm_hadd_pd(sum, sum);
        force_y += _mm_cvtsd_f64(sum_halves);

        i += 4;
    }
    // Process remaining nodes in scalar.
    for j in i..n {
        let dx = worklist[j].com_x - p.x;
        let dy = worklist[j].com_y - p.y;
        let dist_sq = dx * dx + dy * dy + 1e-12;
        let dist = dist_sq.sqrt();
        let force = g * p.mass * worklist[j].mass / dist_sq;
        force_x += force * dx / dist;
        force_y += force * dy / dist;
    }
    (force_x, force_y)
}
pub unsafe fn compute_force_simd_avx_low_precision(p: ParticleData, worklist: &[ApproxNode], g: f32) -> (f32, f32) {
    use std::arch::x86_64::*;
    let mut force_x = 0.0_f32;
    let mut force_y = 0.0_f32;
    let n = worklist.len();
    let mut i = 0;
    while i + 8 <= n {
        let mut mass_arr = [0.0_f32; 8];
        let mut com_x_arr = [0.0_f32; 8];
        let mut com_y_arr = [0.0_f32; 8];
        for j in 0..8 {
            mass_arr[j] = worklist[i + j].mass as f32;
            com_x_arr[j] = worklist[i + j].com_x as f32;
            com_y_arr[j] = worklist[i + j].com_y as f32;
        }
        let mass_v = _mm256_loadu_ps(mass_arr.as_ptr());
        let com_x_v = _mm256_loadu_ps(com_x_arr.as_ptr());
        let com_y_v = _mm256_loadu_ps(com_y_arr.as_ptr());
        let p_x_v = _mm256_set1_ps(p.x as f32);
        let p_y_v = _mm256_set1_ps(p.y as f32);
        let dx_v = _mm256_sub_ps(com_x_v, p_x_v);
        let dy_v = _mm256_sub_ps(com_y_v, p_y_v);
        let dx2 = _mm256_mul_ps(dx_v, dx_v);
        let dy2 = _mm256_mul_ps(dy_v, dy_v);
        let sum_v = _mm256_add_ps(dx2, dy2);
        let eps = _mm256_set1_ps(1e-12);
        let dist_sq_v = _mm256_add_ps(sum_v, eps);
        let dist_v = _mm256_sqrt_ps(dist_sq_v);
        let p_mass_v = _mm256_set1_ps(p.mass as f32);
        let g_v = _mm256_set1_ps(g);
        let numerator = _mm256_mul_ps(g_v, _mm256_mul_ps(p_mass_v, mass_v));
        let force_mag_v = _mm256_div_ps(numerator, dist_sq_v);
        let force_x_v = _mm256_div_ps(_mm256_mul_ps(force_mag_v, dx_v), dist_v);
        let force_y_v = _mm256_div_ps(_mm256_mul_ps(force_mag_v, dy_v), dist_v);

        let sum256 = _mm256_hadd_ps(force_x_v, force_x_v);  // Horizontal add pairs -> (a+b, c+d, a+b, c+d, e+f, g+h, e+f, g+h)
        let sum128 = _mm_add_ps(_mm256_extractf128_ps::<0>(sum256), _mm256_extractf128_ps::<1>(sum256));
        // Now sum128 contains (a+b+e+f, c+d+g+h, a+b+e+f, c+d+g+h)
        let sum64 = _mm_hadd_ps(sum128, sum128); // Contains (a+b+c+d+e+f+g+h, a+b+c+d+e+f+g+h, ...)
        force_x += _mm_cvtss_f32(sum64); // Extract first float (contains total sum)

        let sum256 = _mm256_hadd_ps(force_y_v, force_y_v);  // Horizontal add pairs -> (a+b, c+d, a+b, c+d, e+f, g+h, e+f, g+h)
        let sum128 = _mm_add_ps(_mm256_extractf128_ps::<0>(sum256), _mm256_extractf128_ps::<1>(sum256));
        // Now sum128 contains (a+b+e+f, c+d+g+h, a+b+e+f, c+d+g+h)
        let sum64 = _mm_hadd_ps(sum128, sum128); // Contains (a+b+c+d+e+f+g+h, a+b+c+d+e+f+g+h, ...)
        force_y += _mm_cvtss_f32(sum64); // Extract first float (contains total sum)

        i += 8;
    }
    // Process remaining nodes in scalar.
    for j in i..n {
        let dx = worklist[j].com_x as f32 - p.x as f32;
        let dy = worklist[j].com_y as f32 - p.y as f32;
        let dist_sq = dx * dx + dy * dy + 1e-12;
        let dist = dist_sq.sqrt();
        let force = g as f32 * p.mass as f32 * worklist[j].mass as f32 / dist_sq;
        force_x += force * dx / dist;
        force_y += force * dy / dist;
    }
    (force_x, force_y)
}

/// Scalar fallback function to compute net force from a worklist.
///
/// # Example
///
/// ```
/// use rs_physics::particles::{ApproxNode, ParticleData, compute_force_scalar};
/// let worklist = vec![
///     ApproxNode { mass: 1.0, com_x: 0.5, com_y: 0.0 },
///     ApproxNode { mass: 1.0, com_x: 0.6, com_y: 0.0 },
/// ];
/// let p = ParticleData { x: 0.0, y: 0.0, mass: 1.0 };
/// let g = 6.67430e-11;
/// let (fx, fy) = compute_force_scalar(p, &worklist, g);
/// assert!(fx.abs() > 0.0);
/// ```
pub fn compute_force_scalar(p: ParticleData, worklist: &[ApproxNode], g: f64) -> (f64, f64) {
    let mut force_x = 0.0;
    let mut force_y = 0.0;
    for node in worklist {
        let dx = node.com_x - p.x;
        let dy = node.com_y - p.y;
        let dist_sq = dx * dx + dy * dy + 1e-12;
        let dist = dist_sq.sqrt();
        let force = g * p.mass * node.mass / dist_sq;
        force_x += force * dx / dist;
        force_y += force * dy / dist;
    }
    (force_x, force_y)
}

/// Computes the net force on particle `p` using the Barnes–Hut tree.
/// It first collects a worklist of approximated nodes (using threshold `theta`),
/// then computes the net force using the AVX-optimized function if available,
/// or falls back to scalar computation.
///
/// # Parameters
///
/// - `tree`: The Barnes–Hut tree containing all particles.
/// - `p`: The target particle.
/// - `theta`: The Barnes–Hut threshold parameter controlling approximation accuracy.
/// - `g`: The gravitational constant.
///
/// # Returns
///
/// Returns a tuple `(force_x, force_y)` representing the net force on `p`.
///
/// # Example
///
/// ```
/// // This example assumes AVX is available.
/// use rs_physics::particles::{BarnesHutNode, ParticleData, Quad, build_tree, compute_net_force};
/// let particles = [
///     ParticleData { x: 0.1, y: 0.0, mass: 1.0 },
///     ParticleData { x: 0.2, y: 0.0, mass: 1.0 },
/// ];
/// let quad = Quad { cx: 0.0, cy: 0.0, half_size: 1.0 };
/// let tree = build_tree(&particles, quad);
/// let target = ParticleData { x: 0.0, y: 0.0, mass: 1.0 };
/// let theta = 0.5;
/// let g = 6.67430e-11;
/// let (fx, fy) = compute_net_force(&tree, target, theta, g);
/// // With both particles on the positive x-axis, net force should be positive in x.
/// assert!(fx > 0.0);
/// ```
pub fn compute_net_force(tree: &BarnesHutNode, p: ParticleData, theta: f64, g: f64) -> (f64, f64) {
    let mut worklist = Vec::new();
    collect_approx_nodes(tree, p, theta, &mut worklist);

    if std::is_x86_feature_detected!("avx") {
        if worklist.len() > 1000 {
            return unsafe {
                let res = compute_force_simd_avx_low_precision(p, &worklist, g as f32);
                (res.0 as f64, res.1 as f64)
            };
        }
        unsafe { compute_force_simd_avx(p, &worklist, g) }
    } else {
        compute_force_scalar(p, &worklist, g)
    }
}
