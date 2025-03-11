use crate::models::{Velocity2D, Direction2D};
use rayon::prelude::*;
use std::f64::consts::PI;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use std::arch::x86_64::*;
use std::sync::atomic::{AtomicPtr, Ordering};
use std::sync::atomic::Ordering::Relaxed;
use crate::particles::particle_interactions_simd_functions::{
    compute_forces_simd_avx512,
    compute_forces_simd_avx512_f32,
    compute_forces_simd_soa_avx2,
    compute_forces_simd_soa_avx512,
    compute_forces_simd_soa_sse41,
    compute_forces_simd_sse41,
    update_positions_simd_avx512,
    update_positions_simd_sse41,
    update_positions_simd_avx2,
    update_velocities_simd_avx2,
    update_velocities_simd_avx512,
    update_velocities_simd_sse41
};
use crate::utils::fast_sqrt_f64;

/// Represents a square region in 2D space.
#[derive(Clone, Copy, Debug)]
pub struct Quad {
    pub cx: f64,        // center x-coordinate
    pub cy: f64,        // center y-coordinate
    pub half_size: f64, // half the length of one side
}

impl Quad {
    /// Returns true if the point (x, y) is inside this quad.
    pub fn contains(&self, x: f64, y: f64) -> bool {
        x >= self.cx - self.half_size &&
            x <  self.cx + self.half_size &&
            y >= self.cy - self.half_size &&
            y <  self.cy + self.half_size
    }

    /// Subdivides the quad into four smaller quads (NW, NE, SW, SE).
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

pub trait SoAFromParticles {
    fn from_particle_slice(particles: &[Particle]) -> Self;
}
pub trait SoAFromParticle {
    fn from_particle(particle: &Particle, collection_size: usize) -> Self;
}

/// Memory-efficient Structure of Arrays (SoA) for particles
#[derive(Debug, Clone)]
pub struct ParticleCollection {
    pub positions_x: Vec<f32>,      // Using f32 for positions (half the memory of f64)
    pub positions_y: Vec<f32>,
    pub velocities_x: Vec<f32>,
    pub velocities_y: Vec<f32>,
    pub masses: Vec<f32>,
    pub spins: Vec<f32>,
    pub ages: Vec<f32>,
    pub densities: Vec<f32>,
    pub count: usize,
}

impl SoAFromParticle for ParticleCollection {
    fn from_particle(particle: &Particle, collection_size: usize) -> Self {
        let mut collection = Self::new(collection_size);
        collection.positions_x.push(particle.position.0 as f32);
        collection.positions_y.push(particle.position.1 as f32);
        collection.velocities_x.push(particle.velocity.x as f32);
        collection.velocities_y.push(particle.velocity.y as f32);
        collection.masses.push(particle.mass as f32);
        collection.spins.push(particle.spin as f32);
        collection.ages.push(particle.age as f32);
        collection.densities.push(particle.density as f32);
        collection.count += 1;
        collection
    }
}
impl SoAFromParticles for ParticleCollection {
    fn from_particle_slice(particles: &[Particle]) -> Self {
        let count = particles.len();
        let mut collection = Self::new(count);

        for p in particles {
            collection = ParticleCollection::from_particle(p, count);
        }
        collection
    }
}

impl ParticleCollection {
    pub fn new(capacity: usize) -> Self {
        Self {
            positions_x: Vec::with_capacity(capacity),
            positions_y: Vec::with_capacity(capacity),
            velocities_x: Vec::with_capacity(capacity),
            velocities_y: Vec::with_capacity(capacity),
            masses: Vec::with_capacity(capacity),
            spins: Vec::with_capacity(capacity),
            ages: Vec::with_capacity(capacity),
            densities: Vec::with_capacity(capacity),
            count: 0,
        }
    }

    pub fn get_particle(&self, index: usize) -> Particle {
        Particle {
            position: (self.positions_x[index] as f64, self.positions_y[index] as f64),
            velocity: Velocity2D {
                x: self.velocities_x[index] as f64,
                y: self.velocities_y[index] as f64
            },
            mass: self.masses[index] as f64,
            spin: self.spins[index] as f64,
            age: self.ages[index] as f64,
            density: self.densities[index] as f64,
        }
    }

    pub fn apply_force(&mut self, index: usize, fx: f32, fy: f32, dt: f32) {
        let inv_mass = 1.0 / self.masses[index];
        self.velocities_x[index] += fx * inv_mass * dt;
        self.velocities_y[index] += fy * inv_mass * dt;
    }

    pub fn update_position(&mut self, index: usize, dt: f32) {
        self.positions_x[index] += self.velocities_x[index] * dt;
        self.positions_y[index] += self.velocities_y[index] * dt;
        self.ages[index] += dt;
    }
}

/// Enhanced particle representation with velocity and rotational properties
#[derive(Clone, Copy, Debug)]
pub struct Particle {
    pub position: (f64, f64),     // Position (x, y)
    pub velocity: Velocity2D,     // Velocity vector
    pub mass: f64,                // Mass
    pub spin: f64,                // Angular momentum/spin
    pub age: f64,                 // Age of the particle (for visualization)
    pub density: f64,             // Local density estimation
}

impl Particle {
    /// Creates a new particle with the given properties
    pub fn new(x: f64, y: f64, vx: f64, vy: f64, mass: f64, spin: f64) -> Self {
        Self {
            position: (x, y),
            velocity: Velocity2D { x: vx, y: vy },
            mass,
            spin,
            age: 0.0,
            density: 0.0,
        }
    }

    /// Update the particle's position based on its velocity
    pub fn update_position(&mut self, dt: f64) {
        self.position.0 += self.velocity.x * dt;
        self.position.1 += self.velocity.y * dt;
        self.age += dt;
    }

    /// Apply a force to update the particle's velocity
    pub fn apply_force(&mut self, force_x: f64, force_y: f64, dt: f64) {
        // F = ma, so a = F/m
        let ax = force_x / self.mass;
        let ay = force_y / self.mass;

        // Update velocity: v = v0 + at
        self.velocity.x += ax * dt;
        self.velocity.y += ay * dt;
    }

    /// Calculate the distance to another particle
    pub fn distance_to(&self, other: &Particle) -> f64 {
        let dx = self.position.0 - other.position.0;
        let dy = self.position.1 - other.position.1;
        fast_sqrt_f64(dx * dx + dy * dy)
    }

    /// Get the direction to another particle
    pub fn direction_to(&self, other: &Particle) -> Direction2D {
        let dx = other.position.0 - self.position.0;
        let dy = other.position.1 - self.position.1;
        let magnitude = fast_sqrt_f64(dx * dx + dy * dy);

        if magnitude == 0.0 {
            Direction2D { x: 0.0, y: 0.0 }
        } else {
            Direction2D {
                x: dx / magnitude,
                y: dy / magnitude,
            }
        }
    }
}

/// Barnes–Hut tree node for 2D space with enhanced features for cosmological simulations.
pub enum BarnesHutNode {
    /// The node is empty; it stores the quad representing its region.
    Empty(Quad),

    /// The node is a leaf and contains one particle.
    Leaf(Quad, Particle),

    /// The node is internal and contains aggregated data along with four children.
    Internal {
        quad: Quad,
        mass: f64,           // Total mass of particles in this node
        com: (f64, f64),     // Center of mass (x, y)
        angular_momentum: f64, // Aggregate angular momentum
        children: [Box<BarnesHutNode>; 4], // NW, NE, SW, SE children
        num_particles: usize, // Number of particles in this node (for density calculations)
    },
}

impl BarnesHutNode {
    /// Creates a new empty node with the given quad.
    pub fn new(quad: Quad) -> Self {
        BarnesHutNode::Empty(quad)
    }

    /// Retrieves the quad for this node.
    pub fn quad(&self) -> Quad {
        match self {
            BarnesHutNode::Empty(q) => *q,
            BarnesHutNode::Leaf(q, _) => *q,
            BarnesHutNode::Internal { quad, .. } => *quad,
        }
    }

    /// Inserts a particle into the Barnes–Hut tree.
    pub fn insert(&mut self, p: Particle) {
        match self {
            BarnesHutNode::Empty(quad) => {
                *self = BarnesHutNode::Leaf(*quad, p);
            }
            BarnesHutNode::Leaf(quad, existing) => {
                // Create an internal node and insert both particles
                let (nw_quad, ne_quad, sw_quad, se_quad) = quad.subdivide();

                // Create empty children
                let children = [
                    Box::new(BarnesHutNode::Empty(nw_quad)),
                    Box::new(BarnesHutNode::Empty(ne_quad)),
                    Box::new(BarnesHutNode::Empty(sw_quad)),
                    Box::new(BarnesHutNode::Empty(se_quad)),
                ];

                // Create internal node with initial values
                let mut internal = BarnesHutNode::Internal {
                    quad: *quad,
                    mass: 0.0,
                    com: (0.0, 0.0),
                    angular_momentum: 0.0,
                    children,
                    num_particles: 0,
                };

                // Insert both particles
                let existing_particle = *existing;
                internal.insert(existing_particle);
                internal.insert(p);

                *self = internal;
            }
            BarnesHutNode::Internal { quad, mass, com, angular_momentum, children, num_particles } => {
                // Update aggregate values
                let total_mass = *mass + p.mass;
                let com_x = (com.0 * *mass + p.position.0 * p.mass) / total_mass;
                let com_y = (com.1 * *mass + p.position.1 * p.mass) / total_mass;
                *com = (com_x, com_y);
                *mass = total_mass;
                *num_particles += 1;

                // Calculate angular momentum contribution (r × p)
                let r_x = p.position.0 - com_x;
                let r_y = p.position.1 - com_y;
                let p_x = p.mass * p.velocity.x;
                let p_y = p.mass * p.velocity.y;
                let contribution = r_x * p_y - r_y * p_x + p.spin * p.mass;
                *angular_momentum += contribution;

                // Determine which child quadrant the particle belongs to
                // Don't call self.determine_child_index, calculate it directly
                let is_east = p.position.0 >= quad.cx;
                let is_north = p.position.1 >= quad.cy;

                let child_index = match (is_north, is_east) {
                    (true, false) => 0,  // NW
                    (true, true) => 1,   // NE
                    (false, false) => 2, // SW
                    (false, true) => 3,  // SE
                };

                // Insert particle into the appropriate child
                children[child_index].insert(p);
            }
        }
    }

    /// Determines which child quadrant a point belongs to
    pub(crate) fn determine_child_index(&self, x: f64, y: f64) -> usize {
        let quad = self.quad();
        let is_east = x >= quad.cx;
        let is_north = y >= quad.cy;

        match (is_north, is_east) {
            (true, false) => 0,  // NW
            (true, true) => 1,   // NE
            (false, false) => 2, // SW
            (false, true) => 3,  // SE
        }
    }

    // Compute force specifically for a particle in the SoA structure
    pub fn compute_force_soa(
        &self,
        particles: &ParticleCollection,
        index: usize,
        theta: f64,
        g: f64,
        time: f64
    ) -> (f32, f32) {
        // Create a temporary particle for force calculation
        let p = Particle {
            position: (particles.positions_x[index] as f64, particles.positions_y[index] as f64),
            velocity: Velocity2D {
                x: particles.velocities_x[index] as f64,
                y: particles.velocities_y[index] as f64
            },
            mass: particles.masses[index] as f64,
            spin: particles.spins[index] as f64,
            age: particles.ages[index] as f64,
            density: particles.densities[index] as f64,
        };

        let (fx, fy) = self.compute_force(&p, theta, g, time);
        (fx as f32, fy as f32)
    }

    /// Computes force between a particle and this node, accounting for gravitational and rotational effects
    pub fn compute_force(&self, p: &Particle, theta: f64, g: f64, time: f64) -> (f64, f64) {
        match self {
            BarnesHutNode::Empty(_) => (0.0, 0.0),

            BarnesHutNode::Leaf(_, q) => {
                // Skip self-interaction
                if (q.position.0 - p.position.0).abs() < 1e-10 &&
                    (q.position.1 - p.position.1).abs() < 1e-10 {
                    return (0.0, 0.0);
                }

                // Calculate standard gravitational force
                self.calculate_gravitational_force(p, q.position.0, q.position.1, q.mass, q.spin, theta, g, time)
            },

            BarnesHutNode::Internal { quad, mass, com, angular_momentum, children, .. } => {
                // Calculate if this node is far enough to be approximated
                let dx = com.0 - p.position.0;
                let dy = com.1 - p.position.1;
                let dist_sq = dx * dx + dy * dy;
                let dist = fast_sqrt_f64(dist_sq);

                // If node is distant enough (size/distance < theta), use approximation
                if (quad.half_size * 2.0 / dist) < theta {
                    self.calculate_gravitational_force(p, com.0, com.1, *mass, *angular_momentum, theta, g, time)
                } else {
                    // Otherwise, recursively compute forces from children
                    let forces: Vec<(f64, f64)> = children.iter()
                        .map(|child| child.compute_force(p, theta, g, time))
                        .collect();

                    // Sum up forces from all children
                    let mut total_fx = 0.0;
                    let mut total_fy = 0.0;
                    for (fx, fy) in forces {
                        total_fx += fx;
                        total_fy += fy;
                    }
                    (total_fx, total_fy)
                }
            }
        }
    }

    /// Calculate gravitational force with additional effects for cosmological simulation
    pub(crate) fn calculate_gravitational_force(&self,
                                                p: &Particle,
                                                other_x: f64,
                                                other_y: f64,
                                                other_mass: f64,
                                                other_spin: f64,
                                                theta: f64,
                                                g: f64,
                                                time: f64) -> (f64, f64) {
        let dx = other_x - p.position.0;
        let dy = other_y - p.position.1;

        // Use a softening parameter to avoid numerical instability at very close distances
        // Make the softening parameter adaptive based on theta (Barnes-Hut accuracy parameter)
        // More accurate simulations (smaller theta) use smaller softening
        let softening = 1e-4 * (theta * 10.0).max(0.1);
        let dist_sq = dx * dx + dy * dy + softening;
        let dist = fast_sqrt_f64(dist_sq);

        // Basic gravitational force: F = G * m1 * m2 / r^2
        let basic_force = g * p.mass * other_mass / dist_sq;

        // Direction of force (toward the other mass)
        let force_x = basic_force * dx / dist;
        let force_y = basic_force * dy / dist;

        // Add rotational effect based on angular momentum
        // This creates a slight tangential force component
        // Scale rotation effect based on theta - more accurate simulations get more detailed rotation
        let rotation_strength = other_spin * 0.01 * (1.0 / theta).min(10.0);
        let tangential_x = -dy / dist * rotation_strength;
        let tangential_y = dx / dist * rotation_strength;

        // Add expansion term (Hubble flow) - decreases with time to simulate slowing expansion
        let hubble_param = 70.0 * (1.0 / (1.0 + time * 0.1)); // km/s/Mpc, decreasing with time
        let expansion_scale = hubble_param * 1e-6; // Scale to appropriate units
        let expansion_x = dx * expansion_scale;
        let expansion_y = dy * expansion_scale;

        // Combine forces (gravitational + rotational + expansion)
        (
            force_x + tangential_x + expansion_x,
            force_y + tangential_y + expansion_y
        )
    }
    /// Counts the total number of particles in this node
    pub fn count_particles(&self) -> usize {
        match self {
            BarnesHutNode::Empty(_) => 0,
            BarnesHutNode::Leaf(_, _) => 1,
            BarnesHutNode::Internal { num_particles, .. } => *num_particles,
        }
    }

    /// Calculates local density estimations for all particles
    pub fn update_density_estimates(&mut self) {
        self.update_density_estimates_recursive(None);
    }

    fn update_density_estimates_recursive(&mut self, parent_stats: Option<(f64, usize)>) {
        match self {
            BarnesHutNode::Empty(_) => {},

            BarnesHutNode::Leaf(quad, particle) => {
                // Use parent statistics if available with improved weighting
                if let Some((parent_mass, parent_count)) = parent_stats {
                    let volume = (2.0 * quad.half_size) * (2.0 * quad.half_size);

                    // Weight density calculation based on mass ratio and parent count
                    if parent_count > 1 {
                        // Blend local and parent density for more accurate results
                        let parent_density = parent_mass / volume;
                        let local_density = particle.mass / volume;
                        let mass_ratio = particle.mass / parent_mass;

                        // Weighted average
                        particle.density = (parent_density * (1.0 - mass_ratio) +
                            local_density * mass_ratio * 2.0) /
                            (1.0 + mass_ratio);
                    } else {
                        // Simple case when parent has only one particle
                        particle.density = parent_mass / volume;
                    }
                } else {
                    // Leaf without parent info - use own values
                    let volume = (2.0 * quad.half_size) * (2.0 * quad.half_size);
                    particle.density = particle.mass / volume;
                }
            },

            BarnesHutNode::Internal { quad, mass, num_particles, children, .. } => {
                // Calculate this node's stats
                let volume = (2.0 * quad.half_size) * (2.0 * quad.half_size);
                let density = *mass / volume;

                // Store density in child nodes
                let stats = Some((*mass, *num_particles));

                // First, count non-empty children
                let child_count = children.iter()
                    .filter(|c| !matches!(&***c, BarnesHutNode::Empty(_)))
                    .count();

                // Then update all children
                for child in children.iter_mut() {
                    child.update_density_estimates_recursive(stats);
                }

                // After updating children, calculate density feedback
                // But don't try to match on self again
                if child_count > 0 {
                    // Capture the total child density for potential future use
                    let total_child_density: f64 = children.iter()
                        .map(|child| match &**child {
                            BarnesHutNode::Leaf(_, p) => p.density * p.mass,
                            BarnesHutNode::Internal { mass: m, .. } => density * m,
                            _ => 0.0
                        })
                        .sum();

                    // Store this for future use if needed
                    // We could add an additional field to the Internal node structure
                    // to store this value in a future version
                    let _effective_density = total_child_density / *mass;
                }
            }
        }
    }
}

pub fn update_density_estimates_iterative(root: &mut BarnesHutNode) {
    use std::collections::VecDeque;

    // For density updates, we need to use raw pointers to avoid borrow checker issues
    // with mutably borrowing multiple parts of the tree
    enum QueueItem {
        Node(*mut BarnesHutNode),
        NodeWithStats(*mut BarnesHutNode, f64, usize)  // node, parent_mass, parent_count
    }

    // Use breadth-first traversal to avoid stack overflow
    let mut queue = VecDeque::new();
    queue.push_back(QueueItem::Node(root as *mut BarnesHutNode));

    while let Some(item) = queue.pop_front() {
        // Safe to use unsafe here because we don't modify the tree structure, just update values
        unsafe {
            match item {
                QueueItem::Node(node_ptr) => {
                    match &mut *node_ptr {
                        BarnesHutNode::Empty(_) => {},

                        BarnesHutNode::Leaf(quad, particle) => {
                            // Basic leaf node case - no parent stats
                            let volume = (2.0 * quad.half_size).powi(2);
                            particle.density = particle.mass / volume;
                        },

                        BarnesHutNode::Internal { quad, mass, children, num_particles, .. } => {
                            // Calculate node stats
                            let volume = (2.0 * quad.half_size).powi(2);
                            let parent_mass = *mass;
                            let parent_count = *num_particles;

                            // Add all non-empty children to the queue with parent stats
                            for child in children.iter_mut() {
                                match &mut **child {
                                    BarnesHutNode::Empty(_) => {},
                                    _ => {
                                        let child_ptr = child.as_mut() as *mut BarnesHutNode;
                                        queue.push_back(QueueItem::NodeWithStats(
                                            child_ptr, parent_mass, parent_count
                                        ));
                                    }
                                }
                            }
                        }
                    }
                },

                QueueItem::NodeWithStats(node_ptr, parent_mass, parent_count) => {
                    match &mut *node_ptr {
                        BarnesHutNode::Empty(_) => {},

                        BarnesHutNode::Leaf(quad, particle) => {
                            // Update density using parent stats
                            let volume = (2.0 * quad.half_size).powi(2);

                            if parent_count > 1 {
                                let parent_density = parent_mass / volume;
                                let local_density = particle.mass / volume;
                                let mass_ratio = particle.mass / parent_mass;

                                // Weighted average
                                particle.density = (parent_density * (1.0 - mass_ratio) +
                                    local_density * mass_ratio * 2.0) /
                                    (1.0 + mass_ratio);
                            } else {
                                // Simple case when parent has only one particle
                                particle.density = parent_mass / volume;
                            }
                        },

                        BarnesHutNode::Internal { quad, mass, children, num_particles, .. } => {
                            // Calculate node stats, considering parent stats too
                            let volume = (2.0 * quad.half_size).powi(2);
                            let node_mass = *mass;
                            let node_count = *num_particles;

                            // Add all non-empty children to the queue with this node's stats
                            for child in children.iter_mut() {
                                match &mut **child {
                                    BarnesHutNode::Empty(_) => {},
                                    _ => {
                                        let child_ptr = child.as_mut() as *mut BarnesHutNode;
                                        queue.push_back(QueueItem::NodeWithStats(
                                            child_ptr, node_mass, node_count
                                        ));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
fn update_velocities_scalar(
    particles: &mut ParticleCollection,
    forces_x: &[f32],
    forces_y: &[f32],
    dt: f32
) {
    for i in 0..particles.count {
        let inv_mass = 1.0 / particles.masses[i];
        let fx = forces_x[i];
        let fy = forces_y[i];

        // Calculate acceleration: a = F/m
        let ax = fx * inv_mass;
        let ay = fy * inv_mass;

        // Update velocity: v = v + a*dt
        particles.velocities_x[i] += ax * dt;
        particles.velocities_y[i] += ay * dt;
    }
}

fn update_positions_scalar(
    particles: &mut ParticleCollection,
    dt: f32
) {
    for i in 0..particles.count {
        // Update position: p = p + v*dt
        particles.positions_x[i] += particles.velocities_x[i] * dt;
        particles.positions_y[i] += particles.velocities_y[i] * dt;

        // Update age
        particles.ages[i] += dt;
    }
}

/// Select the best available SIMD implementation for updating velocities
pub fn update_velocities_simd(
    particles: &mut ParticleCollection,
    forces_x: &[f32],
    forces_y: &[f32],
    dt: f32
) {
    if is_x86_feature_detected!("avx512f") {
        unsafe { update_velocities_simd_avx512(particles, forces_x, forces_y, dt) }
    } else if is_x86_feature_detected!("avx2") {
        unsafe { update_velocities_simd_avx2(particles, forces_x, forces_y, dt) }
    } else if is_x86_feature_detected!("sse4.1") {
        unsafe { update_velocities_simd_sse41(particles, forces_x, forces_y, dt) }
    } else {
        // Scalar fallback
        update_velocities_scalar(particles, forces_x, forces_y, dt)
    }
}

/// Select the best available SIMD implementation for updating positions
pub fn update_positions_simd(
    particles: &mut ParticleCollection,
    dt: f32
) {
    if is_x86_feature_detected!("avx512f") {
        unsafe { update_positions_simd_avx512(particles, dt) }
    } else if is_x86_feature_detected!("avx2") {
        unsafe { update_positions_simd_avx2(particles, dt) }
    } else if is_x86_feature_detected!("sse4.1") {
        unsafe { update_positions_simd_sse41(particles, dt) }
    } else {
        // Scalar fallback
        update_positions_scalar(particles, dt)
    }
}

/// Helper structure representing an approximated node for SIMD processing
#[derive(Debug, Clone, Copy)]
pub struct ApproxNode {
    pub mass: f64,
    pub com_x: f64,
    pub com_y: f64,
    pub spin: f64,
}

/// Collects nodes for SIMD batch processing
pub fn collect_approx_nodes(node: &BarnesHutNode, p: &Particle, theta: f64, worklist: &mut Vec<ApproxNode>) {
    match node {
        BarnesHutNode::Empty(_) => {},

        BarnesHutNode::Leaf(_, q) => {
            // Skip self-interaction
            if (q.position.0 - p.position.0).abs() < 1e-10 &&
                (q.position.1 - p.position.1).abs() < 1e-10 {
                return;
            }

            worklist.push(ApproxNode {
                mass: q.mass,
                com_x: q.position.0,
                com_y: q.position.1,
                spin: q.spin
            });
        },

        BarnesHutNode::Internal { quad, mass, com, angular_momentum, children, .. } => {
            // Check if node is far enough to be approximated
            let dx = com.0 - p.position.0;
            let dy = com.1 - p.position.1;
            let dist_sq = dx * dx + dy * dy;
            let dist = fast_sqrt_f64(dist_sq);

            if (quad.half_size * 2.0 / dist) < theta {
                // Node is far enough, use approximation
                worklist.push(ApproxNode {
                    mass: *mass,
                    com_x: com.0,
                    com_y: com.1,
                    spin: *angular_momentum,
                });
            } else {
                // Recurse into children
                for child in children.iter() {
                    collect_approx_nodes(child, p, theta, worklist);
                }
            }
        }
    }
}
pub fn collect_approx_nodes_iterative(node: &BarnesHutNode, p: &Particle, theta: f64, worklist: &mut Vec<ApproxNode>) {
    // Use a stack to replace recursion
    let mut stack = Vec::new();
    stack.push(node);

    while let Some(current) = stack.pop() {
        match current {
            BarnesHutNode::Empty(_) => {},

            BarnesHutNode::Leaf(_, q) => {
                // Skip self-interaction
                if (q.position.0 - p.position.0).abs() < 1e-10 &&
                    (q.position.1 - p.position.1).abs() < 1e-10 {
                    continue;
                }

                worklist.push(ApproxNode {
                    mass: q.mass,
                    com_x: q.position.0,
                    com_y: q.position.1,
                    spin: q.spin
                });
            },

            BarnesHutNode::Internal { quad, mass, com, angular_momentum, children, .. } => {
                // Check if node is far enough to be approximated
                let dx = com.0 - p.position.0;
                let dy = com.1 - p.position.1;
                let dist_sq = dx * dx + dy * dy;
                let dist = fast_sqrt_f64(dist_sq);

                if (quad.half_size * 2.0 / dist) < theta {
                    // Node is far enough, use approximation
                    worklist.push(ApproxNode {
                        mass: *mass,
                        com_x: com.0,
                        com_y: com.1,
                        spin: *angular_momentum,
                    });
                } else {
                    // Add children to stack in reverse order (so they're processed in the right order)
                    for child in children.iter() {
                        match **child {
                            BarnesHutNode::Empty(_) => {}, // Skip empty nodes
                            _ => stack.push(child),
                        }
                    }
                }
            }
        }
    }
}

/// Builds a Barnes-Hut tree from a slice of particles
pub fn build_tree(particles: &[Particle], bounds: Quad) -> BarnesHutNode {
    let mut root = BarnesHutNode::new(bounds);

    // Insert all particles
    for &particle in particles {
        root.insert(particle);
    }

    // Update density estimates after tree construction
    root.update_density_estimates();

    root
}
pub fn build_tree_iterative(particles: &[Particle], bounds: Quad) -> BarnesHutNode {
    let mut root = BarnesHutNode::new(bounds);

    // Process each particle
    for &particle in particles {
        // Insert particle with non-recursive method
        insert_particle_iterative(&mut root, particle);
    }

    // Update density estimates in a separate pass
    update_density_estimates_iterative(&mut root);

    root
}
pub fn build_tree_soa(particles: &ParticleCollection, bounds: Quad) -> BarnesHutNode {
    let mut root = BarnesHutNode::new(bounds);

    for i in 0..particles.count {
        // Create temporary particle for insertion
        let p = Particle {
            position: (particles.positions_x[i] as f64, particles.positions_y[i] as f64),
            velocity: Velocity2D {
                x: particles.velocities_x[i] as f64,
                y: particles.velocities_y[i] as f64
            },
            mass: particles.masses[i] as f64,
            spin: particles.spins[i] as f64,
            age: particles.ages[i] as f64,
            density: particles.densities[i] as f64,
        };

        // Use existing insert method
        root.insert(p);
    }

    // Update density estimates
    root.update_density_estimates();

    root
}

pub fn insert_particle_iterative(root: &mut BarnesHutNode, particle: Particle) {
    // Using an enum to represent operations we need to perform
    enum Operation {
        Insert { node: *mut BarnesHutNode, particle: Particle },
        UpdateInternal {
            node: *mut BarnesHutNode,
            particle: Particle,
            child_idx: usize
        },
        InsertExistingParticle {
            node: *mut BarnesHutNode,
            particle: Particle,
            child_idx: usize
        }
    }

    // Start with inserting at the root
    let mut operations = vec![Operation::Insert {
        node: root as *mut BarnesHutNode,
        particle
    }];

    // Process operations until none are left
    while let Some(op) = operations.pop() {
        match op {
            Operation::Insert { node, particle } => {
                // Need to use unsafe to work with the raw pointer
                unsafe {
                    match &mut *node {
                        BarnesHutNode::Empty(quad) => {
                            // Just replace the empty node with a leaf
                            let quad_copy = *quad;
                            *node = BarnesHutNode::Leaf(quad_copy, particle);
                        },

                        BarnesHutNode::Leaf(quad, existing) => {
                            // Need to split this leaf into an internal node
                            let quad_copy = *quad;
                            let existing_copy = *existing;

                            // Subdivide the quad
                            let (nw_quad, ne_quad, sw_quad, se_quad) = quad_copy.subdivide();

                            // Create internal node
                            let children = [
                                Box::new(BarnesHutNode::Empty(nw_quad)),
                                Box::new(BarnesHutNode::Empty(ne_quad)),
                                Box::new(BarnesHutNode::Empty(sw_quad)),
                                Box::new(BarnesHutNode::Empty(se_quad)),
                            ];

                            // Calculate center of mass and angular momentum
                            let total_mass = existing_copy.mass + particle.mass;
                            let com_x = (existing_copy.position.0 * existing_copy.mass +
                                particle.position.0 * particle.mass) / total_mass;
                            let com_y = (existing_copy.position.1 * existing_copy.mass +
                                particle.position.1 * particle.mass) / total_mass;

                            // Basic angular momentum calculation
                            let angular_momentum = existing_copy.spin * existing_copy.mass +
                                particle.spin * particle.mass;

                            // Create the internal node
                            *node = BarnesHutNode::Internal {
                                quad: quad_copy,
                                mass: total_mass,
                                com: (com_x, com_y),
                                angular_momentum,
                                children,
                                num_particles: 2,
                            };

                            // Now queue up operations to insert both particles
                            if let BarnesHutNode::Internal { ref mut children, quad, .. } = &mut *node {
                                // First, determine which child quadrant each particle belongs to
                                let existing_idx = determine_child_index_for_position(
                                    *quad, existing_copy.position.0, existing_copy.position.1);

                                let new_idx = determine_child_index_for_position(
                                    *quad, particle.position.0, particle.position.1);

                                // Queue the operations
                                operations.push(Operation::Insert {
                                    node: &mut **children.get_mut(new_idx).unwrap() as *mut BarnesHutNode,
                                    particle
                                });

                                operations.push(Operation::Insert {
                                    node: &mut **children.get_mut(existing_idx).unwrap() as *mut BarnesHutNode,
                                    particle: existing_copy
                                });
                            }
                        },

                        BarnesHutNode::Internal {
                            quad, mass, com, angular_momentum, children, num_particles
                        } => {
                            // Update the aggregate values
                            let total_mass = *mass + particle.mass;
                            let com_x = (com.0 * *mass + particle.position.0 * particle.mass) / total_mass;
                            let com_y = (com.1 * *mass + particle.position.1 * particle.mass) / total_mass;
                            *com = (com_x, com_y);
                            *mass = total_mass;
                            *num_particles += 1;

                            // Calculate angular momentum contribution
                            let r_x = particle.position.0 - com_x;
                            let r_y = particle.position.1 - com_y;
                            let p_x = particle.mass * particle.velocity.x;
                            let p_y = particle.mass * particle.velocity.y;
                            let contribution = r_x * p_y - r_y * p_x + particle.spin * particle.mass;
                            *angular_momentum += contribution;

                            // Determine the child index
                            let idx = determine_child_index_for_position(
                                *quad, particle.position.0, particle.position.1);

                            // Queue the operation to insert into the child
                            operations.push(Operation::Insert {
                                node: &mut **children.get_mut(idx).unwrap() as *mut BarnesHutNode,
                                particle
                            });
                        }
                    }
                }
            },

            // Other operations we defined but don't need in this implementation
            Operation::UpdateInternal { .. } => { },
            Operation::InsertExistingParticle { .. } => { }
        }
    }
}

/// Helper function to determine child index without self-reference
#[inline]
fn determine_child_index_for_position(quad: Quad, x: f64, y: f64) -> usize {
    let is_east = x >= quad.cx;
    let is_north = y >= quad.cy;

    match (is_north, is_east) {
        (true, false) => 0,  // NW
        (true, true) => 1,   // NE
        (false, false) => 2, // SW
        (false, true) => 3,  // SE
    }
}

/// Helper function to calculate center of mass
#[inline]
fn calculate_center_of_mass(p1: Particle, p2: Particle) -> (f64, f64) {
    let total_mass = p1.mass + p2.mass;
    let com_x = (p1.position.0 * p1.mass + p2.position.0 * p2.mass) / total_mass;
    let com_y = (p1.position.1 * p1.mass + p2.position.1 * p2.mass) / total_mass;
    (com_x, com_y)
}

/// Helper function to calculate angular momentum
#[inline]
fn calculate_angular_momentum(p1: Particle, p2: Particle) -> f64 {
    // This is a simplified calculation, a full implementation would need more context
    p1.spin * p1.mass + p2.spin * p2.mass
}


/// Scalar fallback function for force calculation
pub fn compute_forces_scalar(
    p: &Particle,
    nodes: &[ApproxNode],
    g: f64,
    time: f64
) -> (f64, f64) {
    let mut total_fx = 0.0;
    let mut total_fy = 0.0;

    for node in nodes {
        let dx = node.com_x - p.position.0;
        let dy = node.com_y - p.position.1;

        // Use a softening parameter to avoid numerical instability
        let softening = 1e-4;
        let dist_sq = dx * dx + dy * dy + softening;
        let dist = fast_sqrt_f64(dist_sq);

        // Basic gravitational force: F = G * m1 * m2 / r^2
        let basic_force = g * p.mass * node.mass / dist_sq;

        // Direction of force (toward the other mass)
        let force_x = basic_force * dx / dist;
        let force_y = basic_force * dy / dist;

        // Add rotational effect based on angular momentum
        let rotation_strength = node.spin * 0.01;
        let tangential_x = -dy / dist * rotation_strength;
        let tangential_y = dx / dist * rotation_strength;

        // Add expansion term (Hubble flow)
        let hubble_param = 70.0 * (1.0 / (1.0 + time * 0.1));
        let expansion_scale = hubble_param * 1e-6;
        let expansion_x = dx * expansion_scale;
        let expansion_y = dy * expansion_scale;

        // Combine forces
        total_fx += force_x + tangential_x + expansion_x;
        total_fy += force_y + tangential_y + expansion_y;
    }

    (total_fx, total_fy)
}

/// Scalar implementation of force calculation for SoA particle data
pub fn compute_forces_scalar_soa(
    particles: &ParticleCollection,
    index: usize,
    nodes: &[ApproxNode],
    g: f32,
    time: f32
) -> (f32, f32) {
    let mut total_fx = 0.0;
    let mut total_fy = 0.0;

    // Get particle data
    let p_x = particles.positions_x[index];
    let p_y = particles.positions_y[index];
    let p_mass = particles.masses[index];

    // Process each node individually
    for node in nodes {
        let node_x = node.com_x as f32;
        let node_y = node.com_y as f32;
        let node_mass = node.mass as f32;
        let node_spin = node.spin as f32;

        // Calculate displacement vectors
        let dx = node_x - p_x;
        let dy = node_y - p_y;

        // Use a softening parameter to avoid numerical instability
        let softening = 1e-4;
        let dist_sq = dx * dx + dy * dy + softening;
        let dist = dist_sq.sqrt();

        // Basic gravitational force: F = G * m1 * m2 / r^2
        let basic_force = g * p_mass * node_mass / dist_sq;

        // Direction of force (toward the other mass)
        let force_x = basic_force * dx / dist;
        let force_y = basic_force * dy / dist;

        // Add rotational effect based on angular momentum
        let rotation_strength = node_spin * 0.01;
        let tangential_x = -dy / dist * rotation_strength;
        let tangential_y = dx / dist * rotation_strength;

        // Add expansion term (Hubble flow)
        let hubble_param = 70.0 * (1.0 / (1.0 + time * 0.1));
        let expansion_scale = hubble_param * 1e-6;
        let expansion_x = dx * expansion_scale;
        let expansion_y = dy * expansion_scale;

        // Combine forces
        total_fx += force_x + tangential_x + expansion_x;
        total_fy += force_y + tangential_y + expansion_y;
    }

    (total_fx, total_fy)
}

/// Modified compute_net_force to use AVX-512 if available
pub fn compute_net_force(tree: &BarnesHutNode, p: &Particle, theta: f64, g: f64, time: f64) -> (f64, f64) {
    // Collect nodes that satisfy Barnes-Hut criterion into a worklist
    let mut worklist = Vec::new();
    collect_approx_nodes(tree, p, theta, &mut worklist);

    // First, check for AVX-512 support and use it for maximum performance
    #[cfg(all(target_arch = "x86_64", feature = "avx512-simd"))]
    {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512vl") {
            // For large numbers of nodes, use single precision for 2x throughput
            if worklist.len() > 1000 {
                return unsafe { compute_forces_simd_avx512_f32(p, &worklist, g, time) };
            }
            // Otherwise use double precision for accuracy
            if is_x86_feature_detected!("avx512f") {
                return unsafe { compute_forces_simd_avx512(p, &worklist, g, time) };
            }
        }
    }

    // Fall back to AVX2 if available
    #[cfg(target_feature = "avx2")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { compute_forces_simd_avx2(p, &worklist, g, time) };
        }
    }

    // Fall back to SSE4.1 if available
    #[cfg(target_feature = "sse4.1")]
    {
        if is_x86_feature_detected!("sse4.1") {
            return unsafe { compute_forces_simd_sse41(p, &worklist, g, time) };
        }
    }

    // Fallback to scalar implementation
    compute_forces_scalar(p, &worklist, g, time)
}
pub fn compute_net_force_iterative(tree: &BarnesHutNode, p: &Particle, theta: f64, g: f64, time: f64) -> (f64, f64) {
    // Use a non-recursive approach to collect nodes
    let mut worklist = Vec::new();
    collect_approx_nodes_iterative(tree, p, theta, &mut worklist);

    // Use existing SIMD or scalar force computation functions
    // (they're already non-recursive)
    #[cfg(all(target_arch = "x86_64", feature = "avx512-simd"))]
    {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512vl") {
            // For large numbers of nodes, use single precision for 2x throughput
            if worklist.len() > 1000 {
                return unsafe { compute_forces_simd_avx512_f32(p, &worklist, g, time) };
            }
            // Otherwise use double precision for accuracy
            if is_x86_feature_detected!("avx512f") {
                return unsafe { compute_forces_simd_avx512(p, &worklist, g, time) };
            }
        }
        // Fall back to AVX2 if available
        #[cfg(target_feature = "avx2")]
        {
            if is_x86_feature_detected!("avx2") {
                return unsafe { compute_forces_simd_avx2(p, &worklist, g, time) };
            }
        }

        // Fall back to SSE4.1 if available
        #[cfg(target_feature = "sse4.1")]
        {
            if is_x86_feature_detected!("sse4.1") {
                return unsafe { compute_forces_simd_sse41(p, &worklist, g, time) };
            }
        }

    }

    // Fall back to scalar implementation
    compute_forces_scalar(p, &worklist, g, time)
}
pub fn compute_forces_simd_soa(
    particles: &ParticleCollection,
    index: usize,
    nodes: &[ApproxNode],
    g: f32,
    time: f32
) -> (f32, f32) {
    // Check for available SIMD features from best to worst
    if is_x86_feature_detected!("avx512f") {
        unsafe { compute_forces_simd_soa_avx512(particles, index, nodes, g, time) }
    } else if is_x86_feature_detected!("avx2") {
        unsafe { compute_forces_simd_soa_avx2(particles, index, nodes, g, time) }
    } else if is_x86_feature_detected!("sse4.1") {
        unsafe { compute_forces_simd_soa_sse41(particles, index, nodes, g, time) }
    } else {
        // Fallback to scalar implementation
        compute_forces_scalar_soa(particles, index, nodes, g, time)
    }
}

/// Creates a set of particles representing a Big Bang simulation
pub fn create_big_bang_particles(num_particles: usize, initial_radius: f64) -> Vec<Particle> {
    let mut particles = Vec::with_capacity(num_particles);

    for _ in 0..num_particles {
        // Create particles with a denser concentration toward the center
        let radius = initial_radius * (rand::random::<f64>().powf(0.5));
        let angle = 2.0 * PI * rand::random::<f64>();

        // Position (polar coordinates converted to Cartesian)
        let x = radius * angle.cos();
        let y = radius * angle.sin();

        // Initial velocity (perpendicular to radius, with some random variation)
        // This creates a basic rotation pattern
        let speed_scale = 0.5 * fast_sqrt_f64(radius); // Velocity increases with distance
        let perpendicular_angle = angle + PI/2.0;
        let velocity_variation = 0.2; // Random velocity component magnitude

        let vx = speed_scale * perpendicular_angle.cos() +
            velocity_variation * (rand::random::<f64>() - 0.5);
        let vy = speed_scale * perpendicular_angle.sin() +
            velocity_variation * (rand::random::<f64>() - 0.5);

        // Mass (some particles are more massive)
        let mass = if rand::random::<f64>() < 0.01 {
            // A few massive particles to simulate primary attractors
            10.0 + 5.0 * rand::random::<f64>()
        } else {
            0.1 + 0.9 * rand::random::<f64>()
        };

        // Spin (angular momentum)
        let spin = 0.01 * rand::random::<f64>();

        particles.push(Particle::new(x, y, vx, vy, mass, spin));
    }

    particles
}

/// SoA particle collection with Big Bang configuration
pub fn create_big_bang_particles_soa(
    num_particles: usize,
    initial_radius: f32
) -> ParticleCollection {
    let mut particles = ParticleCollection::new(num_particles);
    particles.count = num_particles;

    // Pre-allocate all arrays
    particles.positions_x.resize(num_particles, 0.0);
    particles.positions_y.resize(num_particles, 0.0);
    particles.velocities_x.resize(num_particles, 0.0);
    particles.velocities_y.resize(num_particles, 0.0);
    particles.masses.resize(num_particles, 0.0);
    particles.spins.resize(num_particles, 0.0);
    particles.ages.resize(num_particles, 0.0);
    particles.densities.resize(num_particles, 0.0);


    // Initialize in parallel for better performance
    (0..num_particles).into_iter().for_each(|i| {
        // Create particles with a denser concentration toward the center
        let radius = initial_radius * (rand::random::<f32>().powf(0.5));
        let angle = 2.0 * std::f32::consts::PI * rand::random::<f32>();

        // Position (polar coordinates converted to Cartesian)
        let x = radius * angle.cos();
        let y = radius * angle.sin();

        // Initial velocity (perpendicular to radius, with some random variation)
        let speed_scale = 0.5 * radius.sqrt(); // Velocity increases with distance
        let perpendicular_angle = angle + std::f32::consts::PI/2.0;
        let velocity_variation = 0.2; // Random velocity component magnitude

        let vx = speed_scale * perpendicular_angle.cos() +
            velocity_variation * (rand::random::<f32>() - 0.5);
        let vy = speed_scale * perpendicular_angle.sin() +
            velocity_variation * (rand::random::<f32>() - 0.5);

        // Mass (some particles are more massive)
        let mass = if rand::random::<f32>() < 0.01 {
            // A few massive particles to simulate primary attractors
            10.0 + 5.0 * rand::random::<f32>()
        } else {
            0.1 + 0.9 * rand::random::<f32>()
        };

        // Spin (angular momentum)
        let spin = 0.01 * rand::random::<f32>();

        // Fill in the arrays
        particles.positions_x[i] = x;
        particles.positions_y[i] = y;
        particles.velocities_x[i] = vx;
        particles.velocities_y[i] = vy;
        particles.masses[i] = mass;
        particles.spins[i] = spin;
        particles.ages[i] = 0.0;
        particles.densities[i] = 0.0;
    });

    particles
}


/// Simulate particle interactions for a time step using the Barnes-Hut algorithm
pub fn simulate_step(
    particles: &mut [Particle],
    bounds: Quad,
    theta: f64,
    g: f64,
    dt: f64,
    time: f64
) {
    // Build the Barnes-Hut tree
    let tree = build_tree(particles, bounds);

    // Compute forces on each particle in parallel
    // Use SIMD-optimized force calculation when available
    let forces: Vec<(f64, f64)> = particles.iter()
        .map(|p| compute_net_force(&tree, p, theta, g, time))
        .collect();

    // Update particles based on forces
    for (i, p) in particles.iter_mut().enumerate() {
        let (fx, fy) = forces[i];

        // Apply force to update velocity
        p.apply_force(fx, fy, dt);

        // Update position
        p.update_position(dt);

        // Apply any boundary conditions (optional)
        apply_boundary_conditions(p, bounds);
    }
}
pub fn simulate_step_optimized(
    particles: &mut [Particle],
    bounds: Quad,
    theta: f64,
    g: f64,
    dt: f64,
    time: f64
) {
    // Build the Barnes-Hut tree using the iterative method
    let tree = build_tree_iterative(particles, bounds);

    // Calculate parallel batch size based on available memory and CPU cores
    let cpu_cores = rayon::current_num_threads() / 2;
    let batch_size = (particles.len() / cpu_cores).max(1);

    // Process particles in batches to reduce memory pressure
    for batch in particles.chunks_mut(batch_size) {
        let forces: Vec<(f64, f64)> = batch.par_iter()
            .map(|p| {
                // Use the regular compute_net_force but with our non-recursive node collection
                let mut worklist = Vec::new();
                collect_approx_nodes_iterative(&tree, p, theta, &mut worklist);

                // Use the existing SIMD-optimized force calculations
                #[cfg(all(target_arch = "x86_64", feature = "avx512-simd"))]
                {
                    if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512vl") {
                        if worklist.len() > 1000 {
                            return unsafe { compute_forces_simd_avx512_f32(p, &worklist, g, time) };
                        }
                        if is_x86_feature_detected!("avx512f") {
                            return unsafe { compute_forces_simd_avx512(p, &worklist, g, time) };
                        }
                    }
                }
                // Fall back to AVX2 if available
                #[cfg(target_feature = "avx2")]
                {
                    if is_x86_feature_detected!("avx2") {
                        return unsafe { compute_forces_simd_avx2(p, &worklist, g, time) };
                    }
                }
                // Fall back to SSE4.1 if available
                #[cfg(target_feature = "sse4.1")]
                {
                    if is_x86_feature_detected!("sse4.1") {
                        return unsafe { compute_forces_simd_sse41(p, &worklist, g, time) };
                    }
                }

                // Fall back to scalar implementation
                compute_forces_scalar(p, &worklist, g, time)
            })
            .collect();

        // Update particles based on forces
        for (i, p) in batch.iter_mut().enumerate() {
            let (fx, fy) = forces[i];

            // Apply force to update velocity
            p.apply_force(fx, fy, dt);

            // Update position
            p.update_position(dt);

            // Apply any boundary conditions
            apply_boundary_conditions(p, bounds);
        }
    }
}
/// Apply boundary conditions to keep particles within simulation bounds
pub fn apply_boundary_conditions(p: &mut Particle, bounds: Quad) {
    let bound_size = bounds.half_size * 2.0;

    // Periodic boundary conditions (wraparound)
    if p.position.0 < bounds.cx - bounds.half_size {
        p.position.0 += bound_size;
    } else if p.position.0 >= bounds.cx + bounds.half_size {
        p.position.0 -= bound_size;
    }

    if p.position.1 < bounds.cy - bounds.half_size {
        p.position.1 += bound_size;
    } else if p.position.1 >= bounds.cy + bounds.half_size {
        p.position.1 -= bound_size;
    }
}
/// Apply boundary conditions to particles in SoA format
pub fn apply_boundary_conditions_soa(
    particles: &mut ParticleCollection,
    bounds: Quad
) {
    let bound_size_x = bounds.half_size as f32 * 2.0;
    let bound_size_y = bounds.half_size as f32 * 2.0;
    let min_x = (bounds.cx - bounds.half_size) as f32 ;
    let max_x = (bounds.cx + bounds.half_size) as f32 ;
    let min_y = (bounds.cy - bounds.half_size) as f32 ;
    let max_y = (bounds.cy + bounds.half_size) as f32 ;

    for i in 0..particles.count {
        // Periodic boundary conditions (wraparound)
        if particles.positions_x[i] < min_x {
            particles.positions_x[i] += bound_size_x;
        } else if particles.positions_x[i] >= max_x {
            particles.positions_x[i] -= bound_size_x;
        }

        if particles.positions_y[i] < min_y {
            particles.positions_y[i] += bound_size_y;
        } else if particles.positions_y[i] >= max_y {
            particles.positions_y[i] -= bound_size_y;
        }
    }
}

/// Run a complete simulation for a specified number of steps
pub fn run_simulation(
    num_particles: usize,
    initial_radius: f64,
    bounds: Quad,
    num_steps: usize,
    dt: f64,
    theta: f64,
    g: f64
) -> Vec<Vec<Particle>> {
    // Initialize particles
    let mut particles = create_big_bang_particles(num_particles, initial_radius);

    // Storage for simulation history
    let mut history = Vec::with_capacity(num_steps + 1);
    history.push(particles.clone());

    // Run simulation steps
    for step in 0..num_steps {
        let time = step as f64 * dt;
        simulate_step(&mut particles, bounds, theta, g, dt, time);
        history.push(particles.clone());
    }

    history
}

/// Multicore parallel simulation that uses all available CPU cores
pub fn run_parallel_simulation(
    num_particles: usize,
    initial_radius: f64,
    bounds: Quad,
    num_steps: usize,
    dt: f64,
    theta: f64,
    g: f64,
    chunk_size: usize
) -> Vec<Vec<Particle>> {
    use rayon::prelude::*;

    // Initialize particles
    let mut particles = create_big_bang_particles(num_particles, initial_radius);

    // Storage for simulation history (key frames only to save memory)
    let mut history = Vec::with_capacity((num_steps / chunk_size) + 1);
    history.push(particles.clone());

    // Process chunks of steps in parallel
    let chunks: Vec<_> = (0..num_steps).collect::<Vec<_>>()
        .chunks(chunk_size)
        .map(|c| (c[0], c.len()))
        .collect();

    for (start_step, steps_in_chunk) in chunks {
        // Run a chunk of simulation steps
        for step in 0..steps_in_chunk {
            let time = (start_step + step) as f64 * dt;
            simulate_step(&mut particles, bounds, theta, g, dt, time);
        }

        // Save state at the end of each chunk
        history.push(particles.clone());
    }

    history
}

/// Runs a fully optimized simulation with adaptive parameters
pub fn run_optimized_simulation(
    num_particles: usize,
    initial_radius: f64,
    bounds: Quad,
    sim_duration: f64,
    theta: f64,
    g: f64
) -> Vec<Vec<Particle>> {
    // Calculate appropriate time step based on particle density
    let particle_density = num_particles as f64 / (bounds.half_size * bounds.half_size * 4.0);
    let adaptive_dt = (0.01 / fast_sqrt_f64(particle_density)).clamp(0.001, 0.1);

    // Calculate number of steps
    let num_steps = (sim_duration / adaptive_dt) as usize;

    // Determine chunk size for parallel processing
    let available_threads = rayon::current_num_threads();
    let chunk_size = (num_steps / (available_threads * 10)).max(1);

    // Use parallel simulation for best performance
    run_parallel_simulation(
        num_particles,
        initial_radius,
        bounds,
        num_steps,
        adaptive_dt,
        theta,
        g,
        chunk_size
    )
}

/// Perform a simulation step using the SoA data structure and optimized SIMD ( from a &\[Particle] )
pub fn simulate_step_soa_from_slice(
    particles: &[Particle],
    bounds: Quad,
    theta: f32,
    g: f32,
    dt: f32,
    time: f32
) {
    let particles = &mut ParticleCollection::from_particle_slice(particles);
    // 1. Build Barnes-Hut tree
    let mut tree_builder_particles = Vec::with_capacity(particles.count);
    for i in 0..particles.count {
        tree_builder_particles.push(Particle {
            position: (particles.positions_x[i] as f64, particles.positions_y[i] as f64),
            velocity: Velocity2D {
                x: particles.velocities_x[i] as f64,
                y: particles.velocities_y[i] as f64,
            },
            mass: particles.masses[i] as f64,
            spin: particles.spins[i] as f64,
            age: particles.ages[i] as f64,
            density: particles.densities[i] as f64,
        });
    }

    // Convert bounds to f64 for tree building
    let tree_bounds = Quad {
        cx: bounds.cx as f64,
        cy: bounds.cy as f64,
        half_size: bounds.half_size as f64,
    };

    // Use iterative (non-recursive) tree building for large particle counts
    let tree = build_tree_iterative(&tree_builder_particles, tree_bounds);

    // 2. Calculate forces for all particles
    // Pre-allocate force arrays
    let mut forces_x = vec![0.0f32; particles.count];
    let mut forces_y = vec![0.0f32; particles.count];

    let aotr_forces_x = AtomicPtr::new(&mut forces_x);
    let aotr_forces_y = AtomicPtr::new(&mut forces_y);

    // Calculate parallel batch size based on available cores
    let cpu_cores = rayon::current_num_threads();
    let batch_size = (particles.count / cpu_cores).max(1);

    // Process particles in batches to reduce memory pressure
    (0..particles.count).into_par_iter()
        .chunks(batch_size)
        .for_each(|chunk| {
            for &i in &chunk {
                // For each particle, collect approximately nodes
                let mut worklist = Vec::new();
                let p = Particle {
                    position: (particles.positions_x[i] as f64, particles.positions_y[i] as f64),
                    velocity: Velocity2D {
                        x: particles.velocities_x[i] as f64,
                        y: particles.velocities_y[i] as f64,
                    },
                    mass: particles.masses[i] as f64,
                    spin: particles.spins[i] as f64,
                    age: particles.ages[i] as f64,
                    density: particles.densities[i] as f64,
                };

                // Collect approximated nodes using non-recursive method
                collect_approx_nodes_iterative(&tree, &p, theta as f64, &mut worklist);

                // Calculate forces using the best available SIMD implementation
                let (fx, fy) = compute_forces_simd_soa(particles, i, &worklist, g, time);

                let forces_x = unsafe { &mut *aotr_forces_x.load(Relaxed) };
                let forces_y = unsafe { &mut *aotr_forces_y.load(Relaxed) };

                // Store forces for later application
                forces_x[i] = fx;
                forces_y[i] = fy;
            }
        });

    // 3. Update velocities using SIMD
    update_velocities_simd(particles, &forces_x, &forces_y, dt);

    // 4. Update positions using SIMD
    update_positions_simd(particles, dt);

    // 5. Apply boundary conditions if needed
    apply_boundary_conditions_soa(particles, bounds);
}

/// Perform a simulation step using the SoA data structure with tuned resource usage
pub fn simulate_step_soa(
    particles: &mut ParticleCollection,
    bounds: Quad,
    theta: f32,
    g: f32,
    dt: f32,
    time: f32
) {
    // Reduce the tree data to essential components only
    let _tree_start = std::time::Instant::now();
    let particle_count = particles.count;

    // Only allocate what we'll actually need - no need for full capacity
    let reduced_capacity = (particle_count * 3) / 4; // 75% to account for empty zones
    let mut tree_builder_particles = Vec::with_capacity(reduced_capacity);

    // Step through particles with a stride for tree building
    // Many particles are spatially close, so we can use a subset for the tree
    // The approximation will still be valid for force calculations
    let tree_stride = if particle_count > 32_000 { 2 } else { 1 };

    for i in (0..particle_count).step_by(tree_stride) {
        tree_builder_particles.push(Particle {
            position: (particles.positions_x[i] as f64, particles.positions_y[i] as f64),
            velocity: Velocity2D { x: 0.0, y: 0.0 }, // We don't need velocity for tree building
            mass: particles.masses[i] as f64,
            spin: 0.0,       // Not needed for tree
            age: 0.0,        // Not needed for tree
            density: 0.0,    // Not needed for tree
        });
    }

    // Slightly increase theta for performance at expense of some accuracy
    let tree_theta = (theta * 1.05) as f64;

    // Convert bounds for tree building
    let tree_bounds = Quad {
        cx: bounds.cx as f64,
        cy: bounds.cy as f64,
        half_size: bounds.half_size as f64,
    };

    // Use iterative tree building with reduced particle set
    let tree = build_tree_iterative(&tree_builder_particles, tree_bounds);

    // Clear the temporary particle vector to free memory before force calculations
    drop(tree_builder_particles);

    // 2. Calculate forces with adaptive batch sizing
    let _force_start = std::time::Instant::now();

    // Pre-allocate forces arrays
    let mut forces_x = vec![0.0f32; particle_count];
    let mut forces_y = vec![0.0f32; particle_count];

    // Store particle data needed for force calculation to avoid borrow issues
    let positions_x: Vec<f32> = particles.positions_x.clone();
    let positions_y: Vec<f32> = particles.positions_y.clone();
    let masses: Vec<f32> = particles.masses.clone();

    // Determine optimal batch size based on particle count
    // Smaller batches for larger particle counts
    let core_count = rayon::current_num_threads();
    let batch_size = match particle_count {
        n if n > 500_000 => (n / (core_count * 8)).max(64),
        n if n > 100_000 => (n / (core_count * 4)).max(128),
        n if n > 50_000 => (n / (core_count * 2)).max(256),
        _ => (particle_count / core_count).max(512),
    };

    // Using channels to collect results from threads
    let (sender, receiver) = std::sync::mpsc::channel();

    // Process in parallel using thread pool to control concurrency
    rayon::scope(|s| {
        // Split particle range into batches
        for batch_start in (0..particle_count).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(particle_count);

            // Capture references to shared data
            let tree_ref = &tree;
            let pos_x_ref = &positions_x;
            let pos_y_ref = &positions_y;
            let masses_ref = &masses;
            let sender = sender.clone();

            s.spawn(move |_| {
                // Thread-local collections for forces
                let mut local_forces = Vec::with_capacity(batch_end - batch_start);

                for idx in batch_start..batch_end {
                    // Skip processing particles with negligible mass
                    if masses_ref[idx] < 0.01 {
                        continue;
                    }

                    // Create particle for force calculation
                    let p = Particle {
                        position: (pos_x_ref[idx] as f64, pos_y_ref[idx] as f64),
                        velocity: Velocity2D { x: 0.0, y: 0.0 },
                        mass: masses_ref[idx] as f64,
                        spin: 0.0,
                        age: 0.0,
                        density: 0.0,
                    };

                    // Collect nodes that affect this particle
                    let mut worklist = Vec::with_capacity(64);
                    collect_approx_nodes_iterative(tree_ref, &p, tree_theta, &mut worklist);

                    if !worklist.is_empty() {
                        // Use a simpler force calculation
                        let (fx, fy) = compute_forces_scalar(&p, &worklist, g as f64, time as f64);

                        // Store forces with particle index
                        local_forces.push((idx, fx as f32, fy as f32));
                    }
                }

                // Send local results back to main thread
                sender.send(local_forces).expect("Channel send failed");
            });
        }
    });

    // Drop the original sender to close the channel after all spawned threads complete
    drop(sender);

    // Collect results from all threads
    while let Ok(batch_results) = receiver.recv() {
        for (idx, fx, fy) in batch_results {
            forces_x[idx] = fx;
            forces_y[idx] = fy;
        }
    }

    // 3 & 4. Update velocities and positions in one pass for better cache usage
    let _update_start = std::time::Instant::now();

    // Process in chunks for better cache performance
    let chunk_size = 2048; // Adjusted for better cache line utilization

    for chunk_start in (0..particle_count).step_by(chunk_size) {
        let chunk_end = (chunk_start + chunk_size).min(particle_count);

        for i in chunk_start..chunk_end {
            // Skip updates for nearly-stationary particles
            let fx = forces_x[i];
            let fy = forces_y[i];

            if fx.abs() < 1e-6 && fy.abs() < 1e-6 {
                // Minimal force, just age the particle
                particles.ages[i] += dt;
                continue;
            }

            // Update velocity
            let inv_mass = 1.0 / particles.masses[i].max(0.001); // Prevent division by zero
            let dvx = fx * inv_mass * dt;
            let dvy = fy * inv_mass * dt;

            particles.velocities_x[i] += dvx;
            particles.velocities_y[i] += dvy;

            // Update position
            particles.positions_x[i] += particles.velocities_x[i] * dt;
            particles.positions_y[i] += particles.velocities_y[i] * dt;
            particles.ages[i] += dt;
        }
    }

    // 5. Apply boundary conditions with early exit for particles in bounds
    let _bounds_start = std::time::Instant::now();

    let half_size = bounds.half_size as f32;
    let min_x = (bounds.cx as f32 - half_size);
    let max_x = (bounds.cx as f32 + half_size);
    let min_y = (bounds.cy as f32 - half_size);
    let max_y = (bounds.cy as f32 + half_size);

    // Process boundary checks in chunks
    for chunk_start in (0..particle_count).step_by(chunk_size) {
        let chunk_end = (chunk_start + chunk_size).min(particle_count);

        for i in chunk_start..chunk_end {
            let x = particles.positions_x[i];
            let y = particles.positions_y[i];

            // Check if already in bounds (most common case)
            if x >= min_x && x < max_x && y >= min_y && y < max_y {
                continue;
            }

            // Apply wraparound for out-of-bounds particles
            let bound_size_x = half_size * 2.0;
            let bound_size_y = half_size * 2.0;

            if x < min_x {
                particles.positions_x[i] += bound_size_x;
            } else if x >= max_x {
                particles.positions_x[i] -= bound_size_x;
            }

            if y < min_y {
                particles.positions_y[i] += bound_size_y;
            } else if y >= max_y {
                particles.positions_y[i] -= bound_size_y;
            }
        }
    }
}

/// Run a complete simulation for a specified number of steps using SoA
pub fn run_simulation_soa(
    num_particles: usize,
    initial_radius: f32,
    bounds: Quad,
    num_steps: usize,
    dt: f32,
    theta: f32,
    g: f32
) -> ParticleCollection {
    // Initialize particles
    let mut particles = create_big_bang_particles_soa(num_particles, initial_radius);

    // Run simulation steps
    for step in 0..num_steps {
        let time = step as f32 * dt;
        simulate_step_soa(&mut particles, bounds, theta, g, dt, time);
    }

    particles
}

/// Higher-level function that sets up and runs an optimized cosmological simulation
pub fn run_optimized_cosmological_simulation(
    num_particles: usize,
    initial_radius: f32,
    sim_duration: f32,
    theta: f32,
    g: f32
) -> ParticleCollection {
    // Calculate appropriate bounds
    let bounds = Quad {
        cx: 0.0,
        cy: 0.0,
        half_size: initial_radius as f64 * 80.0,
    };

    // Calculate appropriate time step based on particle density
    let particle_density = num_particles as f32 / (bounds.half_size * bounds.half_size * 4.0) as f32;
    let adaptive_dt = (0.01 / particle_density.sqrt()).clamp(0.001, 0.1);

    // Calculate number of steps
    let num_steps = (sim_duration / adaptive_dt) as usize;

    // Create initial particles
    let mut particles = create_big_bang_particles_soa(num_particles, initial_radius);

    // Modify mass distribution for more interesting dynamics
    modify_particle_masses_soa(&mut particles);

    // Randomize directions a bit for more natural distribution
    randomize_particle_directions_soa(&mut particles);

    // Run simulation
    for step in 0..num_steps {
        let time = step as f32 * adaptive_dt;
        simulate_step_soa(&mut particles, bounds, theta, g, adaptive_dt, time);
    }

    particles
}

/// Modify particle masses to create a more interesting distribution
pub fn modify_particle_masses_soa(particles: &mut ParticleCollection) {
    let aptr_particles = AtomicPtr::new(particles);
    // Process in parallel for better performance
    (0..particles.count).into_par_iter().for_each(|i| {
        // Create a more diverse mass distribution
        let mass_type = rand::random::<f32>();
        let particles = unsafe { &mut *aptr_particles.load(Relaxed) };

        if mass_type < 0.001 {
            // Super massive "suns" (0.1% of particles)
            particles.masses[i] = std::f32::consts::PI * rand::random::<f32>().mul_add(5000.0, 2000.0);
            particles.spins[i] *= 20.0;
        } else if mass_type < 0.01 {
            // Medium "planets" (0.9% of particles)
            particles.masses[i] = std::f32::consts::PI * rand::random::<f32>().mul_add(500.0, 100.0);
            particles.spins[i] *= 10.0;
        } else if mass_type < 0.1 {
            // Small "asteroids" (9% of particles)
            particles.masses[i] = std::f32::consts::PI * rand::random::<f32>().mul_add(50.0, 20.0);
            particles.spins[i] *= 5.0;
        } else {
            // Tiny "dust" (90% of particles)
            particles.masses[i] = std::f32::consts::PI * rand::random::<f32>().mul_add(10.0, 1.0);
        }
    });
}

/// Randomize particle directions a bit
pub fn randomize_particle_directions_soa(particles: &mut ParticleCollection) {
    let aptr_particles = AtomicPtr::new(particles);
    // Process in parallel for better performance
    (0..particles.count).into_par_iter().for_each(|i| {
        let particles = unsafe { &mut *aptr_particles.load(Relaxed) };
        let angle_variation = rand::random::<f32>().mul_add(0.5, -0.5) * std::f32::consts::PI;
        let vx = particles.velocities_x[i];
        let vy = particles.velocities_y[i];
        let current_speed = (vx * vx + vy * vy).sqrt();

        let current_dir = if current_speed > 0.0 {
            (vx / current_speed, vy / current_speed)
        } else {
            (0.0, 0.0)
        };

        // Rotate the direction by a random angle
        let cos_angle = angle_variation.cos();
        let sin_angle = angle_variation.sin();
        let new_x = current_dir.0 * cos_angle - current_dir.1 * sin_angle;
        let new_y = current_dir.0 * sin_angle + current_dir.1 * cos_angle;

        // Update velocity with new direction but maintain speed
        particles.velocities_x[i] = new_x * current_speed;
        particles.velocities_y[i] = new_y * current_speed;
    });
}