use crate::models::{Velocity2D, Direction2D};
use rayon::prelude::*;
use std::f64::consts::PI;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use std::arch::x86_64::*;

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
        (dx * dx + dy * dy).sqrt()
    }

    /// Get the direction to another particle
    pub fn direction_to(&self, other: &Particle) -> Direction2D {
        let dx = other.position.0 - self.position.0;
        let dy = other.position.1 - self.position.1;
        let magnitude = (dx * dx + dy * dy).sqrt();

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
                let dist = dist_sq.sqrt();

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
        let dist = dist_sq.sqrt();

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
            let dist = dist_sq.sqrt();

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
                let dist = dist_sq.sqrt();

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
        let dist = dist_sq.sqrt();

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
        let dist = dist_sq.sqrt();

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
        let dist = dist_sq.sqrt();

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
        let dist = dist_sq.sqrt();

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
    }

    // Fall back to scalar implementation
    compute_forces_scalar(p, &worklist, g, time)
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
        let speed_scale = 0.5 * radius.sqrt(); // Velocity increases with distance
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
    let forces: Vec<(f64, f64)> = particles.par_iter()
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
    let cpu_cores = rayon::current_num_threads();
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
    let adaptive_dt = (0.01 / particle_density.sqrt()).clamp(0.001, 0.1);

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

