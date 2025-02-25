use bumpalo::Bump;

/// Represents a square region in 2D space.
///
/// This structure is used to define the boundaries of a region in Barnes-Hut algorithm.
/// Each `Quad` has a center position (cx, cy) and a half-size, which is half the length
/// of one side of the square.
///
/// # Examples
///
/// ```
/// use rs_physics::particles::Quad;
///
/// // Create a square with center at origin and side length of 2.0
/// let quad = Quad { cx: 0.0, cy: 0.0, half_size: 1.0 };
///
/// // Check if a point is inside the quad
/// assert!(quad.contains(0.5, 0.5));
/// assert!(!quad.contains(1.5, 0.5)); // Outside the square
/// ```
#[derive(Clone, Copy, Debug)]
pub struct Quad {
    pub cx: f64,        // center x-coordinate
    pub cy: f64,        // center y-coordinate
    pub half_size: f64, // half the length of one side
}

impl Quad {
    /// Returns true if the point (x, y) is inside this quad.
    ///
    /// The quad's boundary is inclusive on the lower bounds and exclusive on the upper bounds,
    /// which helps avoid ambiguity when placing particles on boundaries.
    ///
    /// # Arguments
    ///
    /// * `x` - The x-coordinate of the point
    /// * `y` - The y-coordinate of the point
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::particles::Quad;
    ///
    /// let quad = Quad { cx: 0.0, cy: 0.0, half_size: 1.0 };
    ///
    /// // Points inside the quad
    /// assert!(quad.contains(0.0, 0.0));  // Center
    /// assert!(quad.contains(-0.9, 0.9)); // Near top-left corner
    ///
    /// // Points outside the quad
    /// assert!(!quad.contains(1.0, 0.0));  // Right edge (exclusive)
    /// assert!(!quad.contains(-2.0, 0.0)); // Far left
    /// ```
    pub fn contains(&self, x: f64, y: f64) -> bool {
        x >= self.cx - self.half_size &&
            x <  self.cx + self.half_size &&
            y >= self.cy - self.half_size &&
            y <  self.cy + self.half_size
    }

    /// Subdivides the quad into four smaller quads (NW, NE, SW, SE).
    ///
    /// This is a key operation in the Barnes-Hut algorithm, which recursively divides
    /// space into quadrants.
    ///
    /// # Returns
    ///
    /// A tuple of four quads representing the northwest, northeast, southwest,
    /// and southeast quadrants, respectively.
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::particles::Quad;
    ///
    /// let quad = Quad { cx: 0.0, cy: 0.0, half_size: 1.0 };
    /// let (nw, ne, sw, se) = quad.subdivide();
    ///
    /// // Check northwest quadrant
    /// assert_eq!(nw.cx, -0.5);
    /// assert_eq!(nw.cy, 0.5);
    /// assert_eq!(nw.half_size, 0.5);
    ///
    /// // Check northeast quadrant
    /// assert_eq!(ne.cx, 0.5);
    /// assert_eq!(ne.cy, 0.5);
    ///
    /// // Check that a point in the original quad is now in the correct subquad
    /// assert!(nw.contains(-0.25, 0.25)); // Point should be in NW quadrant
    /// assert!(se.contains(0.25, -0.25)); // Point should be in SE quadrant
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

/// A simple particle representation for n-body simulations.
///
/// This struct stores the position and mass of a particle in a 2D space.
///
/// # Examples
///
/// ```
/// use rs_physics::particles::ParticleData;
///
/// // Create a particle at position (1.0, 2.0) with mass 3.0
/// let particle = ParticleData { x: 1.0, y: 2.0, mass: 3.0 };
///
/// // Access particle properties
/// assert_eq!(particle.x, 1.0);
/// assert_eq!(particle.y, 2.0);
/// assert_eq!(particle.mass, 3.0);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct ParticleData {
    pub x: f64,
    pub y: f64,
    pub mass: f64,
}

/// Barnes–Hut tree node for 2D space, using arena allocation.
///
/// This enum represents the nodes in a Barnes-Hut tree. There are three types of nodes:
/// - `Empty`: An empty region with no particles
/// - `Leaf`: A region containing exactly one particle
/// - `Internal`: A region containing multiple particles, with aggregated data (center of mass, total mass)
///   and references to four child nodes representing quadrants
///
/// The lifetime parameter `'a` is tied to the arena allocator's lifetime.
///
/// # Examples
///
/// ```
/// use rs_physics::particles::{BarnesHutNode, ParticleData, Quad};
/// use bumpalo::Bump;
///
/// // Create a new arena
/// let arena = Bump::new();
///
/// // Create an empty node
/// let quad = Quad { cx: 0.0, cy: 0.0, half_size: 1.0 };
/// let empty_node = BarnesHutNode::new(quad);
///
/// // Create a leaf node with a particle
/// let particle = ParticleData { x: 0.5, y: 0.5, mass: 1.0 };
/// let leaf_node = BarnesHutNode::Leaf(quad, particle);
/// ```
#[derive(Debug, Clone)]
pub enum BarnesHutNode<'a> {
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
        nw: &'a BarnesHutNode<'a>,
        ne: &'a BarnesHutNode<'a>,
        sw: &'a BarnesHutNode<'a>,
        se: &'a BarnesHutNode<'a>,
    },
}

/// A wrapper around the Barnes-Hut tree that manages the arena allocator.
///
/// This structure holds a reference to a bump allocator and the root node of a Barnes-Hut tree.
/// It provides methods for building and manipulating the tree.
///
/// # Examples
///
/// ```
/// use rs_physics::particles::{BarnesHutTree, ParticleData, Quad};
/// use bumpalo::Bump;
///
/// // Create particles
/// let particles = vec![
///     ParticleData { x: 0.1, y: 0.1, mass: 1.0 },
///     ParticleData { x: -0.5, y: 0.5, mass: 2.0 },
///     ParticleData { x: 0.4, y: -0.2, mass: 1.5 },
/// ];
///
/// // Define simulation boundaries
/// let quad = Quad { cx: 0.0, cy: 0.0, half_size: 1.0 };
///
/// // Create an arena and build the tree
/// let arena = Bump::new();
/// let tree = BarnesHutTree::build(&arena, &particles, quad);
///
/// // Calculate force on a test particle
/// let test_particle = ParticleData { x: 0.0, y: 0.0, mass: 1.0 };
/// let (fx, fy) = tree.compute_force(test_particle, 0.5, 6.67430e-11);
/// ```
pub struct BarnesHutTree<'a> {
    arena: &'a Bump,
    root: BarnesHutNode<'a>,
}

impl<'a> BarnesHutNode<'a> {
    /// Creates a new empty node with the given quad.
    ///
    /// # Arguments
    ///
    /// * `quad` - The spatial region this node represents
    ///
    /// # Returns
    ///
    /// A new empty `BarnesHutNode`
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::particles::{BarnesHutNode, Quad};
    ///
    /// let quad = Quad { cx: 0.0, cy: 0.0, half_size: 1.0 };
    /// let node = BarnesHutNode::new(quad);
    ///
    /// // The new node should be empty
    /// match node {
    ///     BarnesHutNode::Empty(q) => assert_eq!(q.cx, 0.0),
    ///     _ => panic!("Expected an Empty node"),
    /// }
    /// ```
    pub fn new(quad: Quad) -> Self {
        BarnesHutNode::Empty(quad)
    }

    /// Helper method to retrieve the quad for a node.
    ///
    /// # Returns
    ///
    /// The `Quad` representing the spatial region of this node
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::particles::{BarnesHutNode, ParticleData, Quad};
    ///
    /// let quad = Quad { cx: 1.0, cy: 2.0, half_size: 3.0 };
    ///
    /// // Create different types of nodes
    /// let empty_node = BarnesHutNode::Empty(quad);
    /// let leaf_node = BarnesHutNode::Leaf(quad, ParticleData { x: 0.0, y: 0.0, mass: 1.0 });
    ///
    /// // Get the quad for each node
    /// assert_eq!(empty_node.quad().cx, 1.0);
    /// assert_eq!(leaf_node.quad().cy, 2.0);
    /// assert_eq!(leaf_node.quad().half_size, 3.0);
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
    /// This implements the core of the Barnes-Hut approximation algorithm:
    /// - If the node is empty, the force is zero
    /// - If the node is a leaf, compute the exact force between the two particles
    /// - If the node is internal and far enough away (determined by `theta`),
    ///   use the aggregate mass and center of mass
    /// - Otherwise, recurse into the child nodes
    ///
    /// # Arguments
    ///
    /// * `p` - The particle to compute force on
    /// * `theta` - The Barnes-Hut approximation parameter (typically 0.5-1.0)
    /// * `g` - The gravitational constant
    ///
    /// # Returns
    ///
    /// A tuple of (force_x, force_y) representing the force vector
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::particles::{BarnesHutNode, ParticleData, Quad, build_tree};
    ///
    /// // Create some particles
    /// let particles = vec![
    ///     ParticleData { x: 1.0, y: 0.0, mass: 1.0 },
    ///     ParticleData { x: -1.0, y: 0.0, mass: 1.0 },
    /// ];
    ///
    /// // Define the region and build the tree
    /// let quad = Quad { cx: 0.0, cy: 0.0, half_size: 2.0 };
    /// let tree = build_tree(&particles, quad);
    ///
    /// // Compute force on a test particle
    /// let test_particle = ParticleData { x: 0.0, y: 0.0, mass: 1.0 };
    /// let g = 6.67430e-11; // Gravitational constant
    /// let theta = 0.5;     // Approximation parameter
    ///
    /// let (fx, fy) = tree.compute_force(test_particle, theta, g);
    ///
    /// // The forces from the two particles should cancel out
    /// assert!(fx.abs() < 1e-10);
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

impl<'a> BarnesHutTree<'a> {
    /// Creates a new empty Barnes-Hut tree.
    ///
    /// # Arguments
    ///
    /// * `arena` - A reference to a bump allocator
    /// * `quad` - The spatial region this tree will represent
    ///
    /// # Returns
    ///
    /// A new empty `BarnesHutTree`
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::particles::{BarnesHutTree, Quad};
    /// use bumpalo::Bump;
    ///
    /// let arena = Bump::new();
    /// let quad = Quad { cx: 0.0, cy: 0.0, half_size: 10.0 };
    /// let tree = BarnesHutTree::new(&arena, quad);
    ///
    /// // The tree should have an empty root node
    /// ```
    pub fn new(arena: &'a Bump, quad: Quad) -> Self {
        Self {
            arena,
            root: BarnesHutNode::Empty(quad),
        }
    }

    /// Inserts a particle into the Barnes-Hut tree.
    ///
    /// This method creates a new tree by inserting the particle into the current tree.
    /// The tree structure is recursively updated to maintain the Barnes-Hut invariants.
    ///
    /// # Arguments
    ///
    /// * `p` - The particle to insert
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::particles::{BarnesHutTree, ParticleData, Quad};
    /// use bumpalo::Bump;
    ///
    /// let arena = Bump::new();
    /// let quad = Quad { cx: 0.0, cy: 0.0, half_size: 1.0 };
    /// let mut tree = BarnesHutTree::new(&arena, quad);
    ///
    /// // Insert a particle
    /// let particle = ParticleData { x: 0.5, y: 0.5, mass: 1.0 };
    /// tree.insert(particle);
    ///
    /// // Insert another particle
    /// let particle2 = ParticleData { x: -0.5, y: -0.5, mass: 2.0 };
    /// tree.insert(particle2);
    ///
    /// // Calculate the force on a test particle
    /// let test_particle = ParticleData { x: 0.0, y: 0.0, mass: 1.0 };
    /// let (fx, fy) = tree.compute_force(test_particle, 0.5, 6.67430e-11);
    /// ```
    pub fn insert(&mut self, p: ParticleData) {
        // Create a new root by inserting into the current root
        let new_root = self.insert_impl(&self.root, p);
        self.root = new_root;
    }

    /// Internal implementation of particle insertion
    fn insert_impl(&self, node: &BarnesHutNode<'a>, p: ParticleData) -> BarnesHutNode<'a> {
        match node {
            BarnesHutNode::Empty(quad) => {
                BarnesHutNode::Leaf(*quad, p)
            }
            BarnesHutNode::Leaf(quad, existing) => {
                // Create a new internal node with arena-allocated children
                let (nw_quad, ne_quad, sw_quad, se_quad) = quad.subdivide();

                // Allocate empty child nodes in the arena
                let nw_node = self.arena.alloc(BarnesHutNode::Empty(nw_quad));
                let ne_node = self.arena.alloc(BarnesHutNode::Empty(ne_quad));
                let sw_node = self.arena.alloc(BarnesHutNode::Empty(sw_quad));
                let se_node = self.arena.alloc(BarnesHutNode::Empty(se_quad));

                // Insert existing particle
                let existing_particle = *existing;

                let total_mass = p.mass + existing_particle.mass;
                let com_x = (p.x * p.mass + existing_particle.x * existing_particle.mass) / total_mass;
                let com_y = (p.y * p.mass + existing_particle.y * existing_particle.mass) / total_mass;

                // Insert the existing particle into the appropriate quadrant
                if nw_quad.contains(existing_particle.x, existing_particle.y) {
                    *nw_node = self.insert_impl(nw_node, existing_particle);
                } else if ne_quad.contains(existing_particle.x, existing_particle.y) {
                    *ne_node = self.insert_impl(ne_node, existing_particle);
                } else if sw_quad.contains(existing_particle.x, existing_particle.y) {
                    *sw_node = self.insert_impl(sw_node, existing_particle);
                } else if se_quad.contains(existing_particle.x, existing_particle.y) {
                    *se_node = self.insert_impl(se_node, existing_particle);
                }

                // Insert the new particle into the appropriate quadrant
                if nw_quad.contains(p.x, p.y) {
                    *nw_node = self.insert_impl(nw_node, p);
                } else if ne_quad.contains(p.x, p.y) {
                    *ne_node = self.insert_impl(ne_node, p);
                } else if sw_quad.contains(p.x, p.y) {
                    *sw_node = self.insert_impl(sw_node, p);
                } else if se_quad.contains(p.x, p.y) {
                    *se_node = self.insert_impl(se_node, p);
                }

                // Create the internal node
                BarnesHutNode::Internal {
                    quad: *quad,
                    mass: total_mass,
                    com_x,
                    com_y,
                    nw: nw_node,
                    ne: ne_node,
                    sw: sw_node,
                    se: se_node,
                }
            }
            BarnesHutNode::Internal { quad, mass, com_x, com_y, nw, ne, sw, se } => {
                // Update center of mass and total mass
                let new_total_mass = *mass + p.mass;
                let new_com_x = (*com_x * *mass + p.x * p.mass) / new_total_mass;
                let new_com_y = (*com_y * *mass + p.y * p.mass) / new_total_mass;

                // Determine which quadrant the particle belongs in
                let nw_quad = nw.quad();
                let ne_quad = ne.quad();
                let sw_quad = sw.quad();
                let se_quad = se.quad();

                let (new_nw, new_ne, new_sw, new_se) = match (
                    nw_quad.contains(p.x, p.y),
                    ne_quad.contains(p.x, p.y),
                    sw_quad.contains(p.x, p.y),
                    se_quad.contains(p.x, p.y)
                ) {
                    (true, false, false, false) => {
                        // Particle belongs in NW quadrant
                        (self.arena.alloc(self.insert_impl(nw, p)),
                         self.arena.alloc((**ne).clone()),
                         self.arena.alloc((**sw).clone()),
                         self.arena.alloc((**se).clone()))
                    },
                    (false, true, false, false) => {
                        // Particle belongs in NE quadrant
                        (self.arena.alloc((**nw).clone()),
                         self.arena.alloc(self.insert_impl(ne, p)),
                         self.arena.alloc((**sw).clone()),
                         self.arena.alloc((**se).clone()))
                    },
                    (false, false, true, false) => {
                        // Particle belongs in SW quadrant
                        (self.arena.alloc((**nw).clone()),
                         self.arena.alloc((**ne).clone()),
                         self.arena.alloc(self.insert_impl(sw, p)),
                         self.arena.alloc((**se).clone()))
                    },
                    (false, false, false, true) => {
                        // Particle belongs in SE quadrant
                        (self.arena.alloc((**nw).clone()),
                         self.arena.alloc((**ne).clone()),
                         self.arena.alloc((**sw).clone()),
                         self.arena.alloc(self.insert_impl(se, p)))
                    },
                    _ => {
                        // Fallback for edge cases
                        (self.arena.alloc((**nw).clone()),
                         self.arena.alloc((**ne).clone()),
                         self.arena.alloc((**sw).clone()),
                         self.arena.alloc((**se).clone()))
                    }
                };
                // Create the new internal node
                BarnesHutNode::Internal {
                    quad: *quad,
                    mass: new_total_mass,
                    com_x: new_com_x,
                    com_y: new_com_y,
                    nw: new_nw,
                    ne: new_ne,
                    sw: new_sw,
                    se: new_se,
                }
            }
        }
    }

    /// Computes the force on a particle using the tree.
    ///
    /// # Arguments
    ///
    /// * `p` - The particle to compute force on
    /// * `theta` - The Barnes-Hut approximation parameter (typically 0.5-1.0)
    /// * `g` - The gravitational constant
    ///
    /// # Returns
    ///
    /// A tuple of (force_x, force_y) representing the force vector
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::particles::{BarnesHutTree, ParticleData, Quad};
    /// use bumpalo::Bump;
    ///
    /// // Create an arena and some particles
    /// let arena = Bump::new();
    /// let particles = vec![
    ///     ParticleData { x: 1.0, y: 0.0, mass: 1.0 },
    ///     ParticleData { x: 0.0, y: 1.0, mass: 1.0 },
    /// ];
    ///
    /// // Define the region and build the tree
    /// let quad = Quad { cx: 0.0, cy: 0.0, half_size: 2.0 };
    /// let tree = BarnesHutTree::build(&arena, &particles, quad);
    ///
    /// // Compute force on a test particle
    /// let test_particle = ParticleData { x: 0.0, y: 0.0, mass: 1.0 };
    /// let (fx, fy) = tree.compute_force(test_particle, 0.5, 6.67430e-11);
    ///
    /// // The force should be diagonally outward (equal components)
    /// assert!((fx - fy).abs() < 1e-10);
    /// ```
    pub fn compute_force(&self, p: ParticleData, theta: f64, g: f64) -> (f64, f64) {
        self.root.compute_force(p, theta, g)
    }

    /// Build a Barnes-Hut tree from a slice of particles
    ///
    /// This method efficiently constructs a Barnes-Hut tree from a collection of particles,
    /// using arena allocation for optimal performance.
    ///
    /// # Arguments
    ///
    /// * `arena` - A reference to a bump allocator
    /// * `particles` - A slice of particles to include in the tree
    /// * `quad` - The spatial region the tree will represent
    ///
    /// # Returns
    ///
    /// A new `BarnesHutTree` containing all the particles
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::particles::{BarnesHutTree, ParticleData, Quad};
    /// use bumpalo::Bump;
    ///
    /// // Create an arena and some particles
    /// let arena = Bump::new();
    /// let particles = vec![
    ///     ParticleData { x: 0.1, y: 0.1, mass: 1.0 },
    ///     ParticleData { x: -0.5, y: 0.5, mass: 2.0 },
    ///     ParticleData { x: 0.7, y: -0.3, mass: 1.5 },
    /// ];
    ///
    /// // Define the region
    /// let quad = Quad { cx: 0.0, cy: 0.0, half_size: 1.0 };
    ///
    /// // Build the tree
    /// let tree = BarnesHutTree::build(&arena, &particles, quad);
    ///
    /// // Calculate force on a test particle
    /// let test_particle = ParticleData { x: 0.0, y: 0.0, mass: 1.0 };
    /// let (fx, fy) = tree.compute_force(test_particle, 0.5, 6.67430e-11);
    /// ```
    pub fn build(arena: &'a Bump, particles: &[ParticleData], quad: Quad) -> Self {
        if particles.is_empty() {
            return Self::new(arena, quad);
        }

        if particles.len() == 1 {
            return Self {
                arena,
                root: BarnesHutNode::Leaf(quad, particles[0]),
            };
        }

        let (nw_quad, ne_quad, sw_quad, se_quad) = quad.subdivide();

        // Pre-allocate vectors with estimated capacity
        let estimated_capacity = particles.len() / 4;
        let mut nw_particles = Vec::with_capacity(estimated_capacity);
        let mut ne_particles = Vec::with_capacity(estimated_capacity);
        let mut sw_particles = Vec::with_capacity(estimated_capacity);
        let mut se_particles = Vec::with_capacity(estimated_capacity);

        // Distribute particles to quadrants
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

        // Build subtrees sequentially
        let nw_tree = Self::build_subtree(arena, &nw_particles, nw_quad);
        let ne_tree = Self::build_subtree(arena, &ne_particles, ne_quad);
        let sw_tree = Self::build_subtree(arena, &sw_particles, sw_quad);
        let se_tree = Self::build_subtree(arena, &se_particles, se_quad);

        // Calculate total mass and center of mass
        let mut total_mass = 0.0;
        let mut com_x = 0.0;
        let mut com_y = 0.0;

        let mut update_mass_com = |node: &BarnesHutNode<'a>| {
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

        // Allocate nodes in the arena
        let nw_node = arena.alloc(nw_tree);
        let ne_node = arena.alloc(ne_tree);
        let sw_node = arena.alloc(sw_tree);
        let se_node = arena.alloc(se_tree);

        Self {
            arena,
            root: BarnesHutNode::Internal {
                quad,
                mass: total_mass,
                com_x,
                com_y,
                nw: nw_node,
                ne: ne_node,
                sw: sw_node,
                se: se_node,
            }
        }
    }

    // Helper method to build a subtree
    fn build_subtree(arena: &'a Bump, particles: &[ParticleData], quad: Quad) -> BarnesHutNode<'a> {
        if particles.is_empty() {
            return BarnesHutNode::Empty(quad);
        }

        if particles.len() == 1 {
            return BarnesHutNode::Leaf(quad, particles[0]);
        }

        let (nw_quad, ne_quad, sw_quad, se_quad) = quad.subdivide();

        // Pre-allocate vectors with estimated capacity
        let estimated_capacity = particles.len() / 4;
        let mut nw_particles = Vec::with_capacity(estimated_capacity);
        let mut ne_particles = Vec::with_capacity(estimated_capacity);
        let mut sw_particles = Vec::with_capacity(estimated_capacity);
        let mut se_particles = Vec::with_capacity(estimated_capacity);

        // Distribute particles to quadrants
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

        // Recursively build subtrees
        let nw_node = arena.alloc(Self::build_subtree(arena, &nw_particles, nw_quad));
        let ne_node = arena.alloc(Self::build_subtree(arena, &ne_particles, ne_quad));
        let sw_node = arena.alloc(Self::build_subtree(arena, &sw_particles, sw_quad));
        let se_node = arena.alloc(Self::build_subtree(arena, &se_particles, se_quad));

        // Calculate total mass and center of mass
        let mut total_mass = 0.0;
        let mut com_x = 0.0;
        let mut com_y = 0.0;

        let mut update_mass_com = |node: &BarnesHutNode<'a>| {
            if let Some((m, cx, cy)) = get_mass_com(node) {
                total_mass += m;
                com_x += cx * m;
                com_y += cy * m;
            }
        };

        update_mass_com(nw_node);
        update_mass_com(ne_node);
        update_mass_com(sw_node);
        update_mass_com(se_node);

        if total_mass > 0.0 {
            com_x /= total_mass;
            com_y /= total_mass;
        }

        // Create and return the internal node
        BarnesHutNode::Internal {
            quad,
            mass: total_mass,
            com_x,
            com_y,
            nw: nw_node,
            ne: ne_node,
            sw: sw_node,
            se: se_node,
        }
    }
}

/// Constructs a Barnes–Hut tree from a slice of particles.
///
/// This function creates a tree with a static lifetime, which is useful
/// for compatibility with code that doesn't use arena allocation.
///
/// # Arguments
///
/// * `particles` - A slice of particles to include in the tree
/// * `quad` - The spatial region the tree will represent
///
/// # Returns
///
/// A new `BarnesHutNode` containing all the particles
///
/// # Examples
///
/// ```
/// use rs_physics::particles::{ParticleData, Quad, build_tree};
///
/// // Create particles
/// let particles = vec![
///     ParticleData { x: 0.1, y: 0.1, mass: 1.0 },
///     ParticleData { x: -0.5, y: 0.5, mass: 2.0 },
/// ];
///
/// // Define the region
/// let quad = Quad { cx: 0.0, cy: 0.0, half_size: 1.0 };
///
/// // Build the tree
/// let tree = build_tree(&particles, quad);
/// ```
///
/// # Note
///
/// This implementation uses a leaked `Box` to create a static arena, so it will
/// leak a small amount of memory. For most applications, prefer using `BarnesHutTree::build`
/// instead.
pub fn build_tree(particles: &[ParticleData], quad: Quad) -> BarnesHutNode<'static> {
    // This adapts the arena-based approach to work with the original interface
    // For tests and compatibility. For actual use, the BarnesHutTree is recommended.
    let arena = Box::leak(Box::new(Bump::new()));
    let tree = BarnesHutTree::build(arena, particles, quad);
    tree.root
}

/// Helper function that extracts the mass and center-of-mass from a BarnesHutNode.
///
/// # Arguments
///
/// * `node` - A reference to a `BarnesHutNode`
///
/// # Returns
///
/// * `Some((mass, com_x, com_y))` for Leaf or Internal nodes
/// * `None` for Empty nodes
///
/// # Examples
///
/// ```
/// use rs_physics::particles::{BarnesHutNode, ParticleData, Quad, get_mass_com};
///
/// // Create a leaf node
/// let quad = Quad { cx: 0.0, cy: 0.0, half_size: 1.0 };
/// let particle = ParticleData { x: 0.5, y: 0.5, mass: 2.0 };
/// let node = BarnesHutNode::Leaf(quad, particle);
///
/// // Get the mass and center of mass
/// if let Some((mass, com_x, com_y)) = get_mass_com(&node) {
///     assert_eq!(mass, 2.0);
///     assert_eq!(com_x, 0.5);
///     assert_eq!(com_y, 0.5);
/// }
///
/// // Empty nodes return None
/// let empty_node = BarnesHutNode::Empty(quad);
/// assert!(get_mass_com(&empty_node).is_none());
/// ```
pub fn get_mass_com(node: &BarnesHutNode) -> Option<(f64, f64, f64)> {
    match node {
        BarnesHutNode::Leaf(_, p) => Some((p.mass, p.x, p.y)),
        BarnesHutNode::Internal { mass, com_x, com_y, .. } => Some((*mass, *com_x, *com_y)),
        BarnesHutNode::Empty(_) => None,
    }
}

/// Helper structure representing an approximated node in the Barnes-Hut algorithm.
///
/// This structure stores the aggregated mass and center of mass for a node that
/// satisfies the Barnes-Hut criterion for a particular target particle.
///
/// # Examples
///
/// ```
/// use rs_physics::particles::ApproxNode;
///
/// // Create an approximated node (e.g., for a distant cluster)
/// let node = ApproxNode {
///     mass: 100.0,
///     com_x: 10.0,
///     com_y: 5.0,
/// };
///
/// assert_eq!(node.mass, 100.0);
/// assert_eq!(node.com_x, 10.0);
/// assert_eq!(node.com_y, 5.0);
/// ```
#[derive(Debug)]
pub struct ApproxNode {
    pub mass: f64,
    pub com_x: f64, // center of mass x-coordinate
    pub com_y: f64, // center of mass y-coordinate
}

/// Recursively collects nodes that satisfy the Barnes–Hut criterion for particle `p`.
///
/// This function traverses the Barnes-Hut tree and collects nodes that are either:
/// - Leaf nodes, or
/// - Internal nodes that are far enough away from the target particle
///   (as determined by the threshold `theta`)
///
/// # Arguments
///
/// * `node` - The Barnes-Hut tree node to start from
/// * `p` - The target particle
/// * `theta` - The Barnes-Hut approximation parameter (typically 0.5-1.0)
/// * `worklist` - A mutable vector to collect the approximated nodes
///
/// # Examples
///
/// ```
/// use rs_physics::particles::{ApproxNode, BarnesHutNode, ParticleData, Quad, build_tree, collect_approx_nodes};
///
/// // Create some particles
/// let particles = vec![
///     ParticleData { x: 1.0, y: 1.0, mass: 1.0 },
///     ParticleData { x: -1.0, y: -1.0, mass: 1.0 },
///     ParticleData { x: 1.0, y: -1.0, mass: 1.0 },
///     ParticleData { x: -1.0, y: 1.0, mass: 1.0 },
/// ];
///
/// // Define the region and build the tree
/// let quad = Quad { cx: 0.0, cy: 0.0, half_size: 2.0 };
/// let tree = build_tree(&particles, quad);
///
/// // Collect approximated nodes for a target particle
/// let target = ParticleData { x: 10.0, y: 10.0, mass: 1.0 };
/// let mut worklist = Vec::new();
/// collect_approx_nodes(&tree, target, 0.5, &mut worklist);
///
/// // Since the target is far from all particles, there should be
/// // just one approximated node (the root)
/// assert_eq!(worklist.len(), 1);
/// assert_eq!(worklist[0].mass, 4.0); // Total mass of all particles
/// ```
pub fn collect_approx_nodes<'a>(
    node: &BarnesHutNode<'a>,
    p: ParticleData,
    theta: f64,
    worklist: &mut Vec<ApproxNode>
) {
    // Pre-allocate the worklist if it's empty
    if worklist.capacity() == 0 {
        // Estimate based on a typical theta value - might need tuning
        let estimated_nodes = (1.0 / theta).powi(2) as usize;
        worklist.reserve(estimated_nodes.min(10000)); // Cap at reasonable max
    }

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
///
/// This is an optimized version of force calculation that uses SIMD instructions
/// to process multiple nodes in parallel.
///
/// # Arguments
///
/// * `p` - The target particle
/// * `worklist` - A slice of approximated nodes
/// * `g` - The gravitational constant
///
/// # Returns
///
/// A tuple of (force_x, force_y) representing the force vector
///
/// # Safety
///
/// This function requires AVX support on the target CPU and should be called only
/// when AVX is available. Use the `std::is_x86_feature_detected!("avx")` macro to check.
///
/// # Examples
///
/// ```
/// use rs_physics::particles::{ApproxNode, ParticleData};
///
/// // Create a worklist of approximated nodes
/// let worklist = vec![
///     ApproxNode { mass: 1.0, com_x: 1.0, com_y: 0.0 },
///     ApproxNode { mass: 1.0, com_x: 0.0, com_y: 1.0 },
///     ApproxNode { mass: 1.0, com_x: -1.0, com_y: 0.0 },
///     ApproxNode { mass: 1.0, com_x: 0.0, com_y: -1.0 },
/// ];
///
/// // Define a target particle and gravitational constant
/// let target = ParticleData { x: 0.0, y: 0.0, mass: 1.0 };
/// let g = 6.67430e-11;
///
/// // Compute force using SIMD if available
/// if std::is_x86_feature_detected!("avx") {
///     unsafe {
///         use rs_physics::particles::compute_force_simd_avx;
///         let (fx, fy) = compute_force_simd_avx(target, &worklist, g);
///
///         // The forces should cancel out
///         assert!(fx.abs() < 1e-10);
///         assert!(fy.abs() < 1e-10);
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
        // Stack-allocated arrays instead of vectors
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

        // Use horizontal addition for better performance
        let sum_low_x = _mm256_extractf128_pd::<0>(force_x_v);
        let sum_high_x = _mm256_extractf128_pd::<1>(force_x_v);
        let sum_x = _mm_add_pd(sum_low_x, sum_high_x);
        let sum_halves_x = _mm_hadd_pd(sum_x, sum_x);
        force_x += _mm_cvtsd_f64(sum_halves_x);

        let sum_low_y = _mm256_extractf128_pd::<0>(force_y_v);
        let sum_high_y = _mm256_extractf128_pd::<1>(force_y_v);
        let sum_y = _mm_add_pd(sum_low_y, sum_high_y);
        let sum_halves_y = _mm_hadd_pd(sum_y, sum_y);
        force_y += _mm_cvtsd_f64(sum_halves_y);

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

/// Low-precision version of SIMD force calculation for large worklists.
///
/// This version uses 32-bit floats instead of 64-bit doubles, which allows
/// processing 8 nodes at once instead of 4, at the cost of some precision.
///
/// # Arguments
///
/// * `p` - The target particle
/// * `worklist` - A slice of approximated nodes
/// * `g` - The gravitational constant (as f32)
///
/// # Returns
///
/// A tuple of (force_x, force_y) representing the force vector
///
/// # Safety
///
/// This function requires AVX support on the target CPU and should be called only
/// when AVX is available. Use the `std::is_x86_feature_detected!("avx")` macro to check.
#[target_feature(enable = "avx")]
pub unsafe fn compute_force_simd_avx_low_precision(p: ParticleData, worklist: &[ApproxNode], g: f32) -> (f32, f32) {
    use std::arch::x86_64::*;
    let mut force_x = 0.0_f32;
    let mut force_y = 0.0_f32;
    let n = worklist.len();
    let mut i = 0;
    while i + 8 <= n {
        // Stack-allocated arrays instead of vectors
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

        // Use optimized horizontal addition
        let sum256_x = _mm256_hadd_ps(force_x_v, force_x_v);
        let sum128_x = _mm_add_ps(_mm256_extractf128_ps::<0>(sum256_x), _mm256_extractf128_ps::<1>(sum256_x));
        let sum64_x = _mm_hadd_ps(sum128_x, sum128_x);
        force_x += _mm_cvtss_f32(sum64_x);

        let sum256_y = _mm256_hadd_ps(force_y_v, force_y_v);
        let sum128_y = _mm_add_ps(_mm256_extractf128_ps::<0>(sum256_y), _mm256_extractf128_ps::<1>(sum256_y));
        let sum64_y = _mm_hadd_ps(sum128_y, sum128_y);
        force_y += _mm_cvtss_f32(sum64_y);

        i += 8;
    }
    // Process remaining nodes in scalar.
    for j in i..n {
        let dx = worklist[j].com_x as f32 - p.x as f32;
        let dy = worklist[j].com_y as f32 - p.y as f32;
        let dist_sq = dx * dx + dy * dy + 1e-12;
        let dist = dist_sq.sqrt();
        let force = g * p.mass as f32 * worklist[j].mass as f32 / dist_sq;
        force_x += force * dx / dist;
        force_y += force * dy / dist;
    }
    (force_x, force_y)
}

/// Scalar fallback function to compute net force from a worklist.
///
/// This function is used when SIMD instructions are not available.
///
/// # Arguments
///
/// * `p` - The target particle
/// * `worklist` - A slice of approximated nodes
/// * `g` - The gravitational constant
///
/// # Returns
///
/// A tuple of (force_x, force_y) representing the force vector
///
/// # Examples
///
/// ```
/// use rs_physics::particles::{ApproxNode, ParticleData, compute_force_scalar};
///
/// // Create a worklist of approximated nodes
/// let worklist = vec![
///     ApproxNode { mass: 1.0, com_x: 1.0, com_y: 0.0 },
///     ApproxNode { mass: 1.0, com_x: 0.0, com_y: 1.0 },
/// ];
///
/// // Define a target particle and gravitational constant
/// let target = ParticleData { x: 0.0, y: 0.0, mass: 1.0 };
/// let g = 6.67430e-11;
///
/// // Compute force
/// let (fx, fy) = compute_force_scalar(target, &worklist, g);
///
/// // The force should be in the direction of the sum of the nodes
/// assert!(fx > 0.0);
/// assert!(fy > 0.0);
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
///
/// This function first collects a worklist of approximated nodes using the Barnes-Hut
/// algorithm, then computes the net force using the most efficient available method
/// (SIMD or scalar).
///
/// # Arguments
///
/// * `tree` - The Barnes-Hut tree
/// * `p` - The target particle
/// * `theta` - The Barnes-Hut approximation parameter (typically 0.5-1.0)
/// * `g` - The gravitational constant
///
/// # Returns
///
/// A tuple of (force_x, force_y) representing the force vector
///
/// # Examples
///
/// ```
/// use rs_physics::particles::{BarnesHutTree, ParticleData, Quad, compute_net_force};
/// use bumpalo::Bump;
///
/// // Create an arena and some particles
/// let arena = Bump::new();
/// let particles = vec![
///     ParticleData { x: 1.0, y: 0.0, mass: 1.0 },
///     ParticleData { x: -1.0, y: 0.0, mass: 1.0 },
/// ];
///
/// // Define the region and build the tree
/// let quad = Quad { cx: 0.0, cy: 0.0, half_size: 2.0 };
/// let tree = BarnesHutTree::build(&arena, &particles, quad);
///
/// // Compute force on a test particle
/// let test_particle = ParticleData { x: 0.0, y: 1.0, mass: 1.0 };
/// let (fx, fy) = compute_net_force(&tree, test_particle, 0.5, 6.67430e-11);
///
/// // The x-component should be approximately zero
/// assert!(fx.abs() < 1e-10);
/// // The y-component should be negative (downward)
/// assert!(fy < 0.0);
/// ```
pub fn compute_net_force(tree: &BarnesHutTree, p: ParticleData, theta: f64, g: f64) -> (f64, f64) {
    // Pre-allocate worklist with reasonable capacity
    let mut worklist = Vec::with_capacity(1000);
    collect_approx_nodes(&tree.root, p, theta, &mut worklist);

    // Use SIMD if available
    if std::is_x86_feature_detected!("avx") {
        // For large worklists, use lower precision for better performance
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

/// Create a new Barnes-Hut tree from a slice of particles.
///
/// This is a convenience function to build a Barnes-Hut tree using arena allocation.
///
/// # Arguments
///
/// * `arena` - A reference to a bump allocator
/// * `particles` - A slice of particles to include in the tree
/// * `quad` - The spatial region the tree will represent
///
/// # Returns
///
/// A new `BarnesHutTree` containing all the particles
///
/// # Examples
///
/// ```
/// use rs_physics::particles::{ParticleData, Quad, build_barnes_hut_tree};
/// use bumpalo::Bump;
///
/// // Create an arena and some particles
/// let arena = Bump::new();
/// let particles = vec![
///     ParticleData { x: 0.1, y: 0.1, mass: 1.0 },
///     ParticleData { x: -0.5, y: 0.5, mass: 2.0 },
/// ];
///
/// // Define the region
/// let quad = Quad { cx: 0.0, cy: 0.0, half_size: 1.0 };
///
/// // Build the tree
/// let tree = build_barnes_hut_tree(&arena, &particles, quad);
/// ```
pub fn build_barnes_hut_tree<'a>(arena: &'a Bump, particles: &[ParticleData], quad: Quad) -> BarnesHutTree<'a> {
    BarnesHutTree::build(arena, particles, quad)
}

/// Calculates gravitational forces on a set of particles using the Barnes-Hut algorithm.
///
/// This function creates a Barnes-Hut tree, then computes the forces on all particles
/// using the tree. It's a convenient high-level function for performing n-body simulations.
///
/// # Arguments
///
/// * `particles` - A slice of particles in the simulation
/// * `quad` - The spatial region containing all particles
/// * `g` - The gravitational constant
/// * `theta` - The Barnes-Hut approximation parameter (typically 0.5-1.0)
///
/// # Returns
///
/// A vector of (force_x, force_y) pairs, one for each particle
///
/// # Examples
///
/// ```
/// use rs_physics::particles::{ParticleData, Quad, simulate_particles};
///
/// // Create particles
/// let particles = vec![
///     ParticleData { x: 0.0, y: 0.0, mass: 1.0 },
///     ParticleData { x: 1.0, y: 0.0, mass: 1.0 },
/// ];
///
/// // Define the simulation region
/// let quad = Quad { cx: 0.0, cy: 0.0, half_size: 2.0 };
///
/// // Gravitational constant and approximation parameter
/// let g = 6.67430e-11;
/// let theta = 0.5;
///
/// // Calculate forces on all particles
/// let forces = simulate_particles(&particles, quad, g, theta);
///
/// // Check that forces are equal and opposite
/// assert_eq!(forces.len(), 2);
/// assert!((forces[0].0 + forces[1].0).abs() < 1e-10);
/// assert!((forces[0].1 + forces[1].1).abs() < 1e-10);
/// ```
pub fn simulate_particles(particles: &[ParticleData], quad: Quad, g: f64, theta: f64) -> Vec<(f64, f64)> {
    // Create a bump allocator with 4MB capacity (adjust based on particle count)
    let arena = Bump::with_capacity(4 * 1024 * 1024);

    // Build the Barnes-Hut tree
    let tree = build_barnes_hut_tree(&arena, particles, quad);

    // Calculate forces on all particles
    let forces: Vec<(f64, f64)> = match particles.is_empty() {
        true => vec![(0.0, 0.0)],
        false => particles.iter()
            .map(|&p| compute_net_force(&tree, p, theta, g))
            .collect()
    };

    // The arena is automatically freed when it goes out of scope
    forces
}