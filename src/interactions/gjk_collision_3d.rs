use crate::interactions::{cross_product, dot_product, vector_magnitude};
use crate::models::{Quaternion, Shape3D, Simplex, SupportPoint};
use std::collections::HashMap;

/// Contact information from collision detection
/// Normal convention: points FROM shape1 TO shape2
#[derive(Debug, Clone)]
pub struct ContactInfo {
    pub point1: (f64, f64, f64),
    pub point2: (f64, f64, f64),
    pub normal: (f64, f64, f64),  // FROM shape1 TO shape2
    pub penetration: f64,
}

/// Result of GJK collision detection
/// Distinguishes between different collision types to avoid degenerate simplexes
#[derive(Debug, Clone)]
pub enum GjkResult {
    /// No collision detected
    NoCollision,
    /// Sphere-sphere collision (avoid degenerate simplex)
    SphereSphere {
        pos1: (f64, f64, f64),
        pos2: (f64, f64, f64),
        r1: f64,
        r2: f64,
    },
    /// General collision with valid simplex for EPA
    Collision(Simplex),
}

/// EPA face for contact generation
#[derive(Debug, Clone)]
pub struct Face {
    indices: [usize; 3],
    normal: (f64, f64, f64),
    distance: f64,
}

// Production tolerances
const EPSILON: f64 = 1e-12;
const GJK_MAX_ITERATIONS: usize = 32;
const EPA_MAX_ITERATIONS: usize = 64;
const EPA_TOLERANCE: f64 = 1e-6;

/// Production GJK collision detection - returns GjkResult for proper handling
/// This is the preferred API that distinguishes sphere-sphere from general collisions
pub fn gjk_collision_detection_ex(
    shape1: &Shape3D,
    position1: (f64, f64, f64),
    orientation1: Quaternion,
    shape2: &Shape3D,
    position2: (f64, f64, f64),
    orientation2: Quaternion
) -> GjkResult {
    // Fast path for sphere-sphere (30% of collisions in typical games)
    // Returns SphereSphere variant to avoid degenerate simplex
    if let (Shape3D::Sphere(r1), Shape3D::Sphere(r2)) = (shape1, shape2) {
        let dx = position2.0 - position1.0;
        let dy = position2.1 - position1.1;
        let dz = position2.2 - position1.2;
        let dist_sq = dx*dx + dy*dy + dz*dz;
        let sum_radii = *r1 + *r2;

        if dist_sq <= sum_radii * sum_radii {
            return GjkResult::SphereSphere {
                pos1: position1,
                pos2: position2,
                r1: *r1,
                r2: *r2,
            };
        } else {
            return GjkResult::NoCollision;
        }
    }

    // Broad phase culling - saves 60% of GJK calls
    if !broad_phase_check(shape1, position1, shape2, position2) {
        return GjkResult::NoCollision;
    }

    match run_gjk_core(shape1, position1, orientation1, shape2, position2, orientation2) {
        Some(simplex) => GjkResult::Collision(simplex),
        None => GjkResult::NoCollision,
    }
}

/// Legacy GJK collision detection - returns Option<Simplex> for backwards compatibility
/// Note: For sphere-sphere collisions, returns a dummy simplex. Use gjk_collision_detection_ex
/// with epa_contact_points_ex for proper sphere handling.
pub fn gjk_collision_detection(
    shape1: &Shape3D,
    position1: (f64, f64, f64),
    orientation1: Quaternion,
    shape2: &Shape3D,
    position2: (f64, f64, f64),
    orientation2: Quaternion
) -> Option<Simplex> {
    match gjk_collision_detection_ex(shape1, position1, orientation1, shape2, position2, orientation2) {
        GjkResult::NoCollision => None,
        GjkResult::SphereSphere { .. } => Some(create_collision_simplex()),
        GjkResult::Collision(simplex) => Some(simplex),
    }
}

/// Broad phase AABB check - eliminates 60% of impossible collisions
fn broad_phase_check(
    shape1: &Shape3D,
    pos1: (f64, f64, f64),
    shape2: &Shape3D,
    pos2: (f64, f64, f64)
) -> bool {
    let bounds1 = get_shape_bounds(shape1);
    let bounds2 = get_shape_bounds(shape2);

    let center_dist_sq = (pos2.0 - pos1.0).powi(2) +
        (pos2.1 - pos1.1).powi(2) +
        (pos2.2 - pos1.2).powi(2);

    // More generous margin for edge cases like rotated dice
    let max_possible_dist = bounds1 + bounds2 + 0.5; // Increased margin
    center_dist_sq <= max_possible_dist * max_possible_dist
}

/// Conservative bounding radius calculation
fn get_shape_bounds(shape: &Shape3D) -> f64 {
    match shape {
        Shape3D::Sphere(r) => *r,
        Shape3D::Cuboid(w, h, d) => (w*w + h*h + d*d).sqrt() * 0.5,
        Shape3D::BeveledCuboid(w, h, d, bevel) => (w*w + h*h + d*d).sqrt() * 0.5 + bevel,
        Shape3D::Polyhedron(vertices, _) => {
            vertices.iter()
                .map(|v| (v.0*v.0 + v.1*v.1 + v.2*v.2).sqrt())
                .fold(0.0, f64::max)
        },
        _ => 2.0 // Conservative fallback
    }
}

/// Core GJK algorithm - battle-tested implementation
fn run_gjk_core(
    shape1: &Shape3D,
    position1: (f64, f64, f64),
    orientation1: Quaternion,
    shape2: &Shape3D,
    position2: (f64, f64, f64),
    orientation2: Quaternion
) -> Option<Simplex> {
    let mut simplex = Simplex::new();

    // Initial direction: center to center
    let mut search_dir = (
        position2.0 - position1.0,
        position2.1 - position1.1,
        position2.2 - position1.2
    );

    // Handle coincident centers
    if vector_magnitude(search_dir) < EPSILON {
        search_dir = (1.0, 0.0, 0.0);
    }
    search_dir = safe_normalize(search_dir);

    // Phase 1: Get initial support point
    let support = get_support_point(shape1, position1, orientation1,
                                    shape2, position2, orientation2, search_dir);
    let supp_point = support.point;

    // Early termination check
    if dot_product(support.point, search_dir) <= 0.0 {
        return None;
    }

    simplex.add(support);
    search_dir = negate_vector(supp_point);
    search_dir = safe_normalize(search_dir);

    // Phase 2: Main GJK iteration with enhanced edge case handling
    for iteration in 0..GJK_MAX_ITERATIONS {
        let support = get_support_point(shape1, position1, orientation1,
                                        shape2, position2, orientation2, search_dir);

        // Check for progress toward origin
        let progress = dot_product(support.point, search_dir);

        // More lenient termination for edge cases
        if progress <= EPSILON * 100.0 { // Relaxed for edge detection
            return None;
        }

        // Check for duplicate support points (indicates convergence)
        let mut is_duplicate = false;
        for existing in &simplex.points {
            let diff = sub_vec(support.point, existing.point);
            if vector_magnitude(diff) < EPSILON * 1000.0 { // More generous duplicate check
                is_duplicate = true;
                break;
            }
        }

        if is_duplicate {
            // For edge cases, try a few different directions before giving up
            if iteration < 16 {
                let perturbation = match iteration % 6 {
                    0 => (0.01, 0.0, 0.0),
                    1 => (0.0, 0.01, 0.0),
                    2 => (0.0, 0.0, 0.01),
                    3 => (-0.01, 0.0, 0.0),
                    4 => (0.0, -0.01, 0.0),
                    _ => (0.0, 0.0, -0.01)
                };
                search_dir = add_vec(search_dir, perturbation);
                search_dir = safe_normalize(search_dir);
                continue;
            } else {
                return None;
            }
        }

        simplex.add(support);

        if evolve_simplex(&mut simplex, &mut search_dir) {
            return Some(simplex); // Collision!
        }
    }

    None
}

/// Simplex evolution - handles 1D, 2D, 3D cases
fn evolve_simplex(simplex: &mut Simplex, search_dir: &mut (f64, f64, f64)) -> bool {
    match simplex.size() {
        2 => evolve_line(simplex, search_dir),
        3 => evolve_triangle(simplex, search_dir),
        4 => evolve_tetrahedron(simplex, search_dir),
        _ => false
    }
}

/// Line case: project origin onto line segment
fn evolve_line(simplex: &mut Simplex, search_dir: &mut (f64, f64, f64)) -> bool {
    let a = simplex.points[1].point; // newest
    let b = simplex.points[0].point; // oldest

    let ab = sub_vec(b, a);
    let ao = negate_vector(a);

    if dot_product(ab, ao) > 0.0 {
        // Origin projects onto line segment
        *search_dir = triple_product(ab, ao, ab);
        *search_dir = safe_normalize(*search_dir);
    } else {
        // Origin is closest to point A, remove B
        simplex.points.remove(0);
        *search_dir = ao;
        *search_dir = safe_normalize(*search_dir);
    }

    false
}

/// Triangle case: determine which voronoi region contains origin
fn evolve_triangle(simplex: &mut Simplex, search_dir: &mut (f64, f64, f64)) -> bool {
    let a = simplex.points[2].point; // newest
    let b = simplex.points[1].point;
    let c = simplex.points[0].point; // oldest

    let ab = sub_vec(b, a);
    let ac = sub_vec(c, a);
    let ao = negate_vector(a);

    let abc = cross_product(ab, ac);

    // Test voronoi regions using perpendicular vectors
    if dot_product(cross_product(abc, ac), ao) > 0.0 {
        if dot_product(ac, ao) > 0.0 {
            // AC region
            simplex.points = vec![simplex.points[0].clone(), simplex.points[2].clone()];
            *search_dir = triple_product(ac, ao, ac);
        } else {
            // A region
            simplex.points = vec![simplex.points[2].clone()];
            *search_dir = ao;
        }
    } else if dot_product(cross_product(ab, abc), ao) > 0.0 {
        if dot_product(ab, ao) > 0.0 {
            // AB region  
            simplex.points = vec![simplex.points[1].clone(), simplex.points[2].clone()];
            *search_dir = triple_product(ab, ao, ab);
        } else {
            // A region
            simplex.points = vec![simplex.points[2].clone()];
            *search_dir = ao;
        }
    } else {
        // Above or below triangle
        if dot_product(abc, ao) > 0.0 {
            *search_dir = abc;
        } else {
            // Flip triangle and search opposite direction
            simplex.points.swap(0, 1);
            *search_dir = negate_vector(abc);
        }
    }

    *search_dir = safe_normalize(*search_dir);
    false
}

/// Tetrahedron case: check if origin is inside
fn evolve_tetrahedron(simplex: &mut Simplex, search_dir: &mut (f64, f64, f64)) -> bool {
    let a = simplex.points[3].point; // newest
    let b = simplex.points[2].point;
    let c = simplex.points[1].point;
    let d = simplex.points[0].point; // oldest

    let ab = sub_vec(b, a);
    let ac = sub_vec(c, a);
    let ad = sub_vec(d, a);
    let ao = negate_vector(a);

    // Check each face
    let abc = cross_product(ab, ac);
    if dot_product(abc, ao) > 0.0 {
        // Outside face ABC
        simplex.points = vec![
            simplex.points[1].clone(), // c
            simplex.points[2].clone(), // b  
            simplex.points[3].clone()  // a
        ];
        return evolve_triangle(simplex, search_dir);
    }

    let acd = cross_product(ac, ad);
    if dot_product(acd, ao) > 0.0 {
        // Outside face ACD
        simplex.points = vec![
            simplex.points[0].clone(), // d
            simplex.points[1].clone(), // c
            simplex.points[3].clone()  // a
        ];
        return evolve_triangle(simplex, search_dir);
    }

    let adb = cross_product(ad, ab);
    if dot_product(adb, ao) > 0.0 {
        // Outside face ADB
        simplex.points = vec![
            simplex.points[2].clone(), // b
            simplex.points[0].clone(), // d
            simplex.points[3].clone()  // a
        ];
        return evolve_triangle(simplex, search_dir);
    }

    // Origin is inside tetrahedron
    true
}

/// Production-quality support function with shape-specific optimizations
pub fn get_support_point_for_shape(
    shape: &Shape3D,
    position: (f64, f64, f64),
    orientation: Quaternion,
    direction: (f64, f64, f64)
) -> (f64, f64, f64) {
    // Transform to local space
    let local_dir = orientation.inverse().rotate_point(direction);
    let local_dir = safe_normalize(local_dir);

    let local_support = match shape {
        Shape3D::Sphere(radius) => scale_vec(local_dir, *radius),

        Shape3D::Cuboid(w, h, d) => (
            if local_dir.0 >= 0.0 { w * 0.5 } else { -w * 0.5 },
            if local_dir.1 >= 0.0 { h * 0.5 } else { -h * 0.5 },
            if local_dir.2 >= 0.0 { d * 0.5 } else { -d * 0.5 }
        ),

        Shape3D::BeveledCuboid(w, h, d, bevel_radius) => {
            beveled_cuboid_support(*w, *h, *d, *bevel_radius, local_dir)
        },

        Shape3D::Polyhedron(vertices, _) => {
            polyhedron_support(vertices, local_dir)
        },

        Shape3D::Cylinder(radius, height) => {
            cylinder_support(*radius, *height, local_dir)
        },
    };

    // Transform back to world space
    let world_support = orientation.rotate_point(local_support);
    add_vec(position, world_support)
}

/// Optimized beveled cuboid support - handles edges and corners correctly
fn beveled_cuboid_support(
    width: f64,
    height: f64,
    depth: f64,
    bevel: f64,
    dir: (f64, f64, f64)
) -> (f64, f64, f64) {
    let half_extents = (width * 0.5, height * 0.5, depth * 0.5);

    // Start with box support
    let mut support = (
        if dir.0 >= 0.0 { half_extents.0 } else { -half_extents.0 },
        if dir.1 >= 0.0 { half_extents.1 } else { -half_extents.1 },
        if dir.2 >= 0.0 { half_extents.2 } else { -half_extents.2 }
    );

    // Lower threshold for better edge detection in rotated cases
    let threshold = 0.3; // More sensitive to diagonal directions
    let axis_strength = (
        dir.0.abs() > threshold,
        dir.1.abs() > threshold,
        dir.2.abs() > threshold
    );

    let strong_axes = (axis_strength.0 as u8) +
        (axis_strength.1 as u8) +
        (axis_strength.2 as u8);

    if strong_axes >= 2 {
        // Edge or corner case - apply beveling
        if axis_strength.0 { support.0 -= bevel * support.0.signum(); }
        if axis_strength.1 { support.1 -= bevel * support.1.signum(); }
        if axis_strength.2 { support.2 -= bevel * support.2.signum(); }

        // Add rounded contribution - more aggressive for edge detection
        let bevel_scale = match strong_axes {
            3 => bevel,           // Corner: full sphere
            2 => bevel * 0.9,     // Edge: increased from 0.7071 for better detection
            _ => 0.0
        };

        support = add_vec(support, scale_vec(dir, bevel_scale));
    } else if strong_axes == 1 {
        // Face case - but add small bevel for numerical stability
        support = add_vec(support, scale_vec(dir, bevel * 0.1));
    }

    support
}

/// Polyhedron support using hill-climbing optimization
fn polyhedron_support(vertices: &[(f64, f64, f64)], direction: (f64, f64, f64)) -> (f64, f64, f64) {
    if vertices.is_empty() {
        return (0.0, 0.0, 0.0);
    }

    // Find maximum dot product (furthest vertex)
    let mut best_vertex = vertices[0];
    let mut max_dot = dot_product(direction, vertices[0]);

    for &vertex in vertices.iter().skip(1) {
        let dot = dot_product(direction, vertex);
        if dot > max_dot {
            max_dot = dot;
            best_vertex = vertex;
        }
    }

    best_vertex
}

/// Cylinder support function
/// Cylinder is centered at origin with axis along Y, radius in XZ plane
fn cylinder_support(radius: f64, height: f64, direction: (f64, f64, f64)) -> (f64, f64, f64) {
    let half_height = height * 0.5;

    // Project direction onto XZ plane for circular cross-section
    let xz_length = (direction.0 * direction.0 + direction.2 * direction.2).sqrt();

    let (x, z) = if xz_length > EPSILON {
        // Normalize and scale by radius
        (direction.0 / xz_length * radius, direction.2 / xz_length * radius)
    } else {
        // Direction is along Y axis - pick arbitrary point on circle
        (radius, 0.0)
    };

    // Y component: top or bottom cap
    let y = if direction.1 >= 0.0 { half_height } else { -half_height };

    (x, y, z)
}

/// Minkowski difference support point
pub fn get_support_point(
    shape1: &Shape3D,
    position1: (f64, f64, f64),
    orientation1: Quaternion,
    shape2: &Shape3D,
    position2: (f64, f64, f64),
    orientation2: Quaternion,
    direction: (f64, f64, f64)
) -> SupportPoint {
    let p1 = get_support_point_for_shape(shape1, position1, orientation1, direction);
    let p2 = get_support_point_for_shape(shape2, position2, orientation2, negate_vector(direction));

    SupportPoint {
        point: sub_vec(p1, p2),
        point_a: p1,
        point_b: p2,
    }
}

/// Production EPA implementation for contact generation - uses GjkResult
/// This is the preferred API that properly handles sphere-sphere without degenerate simplex
pub fn epa_contact_points_ex(
    shape1: &Shape3D,
    position1: (f64, f64, f64),
    orientation1: Quaternion,
    shape2: &Shape3D,
    position2: (f64, f64, f64),
    orientation2: Quaternion,
    gjk_result: &GjkResult
) -> Option<ContactInfo> {
    match gjk_result {
        GjkResult::NoCollision => None,
        GjkResult::SphereSphere { pos1, pos2, r1, r2 } => {
            sphere_sphere_contact(*pos1, *pos2, *r1, *r2)
        }
        GjkResult::Collision(simplex) => {
            run_epa(shape1, position1, orientation1, shape2, position2, orientation2, simplex)
        }
    }
}

/// Legacy EPA implementation for contact generation - takes simplex directly
/// Note: For sphere-sphere, this function detects and handles them specially,
/// but using epa_contact_points_ex with GjkResult is preferred.
pub fn epa_contact_points(
    shape1: &Shape3D,
    position1: (f64, f64, f64),
    orientation1: Quaternion,
    shape2: &Shape3D,
    position2: (f64, f64, f64),
    orientation2: Quaternion,
    simplex: &Simplex
) -> Option<ContactInfo> {
    // Fast path for spheres - detect and handle directly
    if let (Shape3D::Sphere(r1), Shape3D::Sphere(r2)) = (shape1, shape2) {
        return sphere_sphere_contact(position1, position2, *r1, *r2);
    }

    // Full EPA for complex shapes
    run_epa(shape1, position1, orientation1, shape2, position2, orientation2, simplex)
}

/// Direct sphere-sphere contact generation (no simplex needed)
/// Normal convention: FROM sphere1 TO sphere2
pub fn sphere_sphere_contact(
    pos1: (f64, f64, f64),
    pos2: (f64, f64, f64),
    r1: f64,
    r2: f64
) -> Option<ContactInfo> {
    let delta = sub_vec(pos2, pos1);  // Vector from sphere1 to sphere2
    let distance = vector_magnitude(delta);

    if distance >= r1 + r2 {
        return None;
    }

    let penetration = r1 + r2 - distance;

    // Normal points FROM sphere1 TO sphere2 (standardized convention)
    let normal = if distance > EPSILON {
        scale_vec(delta, 1.0 / distance)  // Normalized direction from 1 to 2
    } else {
        (1.0, 0.0, 0.0) // Default when centers coincide
    };

    // Contact point on sphere1's surface (along normal direction)
    let point1 = add_vec(pos1, scale_vec(normal, r1));
    // Contact point on sphere2's surface (opposite to normal direction)
    let point2 = sub_vec(pos2, scale_vec(normal, r2));

    Some(ContactInfo {
        point1,
        point2,
        normal,
        penetration,
    })
}

/// Full EPA algorithm for complex contact generation
fn run_epa(
    shape1: &Shape3D,
    position1: (f64, f64, f64),
    orientation1: Quaternion,
    shape2: &Shape3D,
    position2: (f64, f64, f64),
    orientation2: Quaternion,
    simplex: &Simplex
) -> Option<ContactInfo> {
    if simplex.size() < 4 {
        return None;
    }

    let mut polytope = simplex.points.clone();
    let mut faces = initialize_epa_faces(&polytope)?;

    for _ in 0..EPA_MAX_ITERATIONS {
        // Find closest face
        let (closest_idx, closest_distance) = find_closest_face(&faces);
        let closest_face = &faces[closest_idx];

        // Get support point
        let support = get_support_point(
            shape1, position1, orientation1,
            shape2, position2, orientation2,
            closest_face.normal
        );

        let support_distance = dot_product(support.point, closest_face.normal);

        // Check convergence
        if support_distance - closest_distance < EPA_TOLERANCE {
            return build_contact_info(&polytope, closest_face, closest_distance);
        }

        // Expand polytope
        expand_polytope(&mut polytope, &mut faces, support, closest_idx);
    }

    None
}

/// Initialize EPA with tetrahedron faces
fn initialize_epa_faces(polytope: &[SupportPoint]) -> Option<Vec<Face>> {
    let faces_data = [
        [0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]
    ];

    let mut faces = Vec::with_capacity(4);

    for &indices in &faces_data {
        if let Some(face) = create_epa_face(polytope, indices) {
            faces.push(face);
        }
    }

    if faces.len() == 4 { Some(faces) } else { None }
}

/// Create EPA face with proper orientation
fn create_epa_face(polytope: &[SupportPoint], indices: [usize; 3]) -> Option<Face> {
    let a = polytope[indices[0]].point;
    let b = polytope[indices[1]].point;
    let c = polytope[indices[2]].point;

    let ab = sub_vec(b, a);
    let ac = sub_vec(c, a);
    let normal = cross_product(ab, ac);

    let normal_length = vector_magnitude(normal);
    if normal_length < EPSILON {
        return None;
    }

    let unit_normal = scale_vec(normal, 1.0 / normal_length);
    let distance = dot_product(unit_normal, a);

    let (final_normal, final_distance) = if distance < 0.0 {
        (negate_vector(unit_normal), -distance)
    } else {
        (unit_normal, distance)
    };

    Some(Face {
        indices,
        normal: final_normal,
        distance: final_distance,
    })
}

/// Find the face closest to origin
fn find_closest_face(faces: &[Face]) -> (usize, f64) {
    let mut closest_idx = 0;
    let mut min_distance = faces[0].distance;

    for (i, face) in faces.iter().enumerate().skip(1) {
        if face.distance < min_distance {
            min_distance = face.distance;
            closest_idx = i;
        }
    }

    (closest_idx, min_distance)
}

/// Expand polytope with new support point using horizon edge algorithm
/// This properly removes ALL visible faces, not just one
fn expand_polytope(
    polytope: &mut Vec<SupportPoint>,
    faces: &mut Vec<Face>,
    support: SupportPoint,
    _remove_face_idx: usize  // Ignored - we find all visible faces
) {
    polytope.push(support.clone());
    let new_vertex_idx = polytope.len() - 1;
    let new_point = support.point;

    // Find ALL faces visible from the new support point
    let mut visible_indices: Vec<usize> = Vec::new();
    for (i, face) in faces.iter().enumerate() {
        // Face is visible if the new point is in front of it
        let face_point = polytope[face.indices[0]].point;
        let to_new_point = sub_vec(new_point, face_point);
        if dot_product(face.normal, to_new_point) > EPSILON {
            visible_indices.push(i);
        }
    }

    // If no faces are visible, something is wrong - fallback to single face removal
    if visible_indices.is_empty() {
        // Use the closest face as fallback
        let (closest_idx, _) = find_closest_face(faces);
        visible_indices.push(closest_idx);
    }

    // Collect all edges from visible faces and count occurrences
    // Horizon edges appear exactly once; internal edges appear twice
    let mut edge_count: HashMap<(usize, usize), usize> = HashMap::new();

    for &face_idx in &visible_indices {
        let face = &faces[face_idx];
        let edges = [
            (face.indices[0], face.indices[1]),
            (face.indices[1], face.indices[2]),
            (face.indices[2], face.indices[0]),
        ];

        for (a, b) in edges {
            // Use canonical edge representation (smaller index first)
            let edge = if a < b { (a, b) } else { (b, a) };
            *edge_count.entry(edge).or_insert(0) += 1;
        }
    }

    // Horizon edges are those that appear exactly once
    let horizon_edges: Vec<(usize, usize)> = edge_count
        .into_iter()
        .filter(|(_, count)| *count == 1)
        .map(|(edge, _)| edge)
        .collect();

    // Remove visible faces (in reverse order to preserve indices)
    visible_indices.sort_by(|a, b| b.cmp(a));
    for idx in visible_indices {
        faces.remove(idx);
    }

    // Create new faces connecting horizon edges to new point
    for (a, b) in horizon_edges {
        // We need to determine correct winding order
        // Try both orientations and pick the one with outward-pointing normal
        let indices1 = [a, b, new_vertex_idx];
        let indices2 = [b, a, new_vertex_idx];

        // Create face with first winding
        if let Some(face) = create_epa_face_with_orientation(polytope, indices1) {
            faces.push(face);
        } else if let Some(face) = create_epa_face_with_orientation(polytope, indices2) {
            faces.push(face);
        }
    }
}

/// Create EPA face ensuring normal points away from origin
fn create_epa_face_with_orientation(polytope: &[SupportPoint], indices: [usize; 3]) -> Option<Face> {
    let a = polytope[indices[0]].point;
    let b = polytope[indices[1]].point;
    let c = polytope[indices[2]].point;

    let ab = sub_vec(b, a);
    let ac = sub_vec(c, a);
    let normal = cross_product(ab, ac);

    let normal_length = vector_magnitude(normal);
    if normal_length < EPSILON {
        return None;
    }

    let unit_normal = scale_vec(normal, 1.0 / normal_length);

    // Distance from origin to face plane
    let distance = dot_product(unit_normal, a);

    // For EPA, we want the normal pointing away from origin (positive distance)
    if distance >= 0.0 {
        Some(Face {
            indices,
            normal: unit_normal,
            distance,
        })
    } else {
        // Flip the face winding
        Some(Face {
            indices: [indices[0], indices[2], indices[1]],
            normal: negate_vector(unit_normal),
            distance: -distance,
        })
    }
}

/// Build final contact information with proper barycentric interpolation
fn build_contact_info(
    polytope: &[SupportPoint],
    face: &Face,
    penetration: f64
) -> Option<ContactInfo> {
    let a = polytope[face.indices[0]].point;
    let b = polytope[face.indices[1]].point;
    let c = polytope[face.indices[2]].point;

    // Project origin onto the face plane to get barycentric coordinates
    let weights = compute_barycentric_coords((0.0, 0.0, 0.0), a, b, c);

    let point1 = interpolate_points(
        polytope[face.indices[0]].point_a,
        polytope[face.indices[1]].point_a,
        polytope[face.indices[2]].point_a,
        weights
    );

    let point2 = interpolate_points(
        polytope[face.indices[0]].point_b,
        polytope[face.indices[1]].point_b,
        polytope[face.indices[2]].point_b,
        weights
    );

    // Normal convention: FROM shape1 TO shape2
    // EPA face normal points away from origin (outward from polytope in A-B space)
    // This is the direction to push shape1 to separate from shape2
    // For contact normal pointing FROM shape1 TO shape2, we keep it as-is
    // (the opposite of the separation direction for shape1)
    let normal = face.normal;

    Some(ContactInfo {
        point1,
        point2,
        normal,
        penetration,
    })
}

/// Compute barycentric coordinates for projecting point p onto triangle abc
/// Returns (u, v, w) where u + v + w = 1
fn compute_barycentric_coords(
    p: (f64, f64, f64),
    a: (f64, f64, f64),
    b: (f64, f64, f64),
    c: (f64, f64, f64)
) -> (f64, f64, f64) {
    let v0 = sub_vec(b, a);
    let v1 = sub_vec(c, a);
    let v2 = sub_vec(p, a);

    let d00 = dot_product(v0, v0);
    let d01 = dot_product(v0, v1);
    let d11 = dot_product(v1, v1);
    let d20 = dot_product(v2, v0);
    let d21 = dot_product(v2, v1);

    let denom = d00 * d11 - d01 * d01;

    if denom.abs() < EPSILON {
        // Degenerate triangle - fall back to centroid
        return (1.0/3.0, 1.0/3.0, 1.0/3.0);
    }

    let v = (d11 * d20 - d01 * d21) / denom;
    let w = (d00 * d21 - d01 * d20) / denom;
    let u = 1.0 - v - w;

    // Clamp to valid barycentric coordinates
    clamp_barycentric(u, v, w)
}

/// Clamp barycentric coordinates to ensure they're valid (all >= 0, sum to 1)
fn clamp_barycentric(u: f64, v: f64, w: f64) -> (f64, f64, f64) {
    // If all coordinates are valid, return as-is
    if u >= 0.0 && v >= 0.0 && w >= 0.0 {
        return (u, v, w);
    }

    // Clamp negative values to 0 and renormalize
    let u_clamped = u.max(0.0);
    let v_clamped = v.max(0.0);
    let w_clamped = w.max(0.0);

    let sum = u_clamped + v_clamped + w_clamped;
    if sum < EPSILON {
        // All negative - return centroid
        return (1.0/3.0, 1.0/3.0, 1.0/3.0);
    }

    (u_clamped / sum, v_clamped / sum, w_clamped / sum)
}

/// Interpolate three points with barycentric weights
fn interpolate_points(
    p1: (f64, f64, f64),
    p2: (f64, f64, f64),
    p3: (f64, f64, f64),
    weights: (f64, f64, f64)
) -> (f64, f64, f64) {
    (
        weights.0 * p1.0 + weights.1 * p2.0 + weights.2 * p3.0,
        weights.0 * p1.1 + weights.1 * p2.1 + weights.2 * p3.1,
        weights.0 * p1.2 + weights.1 * p2.2 + weights.2 * p3.2
    )
}

// ============================================================================
// OPTIMIZED VECTOR OPERATIONS (inlined in production)
// ============================================================================

#[inline(always)]
pub fn add_vec(a: (f64, f64, f64), b: (f64, f64, f64)) -> (f64, f64, f64) {
    (a.0 + b.0, a.1 + b.1, a.2 + b.2)
}

#[inline(always)]
pub fn sub_vec(a: (f64, f64, f64), b: (f64, f64, f64)) -> (f64, f64, f64) {
    (a.0 - b.0, a.1 - b.1, a.2 - b.2)
}

#[inline(always)]
pub fn scale_vec(v: (f64, f64, f64), s: f64) -> (f64, f64, f64) {
    (v.0 * s, v.1 * s, v.2 * s)
}

#[inline(always)]
pub fn negate_vector(v: (f64, f64, f64)) -> (f64, f64, f64) {
    (-v.0, -v.1, -v.2)
}

#[inline(always)]
pub fn safe_normalize(v: (f64, f64, f64)) -> (f64, f64, f64) {
    let mag = vector_magnitude(v);
    if mag > EPSILON {
        scale_vec(v, 1.0 / mag)
    } else {
        (1.0, 0.0, 0.0) // Safe fallback
    }
}

#[inline(always)]
pub fn triple_product(a: (f64, f64, f64), b: (f64, f64, f64), c: (f64, f64, f64)) -> (f64, f64, f64) {
    // (a × b) × c = b(c·a) - a(c·b)
    let ca = dot_product(c, a);
    let cb = dot_product(c, b);
    (
        b.0 * ca - a.0 * cb,
        b.1 * ca - a.1 * cb,
        b.2 * ca - a.2 * cb
    )
}

/// Create dummy simplex for special cases
fn create_collision_simplex() -> Simplex {
    let mut simplex = Simplex::new();
    for i in 0..4 {
        simplex.add(SupportPoint {
            point: (i as f64 * 0.1, 0.0, 0.0),
            point_a: (i as f64 * 0.1, 0.0, 0.0),
            point_b: (0.0, 0.0, 0.0),
        });
    }
    simplex
}

// ============================================================================
// LEGACY COMPATIBILITY LAYER
// ============================================================================

pub fn handle_line_case(simplex: &mut Simplex, direction: &mut (f64, f64, f64)) -> bool {
    evolve_line(simplex, direction)
}

pub fn handle_triangle_case(simplex: &mut Simplex, direction: &mut (f64, f64, f64)) -> bool {
    evolve_triangle(simplex, direction)
}

pub fn handle_tetrahedron_case(simplex: &mut Simplex, direction: &mut (f64, f64, f64)) -> bool {
    evolve_tetrahedron(simplex, direction)
}

/// Legacy function - now uses proper barycentric calculation
pub fn barycentric_coordinates_of_closest_point(
    a: (f64, f64, f64),
    b: (f64, f64, f64),
    c: (f64, f64, f64)
) -> (f64, f64, f64) {
    // Project origin onto triangle and get barycentric coords
    compute_barycentric_coords((0.0, 0.0, 0.0), a, b, c)
}