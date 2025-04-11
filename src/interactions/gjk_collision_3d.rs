use crate::interactions::{cross_product, dot_product, vector_magnitude};
use crate::models::{Quaternion, Shape3D, Simplex, SupportPoint};

/// A face of the polytopes for EPA algorithm
pub struct Face {
    indices: [usize; 3],
    normal: (f64, f64, f64),
    distance: f64,
}

/// Contact information from collision detection
#[derive(Debug, Clone)]
pub struct ContactInfo {
    pub point1: (f64, f64, f64), // Contact point on shape1
    pub point2: (f64, f64, f64), // Contact point on shape2
    pub normal: (f64, f64, f64), // Contact normal (pointing from shape2 to shape1)
    pub penetration: f64,        // Penetration depth
}

/// GJK (Gilbert-Johnson-Keerthi) algorithm for collision detection between convex shapes
/// TODO: I'm not entirely sure i'm following this one correctly tbh but I really want to try it
///         for my dice roll simulation because AABB seems to fail more often than it really should
///         (I'm thinking this is due to the complex rotation and collision speed between multiple objects with multiple impact points)
/// TODO: pt2 - once this is stable, add better documentation
pub fn gjk_collision_detection(
    shape1: &Shape3D,
    position1: (f64, f64, f64),
    orientation1: Quaternion,
    shape2: &Shape3D,
    position2: (f64, f64, f64),
    orientation2: Quaternion
) -> bool {
    // Get the initial search direction (from center of shape1 to center of shape2)
    let mut direction = (
        position2.0 - position1.0,
        position2.1 - position1.1,
        position2.2 - position1.2
    );

    // If the centers are at the same position, use an arbitrary direction
    if vector_magnitude(direction) < 1e-10 {
        direction = (1.0, 0.0, 0.0);
    }

    // Initialize the simplex
    let mut simplex = Simplex::new();

    // Get the first support point
    let first_support = get_support_point(
        shape1, position1, orientation1,
        shape2, position2, orientation2,
        direction
    );
    simplex.add(first_support);

    // Negate the direction for the next iteration
    direction = negate_vector(direction);

    // Main GJK loop
    while simplex.size() < 4 {
        // Get a new support point in the current direction
        let support = get_support_point(
            shape1, position1, orientation1,
            shape2, position2, orientation2,
            direction
        );

        // If the support point doesn't pass the origin, shapes are not colliding
        let support_dot_dir = dot_product(support.point, direction);
        if support_dot_dir < 0.0 {
            return false;
        }

        // Add the new support point to the simplex
        simplex.add(support);

        // Check if the simplex contains the origin
        match simplex.size() {
            2 => {
                // Line case
                if handle_line_case(&mut simplex, &mut direction) {
                    return true;
                }
            },
            3 => {
                // Triangle case
                if handle_triangle_case(&mut simplex, &mut direction) {
                    return true;
                }
            },
            4 => {
                // Tetrahedron case
                if handle_tetrahedron_case(&mut simplex, &mut direction) {
                    return true;
                }
            },
            _ => {}
        }
    }

    // If we've reached here with a full simplex, the origin is inside
    true
}


/// Gets the support point for GJK algorithm
fn get_support_point(
    shape1: &Shape3D,
    position1: (f64, f64, f64),
    orientation1: Quaternion,
    shape2: &Shape3D,
    position2: (f64, f64, f64),
    orientation2: Quaternion,
    direction: (f64, f64, f64)
) -> SupportPoint {
    // Get the farthest point in the given direction for shape1
    let point_a = get_support_point_for_shape(shape1, position1, orientation1, direction);

    // Get the farthest point in the opposite direction for shape2
    let neg_direction = negate_vector(direction);
    let point_b = get_support_point_for_shape(shape2, position2, orientation2, neg_direction);

    // The final support point is the difference between the two points
    let point = (
        point_a.0 - point_b.0,
        point_a.1 - point_b.1,
        point_a.2 - point_b.2
    );

    SupportPoint { point, point_a, point_b }
}

/// Gets the support point for a specific shape
fn get_support_point_for_shape(
    shape: &Shape3D,
    position: (f64, f64, f64),
    orientation: Quaternion,
    direction: (f64, f64, f64)
) -> (f64, f64, f64) {
    // Implementation for various shape types
    match shape {
        Shape3D::Sphere(radius) => {
            // For a sphere, the support point is just radius * normalized direction
            let dir_magnitude = vector_magnitude(direction);
            if dir_magnitude < 1e-10 {
                return position;
            }

            let normalized_dir = (
                direction.0 / dir_magnitude,
                direction.1 / dir_magnitude,
                direction.2 / dir_magnitude
            );

            (
                position.0 + normalized_dir.0 * radius,
                position.1 + normalized_dir.1 * radius,
                position.2 + normalized_dir.2 * radius
            )
        },
        Shape3D::Cuboid(width, height, depth) => {
            // For a cuboid, we need to transform the direction to local space
            let half_width = width / 2.0;
            let half_height = height / 2.0;
            let half_depth = depth / 2.0;

            // Convert direction to shape's local space
            let local_dir = orientation.inverse().rotate_point(direction);

            // Get the farthest point in that direction
            let local_support = (
                if local_dir.0 >= 0.0 { half_width } else { -half_width },
                if local_dir.1 >= 0.0 { half_height } else { -half_height },
                if local_dir.2 >= 0.0 { half_depth } else { -half_depth }
            );

            // Transform back to world space
            let rotated_support = orientation.rotate_point(local_support);

            (
                position.0 + rotated_support.0,
                position.1 + rotated_support.1,
                position.2 + rotated_support.2
            )
        },
        Shape3D::BeveledCuboid(width, height, depth, bevel) => {
            // Similar to cuboid but with beveled edges
            let half_width = width / 2.0;
            let half_height = height / 2.0;
            let half_depth = depth / 2.0;

            // Convert direction to shape's local space
            let local_dir = orientation.inverse().rotate_point(direction);

            // Normalize the local direction
            let dir_magnitude = vector_magnitude(local_dir);
            if dir_magnitude < 1e-10 {
                let rotated_origin = orientation.rotate_point((0.0, 0.0, 0.0));
                return (
                    position.0 + rotated_origin.0,
                    position.1 + rotated_origin.1,
                    position.2 + rotated_origin.2
                );
            }

            let norm_local_dir = (
                local_dir.0 / dir_magnitude,
                local_dir.1 / dir_magnitude,
                local_dir.2 / dir_magnitude
            );

            // Get the farthest point with beveled corners
            let local_support = if norm_local_dir.0.abs() > 0.999 ||
                norm_local_dir.1.abs() > 0.999 ||
                norm_local_dir.2.abs() > 0.999 {
                // Direction is along one of the main axes
                (
                    if norm_local_dir.0 >= 0.0 { half_width } else { -half_width },
                    if norm_local_dir.1 >= 0.0 { half_height } else { -half_height },
                    if norm_local_dir.2 >= 0.0 { half_depth } else { -half_depth }
                )
            } else {
                // Direction is not along the main axes, consider beveling
                let x = if norm_local_dir.0 >= 0.0 { half_width - bevel } else { -half_width + bevel };
                let y = if norm_local_dir.1 >= 0.0 { half_height - bevel } else { -half_height + bevel };
                let z = if norm_local_dir.2 >= 0.0 { half_depth - bevel } else { -half_depth + bevel };

                // Add the beveled part
                let bevel_dir = (
                    norm_local_dir.0 * bevel,
                    norm_local_dir.1 * bevel,
                    norm_local_dir.2 * bevel
                );

                (x + bevel_dir.0, y + bevel_dir.1, z + bevel_dir.2)
            };

            // Transform back to world space
            let rotated_support = orientation.rotate_point(local_support);

            (
                position.0 + rotated_support.0,
                position.1 + rotated_support.1,
                position.2 + rotated_support.2
            )
        },
        Shape3D::Cylinder(radius, height) => {
            // For a cylinder, we need special handling based on the direction
            let half_height = height / 2.0;

            // Convert direction to shape's local space
            let local_dir = orientation.inverse().rotate_point(direction);

            // Decompose into axial and radial components
            let _axial_dir = (0.0, local_dir.1, 0.0);
            let radial_dir = (local_dir.0, 0.0, local_dir.2);
            let radial_magnitude = vector_magnitude(radial_dir);

            // Calculate support point in local space
            let local_support = if radial_magnitude < 1e-10 {
                // Direction is along the cylinder axis
                (
                    0.0,
                    if local_dir.1 >= 0.0 { half_height } else { -half_height },
                    0.0
                )
            } else {
                // Direction has a radial component
                let normalized_radial = (
                    radial_dir.0 / radial_magnitude,
                    0.0,
                    radial_dir.2 / radial_magnitude
                );

                (
                    normalized_radial.0 * radius,
                    if local_dir.1 >= 0.0 { half_height } else { -half_height },
                    normalized_radial.2 * radius
                )
            };

            // Transform back to world space
            let rotated_support = orientation.rotate_point(local_support);

            (
                position.0 + rotated_support.0,
                position.1 + rotated_support.1,
                position.2 + rotated_support.2
            )
        },
        Shape3D::Polyhedron(vertices, _) => {
            // For a polyhedron, find the vertex with maximum dot product with direction
            let local_dir = orientation.inverse().rotate_point(direction);

            let mut max_dot = -f64::INFINITY;
            let mut max_vertex = (0.0, 0.0, 0.0);

            for vertex in vertices {
                let dot = dot_product(*vertex, local_dir);
                if dot > max_dot {
                    max_dot = dot;
                    max_vertex = *vertex;
                }
            }

            // Transform back to world space
            let rotated_vertex = orientation.rotate_point(max_vertex);

            (
                position.0 + rotated_vertex.0,
                position.1 + rotated_vertex.1,
                position.2 + rotated_vertex.2
            )
        }
    }
}

/// returns the inverse of the vector
fn negate_vector(v: (f64, f64, f64)) -> (f64, f64, f64) {
    (-v.0, -v.1, -v.2)
}

/// Handles the line case for GJK algorithm
fn handle_line_case(simplex: &mut Simplex, direction: &mut (f64, f64, f64)) -> bool {
    let a = simplex.get_a().point;
    let b = simplex.get_b().point;

    // Vector from b to a
    let ab = (a.0 - b.0, a.1 - b.1, a.2 - b.2);
    // Vector from b to origin
    let b0 = (0.0 - b.0, 0.0 - b.1, 0.0 - b.2);

    // New direction is perpendicular to the line towards the origin
    *direction = triple_product(ab, b0, ab);

    // If the new direction is close to zero, the origin is on the line
    if vector_magnitude(*direction) < 1e-10 {
        // Try a different approach: direction perpendicular to AB
        if ab.0.abs() > ab.1.abs() && ab.0.abs() > ab.2.abs() {
            *direction = cross_product(ab, (0.0, 1.0, 0.0));
        } else {
            *direction = cross_product(ab, (1.0, 0.0, 0.0));
        }
    }

    false
}

/// Handles the triangle case for GJK algorithm
fn handle_triangle_case(simplex: &mut Simplex, direction: &mut (f64, f64, f64)) -> bool {
    let a = simplex.get_a().point;
    let b = simplex.get_b().point;
    let c = simplex.get_c().point;

    // Vector from c to a
    let ac = (a.0 - c.0, a.1 - c.1, a.2 - c.2);
    // Vector from c to b
    let bc = (b.0 - c.0, b.1 - c.1, b.2 - c.2);
    // Vector from c to origin
    let c0 = (0.0 - c.0, 0.0 - c.1, 0.0 - c.2);

    // Cross product of AC and BC
    let abc = cross_product(ac, bc);

    // Check where the origin is relative to the triangle
    let acf = cross_product(ac, abc);
    if dot_product(acf, c0) > 0.0 {
        // Origin is outside AC edge
        // Remove b and update direction
        simplex.set_ab(simplex.get_a().clone(), simplex.get_c().clone());
        *direction = acf;
        return false;
    }

    let bcf = cross_product(abc, bc);
    if dot_product(bcf, c0) > 0.0 {
        // Origin is outside BC edge
        // Remove a and update direction
        simplex.set_ab(simplex.get_b().clone(), simplex.get_c().clone());
        *direction = bcf;
        return false;
    }

    // Origin is inside the triangle, but check if it's above or below
    if dot_product(abc, c0) > 0.0 {
        // Origin is above the triangle
        *direction = abc;
    } else {
        // Origin is below the triangle
        *direction = negate_vector(abc);
    }

    false
}

/// Handles the tetrahedron case for GJK algorithm
fn handle_tetrahedron_case(simplex: &mut Simplex, direction: &mut (f64, f64, f64)) -> bool {
    let a = simplex.get_a().point;
    let b = simplex.get_b().point;
    let c = simplex.get_c().point;
    let d = simplex.get_d().point;

    // Vector from d to a
    let ad = (a.0 - d.0, a.1 - d.1, a.2 - d.2);
    // Vector from d to b
    let bd = (b.0 - d.0, b.1 - d.1, b.2 - d.2);
    // Vector from d to c
    let cd = (c.0 - d.0, c.1 - d.1, c.2 - d.2);
    // Vector from d to origin
    let d0 = (0.0 - d.0, 0.0 - d.1, 0.0 - d.2);

    // Faces of the tetrahedron
    let abc = cross_product(ad, bd);
    let acd = cross_product(cd, ad);
    let bdc = cross_product(bd, cd);

    // Check where the origin is relative to each face
    if dot_product(abc, d0) > 0.0 {
        // Origin is outside face ABC
        simplex.set_abc(simplex.get_a().clone(), simplex.get_b().clone(), simplex.get_c().clone());
        *direction = abc;
        return false;
    }

    if dot_product(acd, d0) > 0.0 {
        // Origin is outside face ACD
        simplex.set_abc(simplex.get_a().clone(), simplex.get_c().clone(), simplex.get_d().clone());
        *direction = acd;
        return false;
    }

    if dot_product(bdc, d0) > 0.0 {
        // Origin is outside face BDC
        simplex.set_abc(simplex.get_b().clone(), simplex.get_d().clone(), simplex.get_c().clone());
        *direction = bdc;
        return false;
    }

    // If we've reached here, the origin is inside the tetrahedron
    true
}

// Triple product: A × (B × C)
fn triple_product(a: (f64, f64, f64), b: (f64, f64, f64), c: (f64, f64, f64)) -> (f64, f64, f64) {
    let bc = cross_product(b, c);
    cross_product(a, bc)
}

/// EPA (Expanding Polytopes Algorithm) for finding contact information
pub fn epa_contact_points(
    shape1: &Shape3D,
    position1: (f64, f64, f64),
    orientation1: Quaternion,
    shape2: &Shape3D,
    position2: (f64, f64, f64),
    orientation2: Quaternion,
    simplex: &Simplex
) -> Option<ContactInfo> {
    // Create the initial polytope from the simplex
    let mut polytope = simplex.points.clone();
    let mut faces = Vec::new();

    // Generate initial faces
    // For a tetrahedron, we have 4 triangular faces
    if polytope.len() == 4 {
        // Create the faces
        let face_indices = [
            [0, 1, 2],
            [0, 3, 1],
            [0, 2, 3],
            [1, 3, 2]
        ];

        for indices in &face_indices {
            let a = polytope[indices[0]].point;
            let b = polytope[indices[1]].point;
            let c = polytope[indices[2]].point;

            // Calculate face normal and distance
            let ab = (b.0 - a.0, b.1 - a.1, b.2 - a.2);
            let ac = (c.0 - a.0, c.1 - a.1, c.2 - a.2);

            let normal = cross_product(ab, ac);
            let normal_length = vector_magnitude(normal);

            if normal_length < 1e-10 {
                continue; // Skip degenerate faces
            }

            let normalized_normal = (
                normal.0 / normal_length,
                normal.1 / normal_length,
                normal.2 / normal_length
            );

            // Make sure the normal points outward
            let dot = dot_product(normalized_normal, a);
            let normalized_normal = if dot < 0.0 {
                normalized_normal
            } else {
                negate_vector(normalized_normal)
            };

            // Calculate distance to origin
            let distance = -dot_product(normalized_normal, a);

            faces.push(Face {
                indices: [indices[0], indices[1], indices[2]],
                normal: normalized_normal,
                distance: distance.abs(),
            });
        }
    }

    // Main EPA loop
    const MAX_ITERATIONS: usize = 32;
    const TOLERANCE: f64 = 1e-6;

    for _ in 0..MAX_ITERATIONS {
        // Find the closest face to the origin
        if faces.is_empty() {
            return None;
        }

        let closest_face = faces.iter()
            .min_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();

        // Get the support point in the direction of the face normal
        let support = get_support_point(
            shape1, position1, orientation1,
            shape2, position2, orientation2,
            closest_face.normal
        );

        // Calculate distance from support point to closest face
        let support_dist = dot_product(support.point, closest_face.normal);

        // Check if we've reached the tolerance
        if (support_dist - closest_face.distance).abs() < TOLERANCE {
            // We've found the closest point, calculate contact information
            return calculate_contact_from_face(closest_face, &polytope, shape1, position1, orientation1, shape2, position2, orientation2);
        }

        // We haven't reached tolerance, expand the polytope

        // Remove faces that can be seen from the support point
        let mut visible_faces: Vec<usize> = Vec::new();
        let mut visible_edges: Vec<(usize, usize)> = Vec::new();

        for (i, face) in faces.iter().enumerate() {
            // A face is visible if the support point is in front of it
            if dot_product(face.normal, support.point) > 0.0 {
                visible_faces.push(i);

                // Keep track of the edges of visible faces
                let edges = [
                    (face.indices[0], face.indices[1]),
                    (face.indices[1], face.indices[2]),
                    (face.indices[2], face.indices[0])
                ];

                for edge in &edges {
                    let mut found = false;

                    // Check if this edge is already in our list
                    for (j, &existing_edge) in visible_edges.iter().enumerate() {
                        if edge.0 == existing_edge.1 && edge.1 == existing_edge.0 {
                            // This edge is shared with another visible face, remove it
                            visible_edges.swap_remove(j);
                            found = true;
                            break;
                        }
                    }

                    if !found {
                        // This is a new edge of the silhouette
                        visible_edges.push(*edge);
                    }
                }
            }
        }

        // If no faces are visible, something went wrong
        if visible_faces.is_empty() {
            return None;
        }

        // Remove all visible faces
        // Sort in reverse order to avoid invalidating indices
        visible_faces.sort_unstable_by(|a, b| b.cmp(a));
        for &face_index in &visible_faces {
            faces.swap_remove(face_index);
        }

        // Add the new support point to the polytope
        polytope.push(support.clone());
        let new_vertex_index = polytope.len() - 1;

        // Create new faces connecting the supporting point with the boundary edges
        for &(i, j) in &visible_edges {
            let a = polytope[i].point;
            let b = polytope[j].point;
            let c = support.point;

            // Calculate face normal and distance
            let ab = (b.0 - a.0, b.1 - a.1, b.2 - a.2);
            let ac = (c.0 - a.0, c.1 - a.1, c.2 - a.2);

            let normal = cross_product(ab, ac);
            let normal_length = vector_magnitude(normal);

            if normal_length < 1e-10 {
                continue; // Skip degenerate faces
            }

            let normalized_normal = (
                normal.0 / normal_length,
                normal.1 / normal_length,
                normal.2 / normal_length
            );

            // Make sure the normal points outward
            let dot = dot_product(normalized_normal, a);
            let normalized_normal = if dot < 0.0 {
                normalized_normal
            } else {
                negate_vector(normalized_normal)
            };

            // Calculate distance to origin
            let distance = -dot_product(normalized_normal, a);

            faces.push(Face {
                indices: [i, j, new_vertex_index],
                normal: normalized_normal,
                distance: distance.abs(),
            });
        }
    }

    // If we've reached the maximum number of iterations, use the closest face
    let closest_face = faces.iter()
        .min_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap();

    calculate_contact_from_face(closest_face, &polytope, shape1, position1, orientation1, shape2, position2, orientation2)
}

/// Calculate contact information from the closest face found by EPA
///
/// note: The unused parameters here may be removed later but as I refine the approach
///     I would rather require them now to make refactoring easier later if I end up removing some
///     of them.
fn calculate_contact_from_face(
    face: &Face,
    polytope: &[SupportPoint],
    _shape1: &Shape3D,
    _position1: (f64, f64, f64),
    _orientation1: Quaternion,
    _shape2: &Shape3D,
    _position2: (f64, f64, f64),
    _orientation2: Quaternion
) -> Option<ContactInfo> {
    // Get the vertices of the face
    let a = polytope[face.indices[0]].point;
    let b = polytope[face.indices[1]].point;
    let c = polytope[face.indices[2]].point;

    // Barycentric coordinates of the closest point to origin on the face
    let barycentric = barycentric_coordinates_of_closest_point(a, b, c);

    // Points on the original shapes
    let p1_a = polytope[face.indices[0]].point_a;
    let p1_b = polytope[face.indices[1]].point_a;
    let p1_c = polytope[face.indices[2]].point_a;

    let p2_a = polytope[face.indices[0]].point_b;
    let p2_b = polytope[face.indices[1]].point_b;
    let p2_c = polytope[face.indices[2]].point_b;

    // Calculate contact points on both shapes
    let contact_point1: (f64, f64, f64) = (
        barycentric.0 * p1_a.0 + barycentric.1 * p1_b.0 + barycentric.2 * p1_c.0,
        barycentric.0 * p1_a.1 + barycentric.1 * p1_b.1 + barycentric.2 * p1_c.1,
        barycentric.0 * p1_a.2 + barycentric.1 * p1_b.2 + barycentric.2 * p1_c.2
    );

    let contact_point2: (f64, f64, f64) = (
        barycentric.0 * p2_a.0 + barycentric.1 * p2_b.0 + barycentric.2 * p2_c.0,
        barycentric.0 * p2_a.1 + barycentric.1 * p2_b.1 + barycentric.2 * p2_c.1,
        barycentric.0 * p2_a.2 + barycentric.1 * p2_b.2 + barycentric.2 * p2_c.2
    );

    // Normal pointing from shape2 to shape1
    let normal = negate_vector(face.normal);

    // Calculate penetration depth
    let penetration = face.distance;

    Some(ContactInfo {
        point1: contact_point1,
        point2: contact_point2,
        normal,
        penetration,
    })
}

/// Calculate barycentric coordinates of the closest point to origin on a triangle
fn barycentric_coordinates_of_closest_point(
    a: (f64, f64, f64),
    b: (f64, f64, f64),
    c: (f64, f64, f64)
) -> (f64, f64, f64) {
    // Calculate the closest point on the face to the origin
    let ab = (b.0 - a.0, b.1 - a.1, b.2 - a.2);
    let ac = (c.0 - a.0, c.1 - a.1, c.2 - a.2);
    let ap = (0.0 - a.0, 0.0 - a.1, 0.0 - a.2); // Vector from a to origin

    // Gram matrix
    let d00 = dot_product(ab, ab);
    let d01 = dot_product(ab, ac);
    let d11 = dot_product(ac, ac);
    let d20 = dot_product(ap, ab);
    let d21 = dot_product(ap, ac);

    let inv_denom = 1.0 / (d00 * d11 - d01 * d01);

    // Calculate barycentric coordinates
    let v = (d11 * d20 - d01 * d21) * inv_denom;
    let w = (d00 * d21 - d01 * d20) * inv_denom;
    let u = 1.0 - v - w;

    // Clamp to the triangle
    if u < 0.0 || v < 0.0 || w < 0.0 {
        // Closest point is not inside the triangle, find closest edge or vertex

        // - note: the lengths here are currently not used but may be in further refinement,
        //         or they will be removed
        //
        // Check edges
        let _ab_length = vector_magnitude(ab);
        let _ac_length = vector_magnitude(ac);
        let bc = (c.0 - b.0, c.1 - b.1, c.2 - b.2);
        let _bc_length = vector_magnitude(bc);

        // Check vertices
        let dist_a = vector_magnitude(ap);
        let bp = (0.0 - b.0, 0.0 - b.1, 0.0 - b.2);
        let dist_b = vector_magnitude(bp);
        let cp = (0.0 - c.0, 0.0 - c.1, 0.0 - c.2);
        let dist_c = vector_magnitude(cp);

        // Find the closest feature
        if dist_a <= dist_b && dist_a <= dist_c {
            // Closest to vertex A
            return (1.0, 0.0, 0.0);
        } else if dist_b <= dist_a && dist_b <= dist_c {
            // Closest to vertex B
            return (0.0, 1.0, 0.0);
        } else {
            // Closest to vertex C
            return (0.0, 0.0, 1.0);
        }
    }

    (u, v, w)
}
