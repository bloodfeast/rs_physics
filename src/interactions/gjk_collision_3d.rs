use crate::interactions::{cross_product, dot_product, vector_magnitude};
use crate::models::{Quaternion, Shape3D, Simplex, SupportPoint};

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

/// returns the inverse of the vector
pub fn negate_vector(v: (f64, f64, f64)) -> (f64, f64, f64) {
    (-v.0, -v.1, -v.2)
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
