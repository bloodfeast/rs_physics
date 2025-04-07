use crate::interactions::{cross_product, dot_product};
use crate::materials::Material;
use crate::models::{ObjectIn3D, ToCoordinates};
use crate::rotational_dynamics::{calculate_moment_of_inertia, ObjectShape};
use std::f64::consts::PI;

/// Represents different types of 3D shapes for physics simulations
#[derive(Debug, Clone)]
pub enum Shape3D {
    /// Sphere with a radius
    Sphere(f64),
    /// Regular cuboid with dimensions (width, height, depth)
    Cuboid(f64, f64, f64),
    /// Beveled cuboid with dimensions (width, height, depth) and bevel amount
    BeveledCuboid(f64, f64, f64, f64),
    /// Cylinder with radius and height
    Cylinder(f64, f64),
    /// Custom polyhedron with vertices and faces (indices)
    Polyhedron(Vec<(f64, f64, f64)>, Vec<Vec<usize>>),
}

impl Shape3D {
    /// Creates a new sphere with the given radius
    pub fn new_sphere(radius: f64) -> Self {
        Shape3D::Sphere(radius)
    }

    /// Creates a new cuboid with the given dimensions
    pub fn new_cuboid(width: f64, height: f64, depth: f64) -> Self {
        Shape3D::Cuboid(width, height, depth)
    }

    /// Creates a new beveled cuboid (die) with the given dimensions and bevel amount
    pub fn new_beveled_cuboid(width: f64, height: f64, depth: f64, bevel: f64) -> Self {
        if bevel <= 0.0 || bevel >= width.min(height.min(depth)) / 2.0 {
            // If bevel is invalid, create a regular cuboid instead
            Shape3D::Cuboid(width, height, depth)
        } else {
            Shape3D::BeveledCuboid(width, height, depth, bevel)
        }
    }

    /// Creates a new die (beveled cube with equal dimensions) with the given size and bevel amount
    pub fn new_die(size: f64, bevel: f64) -> Self {
        Self::new_beveled_cuboid(size, size, size, bevel)
    }

    /// Creates a new cylinder with the given radius and height
    pub fn new_cylinder(radius: f64, height: f64) -> Self {
        Shape3D::Cylinder(radius, height)
    }

    /// Creates a new polyhedron with the given vertices and faces
    pub fn new_polyhedron(vertices: Vec<(f64, f64, f64)>, faces: Vec<Vec<usize>>) -> Self {
        Shape3D::Polyhedron(vertices, faces)
    }

    /// Returns the volume of the shape
    pub fn volume(&self) -> f64 {
        match self {
            Shape3D::Sphere(radius) => (4.0 / 3.0) * PI * radius.powi(3),
            Shape3D::Cuboid(w, h, d) => w * h * d,
            Shape3D::BeveledCuboid(w, h, d, bevel) => {
                // Approximate volume calculation for a beveled cuboid
                // Start with the volume of the cuboid and subtract the beveled corners
                let base_volume = w * h * d;

                // Estimate the volume reduction from beveling
                // Each corner reduces by approximately 1/3 of a sphere with radius = bevel
                let corner_reduction = 8.0 * (4.0 / 3.0 * PI * bevel.powi(3)) / 3.0;

                // Reduce for the 12 edges (approximated as partial cylinders)
                let edge_reduction = 12.0 * (PI * bevel.powi(2) / 4.0) *
                    ((w - 2.0 * bevel) + (h - 2.0 * bevel) + (d - 2.0 * bevel)) / 3.0;

                base_volume - corner_reduction - edge_reduction
            },
            Shape3D::Cylinder(radius, height) => PI * radius.powi(2) * height,
            Shape3D::Polyhedron(vertices, faces) => {
                // Calculate volume using the divergence theorem with tetrahedra
                let mut total_volume = 0.0;

                for face in faces {
                    if face.len() >= 3 {
                        // Use the first vertex as reference
                        let v0 = vertices[face[0]];

                        // Create tetrahedra from triangles formed by consecutive vertices
                        for i in 1..(face.len() - 1) {
                            let v1 = vertices[face[i]];
                            let v2 = vertices[face[i + 1]];

                            // Calculate signed volume of tetrahedron
                            let edge1 = (v1.0 - v0.0, v1.1 - v0.1, v1.2 - v0.2);
                            let edge2 = (v2.0 - v0.0, v2.1 - v0.1, v2.2 - v0.2);
                            let cross = cross_product(edge1, edge2);

                            total_volume += dot_product(v0, cross) / 6.0;
                        }
                    }
                }

                total_volume.abs()
            }
        }
    }

    /// Calculates the center of mass of the shape
    pub fn center_of_mass(&self) -> (f64, f64, f64) {
        match self {
            // For these simple shapes, the center of mass is at the geometric center
            Shape3D::Sphere(_) |
            Shape3D::Cuboid(_, _, _) |
            Shape3D::BeveledCuboid(_, _, _, _) |
            Shape3D::Cylinder(_, _) => (0.0, 0.0, 0.0),

            // faces are going to be used later I'm sure
            Shape3D::Polyhedron(vertices, _faces) => {
                // For a polyhedron, we need to calculate the center of mass
                // This is a simplified calculation assuming uniform density
                let num_vertices = vertices.len() as f64;
                let mut cx = 0.0;
                let mut cy = 0.0;
                let mut cz = 0.0;

                for (x, y, z) in vertices {
                    cx += x;
                    cy += y;
                    cz += z;
                }

                (cx / num_vertices, cy / num_vertices, cz / num_vertices)
            }
        }
    }

    /// Returns the moment of inertia tensor for the shape around its center of mass
    pub fn moment_of_inertia(&self, mass: f64) -> [f64; 6] {
        match self {
            Shape3D::Sphere(radius) => {
                // For a solid sphere, moment of inertia is (2/5) * m * r²
                let i = (2.0 / 5.0) * mass * radius.powi(2);
                [i, i, i, 0.0, 0.0, 0.0] // Diagonal terms only
            },
            Shape3D::Cuboid(w, h, d) => {
                // For a cuboid, the moments of inertia are:
                // Ixx = (1/12) * m * (h² + d²)
                // Iyy = (1/12) * m * (w² + d²)
                // Izz = (1/12) * m * (w² + h²)
                let ixx = (1.0 / 12.0) * mass * (h.powi(2) + d.powi(2));
                let iyy = (1.0 / 12.0) * mass * (w.powi(2) + d.powi(2));
                let izz = (1.0 / 12.0) * mass * (w.powi(2) + h.powi(2));
                [ixx, iyy, izz, 0.0, 0.0, 0.0] // Diagonal terms only
            },
            Shape3D::BeveledCuboid(w, h, d, bevel) => {
                // Use the regular cuboid calculation with a small adjustment
                // This is an approximation that accounts for the slight reduction in moment of inertia
                let bevel_factor = 1.0 - (bevel / w.min(h.min(*d))) * 0.3;

                let ixx = (1.0 / 12.0) * mass * (h.powi(2) + d.powi(2)) * bevel_factor;
                let iyy = (1.0 / 12.0) * mass * (w.powi(2) + d.powi(2)) * bevel_factor;
                let izz = (1.0 / 12.0) * mass * (w.powi(2) + h.powi(2)) * bevel_factor;

                [ixx, iyy, izz, 0.0, 0.0, 0.0] // Diagonal terms only
            },
            Shape3D::Cylinder(radius, height) => {
                // For a cylinder around its center of mass
                // Ixx = Iyy = (1/12) * m * (3r² + h²)
                // Izz = (1/2) * m * r²
                let ixx_iyy = (1.0 / 12.0) * mass * (3.0 * radius.powi(2) + height.powi(2));
                let izz = (1.0 / 2.0) * mass * radius.powi(2);

                [ixx_iyy, ixx_iyy, izz, 0.0, 0.0, 0.0] // Diagonal terms only
            },
            Shape3D::Polyhedron(vertices, _) => {
                // For a general polyhedron, calculate the inertia tensor numerically
                // This is a simplified approach assuming uniform density
                let mut ixx = 0.0;
                let mut iyy = 0.0;
                let mut izz = 0.0;
                let mut ixy = 0.0;
                let mut ixz = 0.0;
                let mut iyz = 0.0;

                // Get center of mass
                let (cx, cy, cz) = self.center_of_mass();

                // Calculate moment of inertia using the parallel axis theorem
                for &(x, y, z) in vertices {
                    // Adjust coordinates relative to center of mass
                    let rx = x - cx;
                    let ry = y - cy;
                    let rz = z - cz;

                    // Contribute to moment of inertia tensor
                    ixx += ry.powi(2) + rz.powi(2);
                    iyy += rx.powi(2) + rz.powi(2);
                    izz += rx.powi(2) + ry.powi(2);
                    ixy -= rx * ry;
                    ixz -= rx * rz;
                    iyz -= ry * rz;
                }

                // Scale by mass / number of vertices
                let scale_factor = mass / (vertices.len() as f64);
                [
                    ixx * scale_factor,
                    iyy * scale_factor,
                    izz * scale_factor,
                    ixy * scale_factor,
                    ixz * scale_factor,
                    iyz * scale_factor
                ]
            }
        }
    }

    /// Returns the minimum bounding sphere radius
    pub fn bounding_radius(&self) -> f64 {
        match self {
            Shape3D::Sphere(radius) => *radius,
            Shape3D::Cuboid(w, h, d) => {
                // Half-diagonal of the cuboid
                (w.powi(2) + h.powi(2) + d.powi(2)).sqrt() / 2.0
            },
            Shape3D::BeveledCuboid(w, h, d, _) => {
                // Beveling doesn't significantly change the bounding radius
                (w.powi(2) + h.powi(2) + d.powi(2)).sqrt() / 2.0
            },
            Shape3D::Cylinder(radius, height) => {
                // Maximum distance from center to any point
                (radius.powi(2) + (height / 2.0).powi(2)).sqrt()
            },
            Shape3D::Polyhedron(vertices, _) => {
                // Find the furthest vertex from center
                let (cx, cy, cz) = self.center_of_mass();

                vertices.iter()
                    .map(|(x, y, z)| {
                        ((x - cx).powi(2) + (y - cy).powi(2) + (z - cz).powi(2)).sqrt()
                    })
                    .fold(0.0, |max, dist| if dist > max { dist } else { max })
            }
        }
    }

    /// Creates vertices for the shape
    /// Returns a vector of (x,y,z) coordinates
    pub fn create_vertices(&self) -> Vec<(f64, f64, f64)> {
        match self {
            Shape3D::Sphere(radius) => {
                // Create a simple icosphere approximation
                let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
                let mut vertices = vec![
                    (0.0, *radius, phi * radius),
                    (0.0, *radius, -phi * radius),
                    (0.0, -*radius, phi * radius),
                    (0.0, -*radius, -phi * radius),
                    (*radius, phi * radius, 0.0),
                    (*radius, -phi * radius, 0.0),
                    (-*radius, phi * radius, 0.0),
                    (-*radius, -phi * radius, 0.0),
                    (phi * radius, 0.0, *radius),
                    (phi * radius, 0.0, -*radius),
                    (-phi * radius, 0.0, *radius),
                    (-phi * radius, 0.0, -*radius),
                ];

                // Normalize vertices to radius
                for vertex in &mut vertices {
                    let len = (vertex.0.powi(2) + vertex.1.powi(2) + vertex.2.powi(2)).sqrt();
                    vertex.0 = vertex.0 * radius / len;
                    vertex.1 = vertex.1 * radius / len;
                    vertex.2 = vertex.2 * radius / len;
                }

                vertices
            },
            Shape3D::Cuboid(w, h, d) => {
                // 8 vertices of a cuboid
                let w2 = w / 2.0;
                let h2 = h / 2.0;
                let d2 = d / 2.0;

                vec![
                    (-w2, -h2, -d2),
                    (w2, -h2, -d2),
                    (w2, h2, -d2),
                    (-w2, h2, -d2),
                    (-w2, -h2, d2),
                    (w2, -h2, d2),
                    (w2, h2, d2),
                    (-w2, h2, d2),
                ]
            },
            Shape3D::BeveledCuboid(w, h, d, bevel) => {
                let w2 = w / 2.0;
                let h2 = h / 2.0;
                let d2 = d / 2.0;
                let b = *bevel;

                let mut vertices = Vec::new();

                // CORNER VERTICES (8 corners)
                // Each corner will be replaced by a spherical section
                // represented by multiple vertices

                // Define the 8 corners of the cuboid (pre-beveling)
                let corners: [(f64, f64, f64); 8] = [
                    (-w2, -h2, -d2), // 0: bottom-left-front
                    (w2, -h2, -d2),  // 1: bottom-right-front
                    (w2, h2, -d2),   // 2: top-right-front
                    (-w2, h2, -d2),  // 3: top-left-front
                    (-w2, -h2, d2),  // 4: bottom-left-back
                    (w2, -h2, d2),   // 5: bottom-right-back
                    (w2, h2, d2),    // 6: top-right-back
                    (-w2, h2, d2),   // 7: top-left-back
                ];

                // For each corner, add vertices for a spherical bevel
                let corner_segments = 4; // Number of segments for each corner
                let mut corner_vertices_indices = Vec::new();

                // c_idx may be useful in the future
                for (_c_idx, &(cx, cy, cz)) in corners.iter().enumerate() {
                    // Direction vectors from corner to center
                    let dx = if cx < 0.0 { 1.0 } else { -1.0 };
                    let dy = if cy < 0.0 { 1.0 } else { -1.0 };
                    let dz = if cz < 0.0 { 1.0 } else { -1.0 };

                    // Calculate corner center (moved inward by bevel)
                    let corner_center: (f64, f64, f64) = (
                        cx + dx * b,
                        cy + dy * b,
                        cz + dz * b
                    );

                    // Add indices for this corner
                    let mut corner_indices = Vec::new();

                    // Create vertices in a spherical pattern around the corner center
                    for i in 0..corner_segments {
                        let theta = i as f64 * PI / 2.0 / (corner_segments as f64);

                        for j in 0..corner_segments {
                            let phi = j as f64 * PI / 2.0 / (corner_segments as f64);

                            // Calculate position on a unit sphere in the correct octant
                            let sx = theta.sin() * phi.cos() * dx.abs();
                            let sy = theta.sin() * phi.sin() * dy.abs();
                            let sz = theta.cos() * dz.abs();

                            // Transform to corner space and add bevel
                            let vertex: (f64, f64, f64) = (
                                corner_center.0 + sx * b * dx.signum(),
                                corner_center.1 + sy * b * dy.signum(),
                                corner_center.2 + sz * b * dz.signum()
                            );

                            vertices.push(vertex);
                            corner_indices.push(vertices.len() - 1);
                        }
                    }

                    corner_vertices_indices.push(corner_indices);
                }

                // EDGE VERTICES (12 edges)
                // Each edge will be replaced by a cylindrical section
                // represented by multiple vertices

                // Define the 12 edges as pairs of corner indices
                let edges: [(usize, usize); 12] = [
                    (0, 1), (1, 2), (2, 3), (3, 0), // Front face edges
                    (4, 5), (5, 6), (6, 7), (7, 4), // Back face edges
                    (0, 4), (1, 5), (2, 6), (3, 7)  // Connecting edges
                ];

                let edge_segments = 6; // Number of segments for each edge
                let mut edge_vertices_indices = Vec::new();

                for &(c1, c2) in edges.iter() {
                    // Get the corner positions (pre-beveling)
                    let (c1x, c1y, c1z) = corners[c1];
                    let (c2x, c2y, c2z) = corners[c2];

                    // Calculate edge axis (which axis is changing)
                    let x_same = (c1x - c2x).abs() < 1e-5;
                    let y_same = (c1y - c2y).abs() < 1e-5;
                    let z_same = (c1z - c2z).abs() < 1e-5;

                    // Determine axis directions
                    let x_dir = if !x_same { if c1x < c2x { 1.0 } else { -1.0 } } else { 0.0 };
                    let y_dir = if !y_same { if c1y < c2y { 1.0 } else { -1.0 } } else { 0.0 };
                    let z_dir = if !z_same { if c1z < c2z { 1.0 } else { -1.0 } } else { 0.0 };

                    // Calculate edge length (accounting for bevels at both ends)
                    let edge_length = if !x_same {
                        w - 2.0 * b
                    } else if !y_same {
                        h - 2.0 * b
                    } else {
                        d - 2.0 * b
                    };

                    // Get perpendicular directions for the cylinder
                    let (perp1, perp2): ((f64, f64, f64), (f64, f64, f64)) = if x_same {
                        ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0))
                    } else if y_same {
                        ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
                    } else {
                        ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0))
                    };

                    // Calculate start point of the edge (after beveling)
                    let start: (f64, f64, f64) = (
                        c1x + (if !x_same { x_dir * b } else { 0.0 }),
                        c1y + (if !y_same { y_dir * b } else { 0.0 }),
                        c1z + (if !z_same { z_dir * b } else { 0.0 })
                    );

                    let mut edge_indices = Vec::new();

                    // Create vertices along the edge
                    for i in 0..=edge_segments {
                        // Position along the edge (0 to 1)
                        let t = i as f64 / edge_segments as f64;

                        // Center point at this position
                        let center: (f64, f64, f64) = (
                            start.0 + x_dir * t * edge_length,
                            start.1 + y_dir * t * edge_length,
                            start.2 + z_dir * t * edge_length
                        );

                        // Create vertices in a circular pattern around the center
                        let circle_segments = 8; // Number of segments for the circle
                        let mut circle_indices = Vec::new();

                        for j in 0..circle_segments {
                            let angle = j as f64 * 2.0 * PI / circle_segments as f64;
                            let cos_angle = angle.cos();
                            let sin_angle = angle.sin();

                            // Calculate position on circle
                            let vertex: (f64, f64, f64) = (
                                center.0 + (perp1.0 * cos_angle + perp2.0 * sin_angle) * b,
                                center.1 + (perp1.1 * cos_angle + perp2.1 * sin_angle) * b,
                                center.2 + (perp1.2 * cos_angle + perp2.2 * sin_angle) * b
                            );

                            vertices.push(vertex);
                            circle_indices.push(vertices.len() - 1);
                        }

                        edge_indices.push(circle_indices);
                    }

                    edge_vertices_indices.push(edge_indices);
                }

                // FACE VERTICES (6 faces)
                // Each face is a flat surface with rounded edges

                // Define the 6 faces with their center positions and normals
                let faces_data: [((f64, f64, f64), (f64, f64, f64)); 6] = [
                    ((0.0, -h2, 0.0), (0.0, -1.0, 0.0)), // Bottom face (y-)
                    ((0.0, h2, 0.0), (0.0, 1.0, 0.0)),   // Top face (y+)
                    ((w2, 0.0, 0.0), (1.0, 0.0, 0.0)),   // Right face (x+)
                    ((-w2, 0.0, 0.0), (-1.0, 0.0, 0.0)),  // Left face (x-)
                    ((0.0, 0.0, -d2), (0.0, 0.0, -1.0)),  // Front face (z-)
                    ((0.0, 0.0, d2), (0.0, 0.0, 1.0))     // Back face (z+)
                ];

                let face_segments = 6; // Number of segments for face grid

                for &((fx, fy, fz), (nx, ny, nz)) in faces_data.iter() {
                    // Determine which axes are in the face plane
                    let use_x = nx.abs() < 0.5;
                    let use_y = ny.abs() < 0.5;
                    let use_z = nz.abs() < 0.5;

                    // Calculate face dimensions (accounting for bevels)
                    let face_w = if use_x { w - 2.0 * b } else { 0.0 };
                    let face_h = if use_y { h - 2.0 * b } else { 0.0 };
                    let face_d = if use_z { d - 2.0 * b } else { 0.0 };

                    // Calculate face center (moved inward by normal * b)
                    let face_center: (f64, f64, f64) = (
                        fx - nx * b,
                        fy - ny * b,
                        fz - nz * b
                    );

                    // Create a grid of vertices for the face
                    for i in 0..=face_segments {
                        for j in 0..=face_segments {
                            // Grid positions from -1 to 1
                            let u = (i as f64 / face_segments as f64) * 2.0 - 1.0;
                            let v = (j as f64 / face_segments as f64) * 2.0 - 1.0;

                            // Skip corners and edges (handled separately)
                            let u_abs = u.abs();
                            let v_abs = v.abs();
                            if (u_abs > 0.8 && v_abs > 0.8) ||
                                (u_abs < 0.2 && v_abs < 0.2) {
                                continue;
                            }

                            // Calculate vertex position based on face orientation
                            let vertex: (f64, f64, f64) = if nx.abs() > 0.5 {
                                // X-aligned face
                                (
                                    face_center.0,
                                    face_center.1 + v * face_h / 2.0,
                                    face_center.2 + u * face_d / 2.0
                                )
                            } else if ny.abs() > 0.5 {
                                // Y-aligned face
                                (
                                    face_center.0 + u * face_w / 2.0,
                                    face_center.1,
                                    face_center.2 + v * face_d / 2.0
                                )
                            } else {
                                // Z-aligned face
                                (
                                    face_center.0 + u * face_w / 2.0,
                                    face_center.1 + v * face_h / 2.0,
                                    face_center.2
                                )
                            };

                            vertices.push(vertex);
                        }
                    }
                }

                // Add central vertices for all 6 faces
                // (useful for creating the main face planes)
                let face_centers: [(f64, f64, f64); 6] = [
                    (0.0, -h2 + b, 0.0), // Bottom
                    (0.0, h2 - b, 0.0),  // Top
                    (w2 - b, 0.0, 0.0),  // Right
                    (-w2 + b, 0.0, 0.0), // Left
                    (0.0, 0.0, -d2 + b), // Front
                    (0.0, 0.0, d2 - b)   // Back
                ];

                for &center in face_centers.iter() {
                    vertices.push(center);
                }

                // Final result
                vertices
            },
            Shape3D::Cylinder(radius, height) => {
                // Create a low-poly cylinder approximation
                let num_segments = 16;
                let h2 = height / 2.0;
                let mut vertices: Vec<(f64, f64, f64)> = Vec::with_capacity(num_segments * 2 + 2);

                // Center points for top and bottom faces
                vertices.push((0.0, h2, 0.0));
                vertices.push((0.0, -h2, 0.0));

                // Create points around the circumference
                for i in 0..num_segments {
                    let angle = 2.0 * PI * (i as f64) / (num_segments as f64);
                    let x = radius * angle.cos();
                    let z = radius * angle.sin();

                    // Top rim
                    vertices.push((x, h2, z));
                    // Bottom rim
                    vertices.push((x, -h2, z));
                }

                vertices
            },
            Shape3D::Polyhedron(vertices, _) => {
                // Just return the existing vertices
                vertices.clone()
            }
        }
    }

    /// Creates faces (indices) for the shape
    /// Returns a vector of face indices where each face is a vector of vertex indices
    pub fn create_faces(&self) -> Vec<Vec<usize>> {
        match self {
            Shape3D::Sphere(_) => {
                // Create faces for icosphere (simplified version with 20 triangular faces)
                vec![
                    vec![0, 8, 4], vec![0, 4, 6], vec![0, 6, 10], vec![0, 10, 2], vec![0, 2, 8],
                    vec![1, 9, 5], vec![1, 5, 7], vec![1, 7, 11], vec![1, 11, 3], vec![1, 3, 9],
                    vec![4, 8, 9], vec![4, 9, 5], vec![6, 4, 5], vec![6, 5, 7], vec![10, 6, 7],
                    vec![10, 7, 11], vec![2, 10, 11], vec![2, 11, 3], vec![8, 2, 3], vec![8, 3, 9]
                ]
            },
            Shape3D::Cuboid(_, _, _) => {
                // 6 faces of a cuboid
                vec![
                    vec![0, 1, 2, 3], // Bottom
                    vec![4, 5, 6, 7], // Top
                    vec![0, 4, 7, 3], // Left
                    vec![1, 5, 6, 2], // Right
                    vec![0, 1, 5, 4], // Front
                    vec![3, 2, 6, 7]  // Back
                ]
            },
            Shape3D::BeveledCuboid(_, _, _, _) => {
                let mut faces = Vec::new();
                let vertices = self.create_vertices();

                // Based on the vertex generation in your implementation:
                // 1. The last 6 vertices are the face centers
                let num_vertices = vertices.len();
                let face_centers_start = num_vertices - 6;

                // Find and collect vertices for each face
                // This requires knowing the pattern in which vertices were generated

                // For each face center, identify the surrounding vertices
                // that form the face plane, and connect them
                for face_idx in 0..6 {
                    let center_idx = face_centers_start + face_idx;
                    let center = vertices[center_idx];

                    // Find vertices that belong to this face by checking their coordinates
                    // against the center's position and the face normal
                    let mut face_vertices = Vec::new();

                    // Determine face normal based on face index
                    let face_normal = match face_idx {
                        0 => (0.0, -1.0, 0.0), // Bottom
                        1 => (0.0, 1.0, 0.0),  // Top
                        2 => (1.0, 0.0, 0.0),  // Right
                        3 => (-1.0, 0.0, 0.0), // Left
                        4 => (0.0, 0.0, -1.0), // Front
                        5 => (0.0, 0.0, 1.0),  // Back
                        _ => unreachable!()
                    };

                    // Find vertices that are on this face plane
                    for (i, &vertex) in vertices.iter().enumerate() {
                        // Skip the center vertex itself
                        if i == center_idx {
                            continue;
                        }

                        // Calculate vector from center to vertex
                        let vec = (
                            vertex.0 - center.0,
                            vertex.1 - center.1,
                            vertex.2 - center.2
                        );

                        // Calculate dot product with normal
                        let dot = vec.0 * face_normal.0 + vec.1 * face_normal.1 + vec.2 * face_normal.2;

                        // If dot product is close to zero, vertex is on the face plane
                        if dot.abs() < 0.01 {
                            // Calculate distance from center in face plane
                            let dist_sq =
                                vec.0 * vec.0 +
                                    vec.1 * vec.1 +
                                    vec.2 * vec.2 -
                                    dot * dot;

                            // If within face bounds, add to face vertices
                            if dist_sq < 0.25 { // Adjust this threshold as needed
                                face_vertices.push(i);
                            }
                        }
                    }

                    // Sort face vertices in circular order around the center
                    // (This is complex but necessary for proper face triangulation)
                    face_vertices.sort_by(|&a, &b| {
                        let va = vertices[a];
                        let vb = vertices[b];

                        // Calculate vectors from center to each vertex
                        let vec_a = (va.0 - center.0, va.1 - center.1, va.2 - center.2);
                        let vec_b = (vb.0 - center.0, vb.1 - center.1, vb.2 - center.2);

                        // Calculate angles in the face plane
                        let angle_a = vec_a.0.atan2(vec_a.1);
                        let angle_b = vec_b.0.atan2(vec_b.1);

                        angle_a.partial_cmp(&angle_b).unwrap_or(std::cmp::Ordering::Equal)
                    });

                    // Create triangular faces by connecting center to each pair of adjacent vertices
                    for i in 0..face_vertices.len() {
                        let next_i = (i + 1) % face_vertices.len();
                        faces.push(vec![
                            center_idx,
                            face_vertices[i],
                            face_vertices[next_i]
                        ]);
                    }
                }
                faces
            },
            Shape3D::Cylinder(_, _) => {
                let num_segments = 16;
                let mut faces = Vec::new();

                // Top face (fan from center)
                let mut top_face = vec![0]; // Center point
                for i in 0..num_segments {
                    top_face.push(2 + i*2);
                }
                top_face.push(2); // Close the loop
                faces.push(top_face);

                // Bottom face (fan from center)
                let mut bottom_face = vec![1]; // Center point
                for i in (0..num_segments).rev() {
                    bottom_face.push(3 + i*2);
                }
                bottom_face.push(3 + (num_segments-1)*2); // Close the loop
                faces.push(bottom_face);

                // Side faces (quads)
                for i in 0..num_segments {
                    let next_i = (i + 1) % num_segments;
                    faces.push(vec![
                        2 + i*2,             // Top current
                        2 + next_i*2,        // Top next
                        3 + next_i*2,        // Bottom next
                        3 + i*2              // Bottom current
                    ]);
                }

                faces
            },
            Shape3D::Polyhedron(_, faces) => {
                // Return the existing faces
                faces.clone()
            }
        }
    }

    /// Checks if a point is inside the shape
    pub fn contains_point(&self, point: (f64, f64, f64)) -> bool {
        let (x, y, z) = point;

        match self {
            Shape3D::Sphere(radius) => {
                // Point is inside if distance from center <= radius
                x.powi(2) + y.powi(2) + z.powi(2) <= radius.powi(2)
            },
            Shape3D::Cuboid(w, h, d) => {
                // Point is inside if within all dimensions
                let w2 = w / 2.0;
                let h2 = h / 2.0;
                let d2 = d / 2.0;

                x.abs() <= w2 && y.abs() <= h2 && z.abs() <= d2
            },
            Shape3D::BeveledCuboid(w, h, d, bevel) => {
                // Check if inside the main cuboid
                let w2 = w / 2.0;
                let h2 = h / 2.0;
                let d2 = d / 2.0;

                if x.abs() <= w2 - bevel && y.abs() <= h2 - bevel && z.abs() <= d2 - bevel {
                    return true;
                }

                // Check edges and corners (simplified approximation)
                // This is a simplified check that might not be perfect for all points
                // For fully accurate check, we'd need to test against all beveled faces
                let dx = x.abs() - (w2 - bevel);
                let dy = y.abs() - (h2 - bevel);
                let dz = z.abs() - (d2 - bevel);

                if dx <= 0.0 || dy <= 0.0 || dz <= 0.0 {
                    // Inside the main cuboid region including one face
                    return true;
                }

                // Check if within the beveled corner sphere
                dx.powi(2) + dy.powi(2) + dz.powi(2) <= bevel.powi(2)
            },
            Shape3D::Cylinder(radius, height) => {
                let h2 = height / 2.0;

                // Check if point is within cylinder bounds
                x.powi(2) + z.powi(2) <= radius.powi(2) && y.abs() <= h2
            },
            Shape3D::Polyhedron(_, _) => {
                // Point-in-polyhedron check is complex
                // A full implementation would use ray casting or similar algorithm
                // For simplicity, we'll just check against the bounding sphere
                let radius = self.bounding_radius();
                let (cx, cy, cz) = self.center_of_mass();

                ((x - cx).powi(2) + (y - cy).powi(2) + (z - cz).powi(2)) <= radius.powi(2)
            }
        }
    }

    fn is_cuboid_cuboid_collision(shape_1: &Shape3D, shape_2: &Shape3D) -> bool {
        match (shape_1, shape_2) {
            (Shape3D::Cuboid(w1, h1, d1), Shape3D::Cuboid(w2, h2, d2)) |
            (Shape3D::BeveledCuboid(w1, h1, d1, _), Shape3D::BeveledCuboid(w2, h2, d2, _)) |
            (Shape3D::Cuboid(w1, h1, d1), Shape3D::BeveledCuboid(w2, h2, d2, _)) |
            (Shape3D::BeveledCuboid(w1, h1, d1, _), Shape3D::Cuboid(w2, h2, d2)) => true,
            _ => false
        }
    }

    /// Checks for collision between two shapes
    pub fn check_collision(
        &self,
        position1: (f64, f64, f64),
        other: &Shape3D,
        position2: (f64, f64, f64)
    ) -> bool {
        // First, do a quick bounding sphere check for early rejection
        let dx = position2.0 - position1.0;
        let dy = position2.1 - position1.1;
        let dz = position2.2 - position1.2;

        let distance_squared = dx.powi(2) + dy.powi(2) + dz.powi(2);
        let r1 = self.bounding_radius();
        let r2 = other.bounding_radius();

        if distance_squared > (r1 + r2).powi(2) {
            return false;
        }

        // More specific collision detection based on shape types
        match (self, other) {
            // Sphere-sphere collision
            (Shape3D::Sphere(r1), Shape3D::Sphere(r2)) => {
                distance_squared <= (r1 + r2).powi(2)
            },

            // Cuboid-cuboid collision (AABB)
            (Shape3D::Cuboid(w1, h1, d1), Shape3D::Cuboid(w2, h2, d2)) |
            (Shape3D::BeveledCuboid(w1, h1, d1, _), Shape3D::BeveledCuboid(w2, h2, d2, _)) |
            (Shape3D::Cuboid(w1, h1, d1), Shape3D::BeveledCuboid(w2, h2, d2, _)) |
            (Shape3D::BeveledCuboid(w1, h1, d1, _), Shape3D::Cuboid(w2, h2, d2)) => {
                // Axis-Aligned Bounding Box test
                let w1_half = w1 / 2.0;
                let h1_half = h1 / 2.0;
                let d1_half = d1 / 2.0;

                let w2_half = w2 / 2.0;
                let h2_half = h2 / 2.0;
                let d2_half = d2 / 2.0;

                // Check overlap in all three axes
                dx.abs() <= (w1_half + w2_half) &&
                    dy.abs() <= (h1_half + h2_half) &&
                    dz.abs() <= (d1_half + d2_half)
            },

            // Sphere-cuboid collision
            (Shape3D::Sphere(radius), Shape3D::Cuboid(w, h, d)) |
            (Shape3D::Sphere(radius), Shape3D::BeveledCuboid(w, h, d, _)) => {
                // Find the closest point on the cuboid to the sphere
                let w_half = w / 2.0;
                let h_half = h / 2.0;
                let d_half = d / 2.0;

                // Find the closest point on the cuboid (clamped to cuboid bounds)
                let closest_x = (-dx).clamp(-w_half, w_half);
                let closest_y = (-dy).clamp(-h_half, h_half);
                let closest_z = (-dz).clamp(-d_half, d_half);

                // Calculate squared distance from closest point to sphere center
                let closest_dist_sq =
                    (dx + closest_x).powi(2) +
                        (dy + closest_y).powi(2) +
                        (dz + closest_z).powi(2);

                // Collision if the closest point is within the sphere
                closest_dist_sq <= radius.powi(2)
            },

            // Cuboid-sphere collision (reverse of above)
            (Shape3D::Cuboid(w, h, d), Shape3D::Sphere(radius)) |
            (Shape3D::BeveledCuboid(w, h, d, _), Shape3D::Sphere(radius)) => {
                // Just use the same logic with positions reversed
                let w_half = w / 2.0;
                let h_half = h / 2.0;
                let d_half = d / 2.0;

                let closest_x = dx.clamp(-w_half, w_half);
                let closest_y = dy.clamp(-h_half, h_half);
                let closest_z = dz.clamp(-d_half, d_half);

                let closest_dist_sq =
                    (dx - closest_x).powi(2) +
                        (dy - closest_y).powi(2) +
                        (dz - closest_z).powi(2);

                closest_dist_sq <= radius.powi(2)
            },

            // Cylinder collisions
            (Shape3D::Cylinder(r1, h1), Shape3D::Cylinder(r2, h2)) => {
                // Check if cylinders overlap
                // First check height (y-axis)
                let h1_half = h1 / 2.0;
                let h2_half = h2 / 2.0;

                if dy.abs() > (h1_half + h2_half) {
                    return false;
                }

                // Then check radius (xz-plane)
                let xz_dist_sq = dx.powi(2) + dz.powi(2);
                xz_dist_sq <= (r1 + r2).powi(2)
            },

            // For all other combinations, use simpler approximations
            _ => {
                // Fallback to the bounding sphere check we already did
                true
            }
        }
    }

    /// Calculates the collision normal between this shape and another
    /// Returns the normal vector pointing from this shape to the other
    pub fn collision_normal(
        &self,
        position1: (f64, f64, f64),
        other: &Shape3D,
        position2: (f64, f64, f64)
    ) -> Option<(f64, f64, f64)> {
        if !self.check_collision(position1, other, position2) {
            return None;
        }

        // Vector from position1 to position2
        let dx = position2.0 - position1.0;
        let dy = position2.1 - position1.1;
        let dz = position2.2 - position1.2;

        let distance_squared = dx.powi(2) + dy.powi(2) + dz.powi(2);

        // If centers are at same position, return a default normal
        if distance_squared < 1e-10 {
            return Some((0.0, 0.0, 1.0));
        }

        // For sphere-sphere collision, the normal is the center-to-center vector
        match (self, other) {
            (Shape3D::Sphere(_), Shape3D::Sphere(_)) => {
                let distance = distance_squared.sqrt();
                Some((dx / distance, dy / distance, dz / distance))
            },

            // For cuboid collisions, find the minimum penetration axis
            (Shape3D::Cuboid(w1, h1, d1), Shape3D::Cuboid(w2, h2, d2)) |
            (Shape3D::BeveledCuboid(w1, h1, d1, _), Shape3D::BeveledCuboid(w2, h2, d2, _)) |
            (Shape3D::Cuboid(w1, h1, d1), Shape3D::BeveledCuboid(w2, h2, d2, _)) |
            (Shape3D::BeveledCuboid(w1, h1, d1, _), Shape3D::Cuboid(w2, h2, d2)) => {
                let w1_half = w1 / 2.0;
                let h1_half = h1 / 2.0;
                let d1_half = d1 / 2.0;

                let w2_half = w2 / 2.0;
                let h2_half = h2 / 2.0;
                let d2_half = d2 / 2.0;

                // Calculate penetration depths in each axis
                let x_overlap = w1_half + w2_half - dx.abs();
                let y_overlap = h1_half + h2_half - dy.abs();
                let z_overlap = d1_half + d2_half - dz.abs();

                // Find minimum penetration axis
                if x_overlap < y_overlap && x_overlap < z_overlap {
                    // X-axis has minimum penetration
                    Some((dx.signum(), 0.0, 0.0))
                } else if y_overlap < z_overlap {
                    // Y-axis has minimum penetration
                    Some((0.0, dy.signum(), 0.0))
                } else {
                    // Z-axis has minimum penetration
                    Some((0.0, 0.0, dz.signum()))
                }
            },

            // For other combinations, use center-to-center as approximation
            _ => {
                // Normalize the vector
                let distance = distance_squared.sqrt();
                Some((dx / distance, dy / distance, dz / distance))
            }
        }
    }

    /// Gets the face number (1-6) that corresponds to the normal direction
    /// Useful for determining which face of a die is pointing in a particular direction
    pub fn face_from_normal(&self, normal: (f64, f64, f64)) -> Option<u8> {
        match self {
            Shape3D::BeveledCuboid(_, _, _, _) => {
                // For a die, map the normal direction to a face number
                // Standard dice have opposite faces sum to 7
                let (nx, ny, nz) = normal;

                // Find which component has the largest absolute value
                let abs_nx = nx.abs();
                let abs_ny = ny.abs();
                let abs_nz = nz.abs();

                if abs_nx >= abs_ny && abs_nx >= abs_nz {
                    // X-axis dominant
                    if nx > 0.0 {
                        Some(2) // Right face
                    } else {
                        Some(5) // Left face
                    }
                } else if abs_ny >= abs_nx && abs_ny >= abs_nz {
                    // Y-axis dominant
                    if ny > 0.0 {
                        Some(1) // Top face
                    } else {
                        Some(6) // Bottom face
                    }
                } else {
                    // Z-axis dominant
                    if nz > 0.0 {
                        Some(4) // Back face
                    } else {
                        Some(3) // Front face
                    }
                }
            },
            _ => None // Not applicable for non-die shapes
        }
    }

    /// Returns a vector string representation of the shape for debugging
    pub fn shape_type_string(&self) -> String {
        match self {
            Shape3D::Sphere(radius) => format!("Sphere(radius={})", radius),
            Shape3D::Cuboid(w, h, d) => format!("Cuboid({}×{}×{})", w, h, d),
            Shape3D::BeveledCuboid(w, h, d, b) => format!("BeveledCuboid({}×{}×{}, bevel={})", w, h, d, b),
            Shape3D::Cylinder(r, h) => format!("Cylinder(radius={}, height={})", r, h),
            Shape3D::Polyhedron(vertices, faces) =>
                format!("Polyhedron({} vertices, {} faces)", vertices.len(), faces.len()),
        }
    }

    /// Transforms a point from local to world space based on position and orientation
    pub fn transform_point(
        &self,
        local_point: (f64, f64, f64),
        position: (f64, f64, f64),
        orientation: (f64, f64, f64)
    ) -> (f64, f64, f64) {
        // First rotate the point
        let rotated: (f64, f64, f64) = rotate_point(local_point, orientation);

        // Then translate to world space
        (
            rotated.0 + position.0,
            rotated.1 + position.1,
            rotated.2 + position.2
        )
    }
}

/// Physical object with 3D shape, position, velocity, and orientation
#[derive(Debug, Clone)]
pub struct PhysicalObject3D {
    /// The base physics object
    pub object: ObjectIn3D,
    /// The shape of the object
    pub shape: Shape3D,
    /// Material properties of the object
    pub material: Option<Material>,
    /// Angular velocity in radians per second (around x, y, z axes)
    pub angular_velocity: (f64, f64, f64),
    /// Orientation as Euler angles in radians (roll, pitch, yaw)
    pub orientation: (f64, f64, f64),
}

impl PhysicalObject3D {
    /// Creates a new physical object with the given parameters
    pub fn new(
        mass: f64,
        velocity: (f64, f64, f64),
        position: (f64, f64, f64),
        shape: Shape3D,
        material: Option<Material>,
        angular_velocity: (f64, f64, f64),
        orientation: (f64, f64, f64)
    ) -> Self {
        PhysicalObject3D {
            object: ObjectIn3D::new(
                mass,
                velocity.0,
                velocity.1,
                velocity.2,
                position
            ),
            shape,
            material,
            angular_velocity,
            orientation,
        }
    }

    /// Creates a new die (standard six-sided die) with the given parameters
    pub fn new_die(
        mass: f64,
        size: f64,
        bevel: f64,
        position: (f64, f64, f64),
        velocity: (f64, f64, f64),
        angular_velocity: (f64, f64, f64),
        orientation: (f64, f64, f64),
        material: Option<Material>
    ) -> Self {
        Self::new(
            mass,
            velocity,
            position,
            Shape3D::new_die(size, bevel),
            material,
            angular_velocity,
            orientation
        )
    }

    /// Updates the object's position and orientation based on velocities
    pub fn update(&mut self, dt: f64) {
        // Update position based on linear velocity
        self.object.position.x += self.object.velocity.x * dt;
        self.object.position.y += self.object.velocity.y * dt;
        self.object.position.z += self.object.velocity.z * dt;

        // Update orientation based on angular velocity
        self.orientation.0 += self.angular_velocity.0 * dt;
        self.orientation.1 += self.angular_velocity.1 * dt;
        self.orientation.2 += self.angular_velocity.2 * dt;

        // Normalize angles to [0, 2π)
        self.orientation.0 = normalize_angle(self.orientation.0);
        self.orientation.1 = normalize_angle(self.orientation.1);
        self.orientation.2 = normalize_angle(self.orientation.2);
    }

    /// Applies a force to the object
    pub fn apply_force(&mut self, force: (f64, f64, f64), duration: f64) -> Result<(), &'static str> {
        crate::interactions::apply_force_3d(
            &crate::utils::DEFAULT_PHYSICS_CONSTANTS,
            &mut self.object,
            force,
            duration
        )
    }

    /// Applies a torque (rotational force) to the object
    pub fn apply_torque(&mut self, torque: (f64, f64, f64), duration: f64) {
        // Get the moment of inertia tensor
        let inertia = self.shape.moment_of_inertia(self.object.mass);

        // Apply torque (simplified model)
        self.angular_velocity.0 += torque.0 * duration / inertia[0];
        self.angular_velocity.1 += torque.1 * duration / inertia[1];
        self.angular_velocity.2 += torque.2 * duration / inertia[2];
    }

    /// Gets the current coefficient of restitution for collisions
    pub fn get_restitution(&self) -> f64 {
        if let Some(ref material) = self.material {
            material.restitution_coefficient
        } else {
            0.5 // Default value
        }
    }

    /// Checks for collision with another physical object
    pub fn collides_with(&self, other: &PhysicalObject3D) -> bool {
        let pos1: (f64, f64, f64) = self.object.position.to_coord();
        let pos2: (f64, f64, f64) = other.object.position.to_coord();

        self.shape.check_collision(pos1, &other.shape, pos2)
    }

    /// Handles collision between this object and another
    pub fn handle_collision(
        &mut self,
        other: &mut PhysicalObject3D,
        dt: f64
    ) -> bool {
        let pos1: (f64, f64, f64) = self.object.position.to_coord();
        let pos2: (f64, f64, f64) = other.object.position.to_coord();

        // Get collision normal
        if let Some(normal) = self.shape.collision_normal(pos1, &other.shape, pos2) {
            // Calculate combined coefficient of restitution
            let restitution = (self.get_restitution() + other.get_restitution()) / 2.0;

            // Handle linear collision
            let _ = crate::interactions::elastic_collision_3d(
                &crate::utils::DEFAULT_PHYSICS_CONSTANTS,
                &mut self.object,
                &mut other.object,
                normal,
                dt,
                0.1, // Drag coefficient
                1.0  // Cross-sectional area
            );

            // Calculate impact point (approximation at the midpoint between objects)
            let impact_point = (
                (pos1.0 + pos2.0) / 2.0,
                (pos1.1 + pos2.1) / 2.0,
                (pos1.2 + pos2.2) / 2.0
            );

            // Calculate relative position vectors from center to impact point
            let r1 = (
                impact_point.0 - pos1.0,
                impact_point.1 - pos1.1,
                impact_point.2 - pos1.2
            );

            let r2 = (
                impact_point.0 - pos2.0,
                impact_point.1 - pos2.1,
                impact_point.2 - pos2.2
            );

            // Calculate relative velocity at impact point
            let v1 = (
                self.object.velocity.x +
                    self.angular_velocity.1 * r1.2 - self.angular_velocity.2 * r1.1,
                self.object.velocity.y +
                    self.angular_velocity.2 * r1.0 - self.angular_velocity.0 * r1.2,
                self.object.velocity.z +
                    self.angular_velocity.0 * r1.1 - self.angular_velocity.1 * r1.0
            );

            let v2 = (
                other.object.velocity.x +
                    other.angular_velocity.1 * r2.2 - other.angular_velocity.2 * r2.1,
                other.object.velocity.y +
                    other.angular_velocity.2 * r2.0 - other.angular_velocity.0 * r2.2,
                other.object.velocity.z +
                    other.angular_velocity.0 * r2.1 - other.angular_velocity.1 * r2.0
            );

            // Calculate relative velocity
            let vrel = (
                v1.0 - v2.0,
                v1.1 - v2.1,
                v1.2 - v2.2
            );

            // Calculate impulse magnitude (simplified)
            let impulse_mag = (-(1.0 + restitution) * dot_product(vrel, normal)) /
                (1.0 / self.object.mass + 1.0 / other.object.mass);

            // Apply impulse to angular velocities
            let impulse = (
                normal.0 * impulse_mag,
                normal.1 * impulse_mag,
                normal.2 * impulse_mag
            );

            // Calculate torque from impulse
            let torque1: (f64, f64, f64) = cross_product(r1, impulse);
            let torque2: (f64, f64, f64) = cross_product(r2, (-impulse.0, -impulse.1, -impulse.2));

            // Get moments of inertia
            let inertia1: [f64; 6] = self.shape.moment_of_inertia(self.object.mass);
            let inertia2: [f64; 6] = other.shape.moment_of_inertia(other.object.mass);

            // Update angular velocities
            self.angular_velocity.0 += torque1.0 / inertia1[0] * dt;
            self.angular_velocity.1 += torque1.1 / inertia1[1] * dt;
            self.angular_velocity.2 += torque1.2 / inertia1[2] * dt;

            other.angular_velocity.0 += torque2.0 / inertia2[0] * dt;
            other.angular_velocity.1 += torque2.1 / inertia2[1] * dt;
            other.angular_velocity.2 += torque2.2 / inertia2[2] * dt;

            return true;
        }

        false
    }

    /// Gets the vertices of the shape in world space
    pub fn world_vertices(&self) -> Vec<(f64, f64, f64)> {
        let local_vertices = self.shape.create_vertices();
        let position = self.object.position.to_coord();

        // Transform local vertices to world space
        local_vertices.iter()
            .map(|v| self.shape.transform_point(*v, position, self.orientation))
            .collect()
    }

    /// Gets the faces (vertices) of the shape in world space
    pub fn world_faces(&self) -> Vec<Vec<(f64, f64, f64)>> {
        let local_vertices = self.shape.create_vertices();
        let faces = self.shape.create_faces();
        let position = self.object.position.to_coord();

        // Transform faces to world space
        faces.iter().map(|face| {
            face.iter().map(|&idx| {
                let vertex = local_vertices[idx];
                self.shape.transform_point(vertex, position, self.orientation)
            }).collect()
        }).collect()
    }

    /// For dice, determines which face is currently facing up
    pub fn die_face_up(&self) -> Option<u8> {
        // Only applicable to beveled cuboids (dice)
        if let Shape3D::BeveledCuboid(_, _, _, _) = self.shape {
            // The up direction in world space
            let up = (0.0, 1.0, 0.0);

            // Transform the up vector to object space (inverse of orientation)
            let inverted_orientation = (
                -self.orientation.0,
                -self.orientation.1,
                -self.orientation.2
            );

            let obj_up = rotate_point(up, inverted_orientation);

            // Find which face normal is most aligned with the up direction
            self.shape.face_from_normal(obj_up)
        } else {
            None
        }
    }

    /// Apply damping to angular and linear velocities (simulates friction and air resistance)
    pub fn apply_damping(&mut self, linear_damping: f64, angular_damping: f64, dt: f64) {
        // Apply linear damping
        let linear_factor = (1.0 - linear_damping * dt).max(0.0);
        self.object.velocity.x *= linear_factor;
        self.object.velocity.y *= linear_factor;
        self.object.velocity.z *= linear_factor;

        // Apply angular damping
        let angular_factor = (1.0 - angular_damping * dt).max(0.0);
        self.angular_velocity.0 *= angular_factor;
        self.angular_velocity.1 *= angular_factor;
        self.angular_velocity.2 *= angular_factor;
    }

    /// Apply gravity to the object
    pub fn apply_gravity(&mut self, gravity: f64, dt: f64) {
        self.object.velocity.y -= gravity * dt;
    }

    /// Check if the object is at rest (stopped moving)
    pub fn is_at_rest(&self, linear_threshold: f64, angular_threshold: f64) -> bool {
        // Calculate linear and angular speed
        let linear_speed = (
            self.object.velocity.x.powi(2) +
                self.object.velocity.y.powi(2) +
                self.object.velocity.z.powi(2)
        ).sqrt();

        let angular_speed = (
            self.angular_velocity.0.powi(2) +
                self.angular_velocity.1.powi(2) +
                self.angular_velocity.2.powi(2)
        ).sqrt();

        // Check if both are below threshold
        linear_speed < linear_threshold && angular_speed < angular_threshold
    }
}

/// Normalizes an angle to the range [0, 2π)
pub fn normalize_angle(angle: f64) -> f64 {
    let two_pi = 2.0 * PI;
    ((angle % two_pi) + two_pi) % two_pi
}

/// Rotates a point using Euler angles (roll, pitch, yaw)
pub fn rotate_point(point: (f64, f64, f64), orientation: (f64, f64, f64)) -> (f64, f64, f64) {
    let (roll, pitch, yaw) = orientation;

    // First, rotate around Z axis (roll)
    let x1 = point.0 * roll.cos() - point.1 * roll.sin();
    let y1 = point.0 * roll.sin() + point.1 * roll.cos();
    let z1 = point.2;

    // Then, rotate around X axis (pitch)
    let x2 = x1;
    let y2 = y1 * pitch.cos() - z1 * pitch.sin();
    let z2 = y1 * pitch.sin() + z1 * pitch.cos();

    // Finally, rotate around Y axis (yaw)
    let x3 = x2 * yaw.cos() + z2 * yaw.sin();
    let y3 = y2;
    let z3 = -x2 * yaw.sin() + z2 * yaw.cos();

    (x3, y3, z3)
}