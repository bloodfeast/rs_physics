use crate::interactions::{cross_product, dot_product};
use crate::materials::Material;
use crate::models::{ObjectIn3D, ToCoordinates};
use std::f64::consts::PI;
use rand::Rng;
use crate::physics::PhysicsConstants;

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

    // Helper function to check overlap along an axis
    fn check_overlap_along_axis(
        corners1: &[(f64, f64, f64)],
        corners2: &[(f64, f64, f64)],
        axis: &(f64, f64, f64)
    ) -> bool {
        // Project all corners onto axis
        let projections1: Vec<f64> = corners1.iter()
            .map(|c| c.0 * axis.0 + c.1 * axis.1 + c.2 * axis.2)
            .collect();

        let projections2: Vec<f64> = corners2.iter()
            .map(|c| c.0 * axis.0 + c.1 * axis.1 + c.2 * axis.2)
            .collect();

        // Find min and max projections
        let min1 = projections1.iter().fold(f64::MAX, |a, &b| a.min(b));
        let max1 = projections1.iter().fold(f64::MIN, |a, &b| a.max(b));
        let min2 = projections2.iter().fold(f64::MAX, |a, &b| a.min(b));
        let max2 = projections2.iter().fold(f64::MIN, |a, &b| a.max(b));

        // Check for overlap
        max1 >= min2 && max2 >= min1
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
    pub physics_constants: PhysicsConstants
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
        orientation: (f64, f64, f64),
        physics_constants: PhysicsConstants
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
            physics_constants,
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
        material: Option<Material>,
        physics_constants: PhysicsConstants
    ) -> Self {
        Self::new(
            mass,
            velocity,
            position,
            Shape3D::new_die(size, bevel),
            material,
            angular_velocity,
            orientation,
            physics_constants,
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

        // Handle ground collision based on shape type
        match &self.shape {
            Shape3D::BeveledCuboid(_width, _height, _depth, _) => {
                // Calculate the positions of the 8 corners of the cube
                let corners = self.get_corner_positions();

                // Find the lowest point (corner closest to the ground)
                let min_y = corners.iter().map(|(_,y,_)| *y).fold(f64::MAX, f64::min);

                // If the lowest point is below ground level
                if min_y < self.physics_constants.ground_level {
                    self.handle_cuboid_ground_collision(min_y, &corners, dt);
                }
            },
            Shape3D::Cuboid(_width, _height, _depth) => {
                // Reuse the same code for regular cuboids
                let corners: [(f64, f64, f64); 8] = self.get_corner_positions();
                let min_y = corners.iter().map(|(_,y,_)| *y).fold(f64::MAX, f64::min);

                if min_y < self.physics_constants.ground_level {
                    self.handle_cuboid_ground_collision(min_y, &corners, dt);
                }
            },
            Shape3D::Sphere(radius) => {
                // For sphere, just check if the bottom point is below ground
                let sphere_bottom = self.object.position.y - radius;

                if sphere_bottom < self.physics_constants.ground_level {
                    self.handle_sphere_ground_collision(*radius, dt);
                }
            },
            Shape3D::Cylinder(radius, height) => {
                // For cylinder, check the bottom rim points
                let half_height = height / 2.0;
                let bottom_y = self.object.position.y - half_height;

                if bottom_y < self.physics_constants.ground_level {
                    self.handle_cylinder_ground_collision(*radius, *height, dt);
                }
            },
            Shape3D::Polyhedron(_vertices, _) => {
                // For polyhedron, transform all vertices and find lowest point
                let world_vertices = self.world_vertices();
                let min_y = world_vertices.iter().map(|(_,y,_)| *y).fold(f64::MAX, f64::min);

                if min_y < self.physics_constants.ground_level {
                    self.handle_polyhedron_ground_collision(&world_vertices, dt);
                }
            }
        }
    }

    fn handle_cuboid_ground_collision(&mut self, min_y: f64, corners: &[(f64, f64, f64)], dt: f64) {
        // Calculate penetration depth
        let penetration = self.physics_constants.ground_level - min_y;

        // Adjust position to resolve penetration
        self.object.position.y += penetration;

        // Apply bounce physics if moving downward
        if self.object.velocity.y < 0.0 {
            // Calculate which corners are in contact with the ground
            let ground_corners: Vec<(f64, f64, f64)> = corners.iter()
                .filter(|(_, y, _)| (*y - self.physics_constants.ground_level).abs() < 0.01)
                .cloned()
                .collect();

            // Only apply bounce if we have ground contact
            if !ground_corners.is_empty() {
                // Bounce with appropriate energy loss
                let restitution = self.get_restitution() * 0.8;
                self.object.velocity.y = -self.object.velocity.y * restitution;

                // Friction coefficients
                let sliding_friction = 0.7;
                let rolling_friction = 0.4;

                // Apply friction to horizontal velocity
                self.object.velocity.x *= 1.0 - sliding_friction * dt;
                self.object.velocity.z *= 1.0 - sliding_friction * dt;

                // Apply friction to angular velocity
                let angular_damping = rolling_friction * dt;
                self.angular_velocity.0 *= 1.0 - angular_damping;
                self.angular_velocity.2 *= 1.0 - angular_damping;

                // Calculate torque based on impact with ground
                if self.object.velocity.y.abs() > 0.1 {
                    // Calculate impact point (average of ground corners)
                    let impact_point: (f64, f64, f64) = if !ground_corners.is_empty() {
                        let sum = ground_corners.iter().fold((0.0, 0.0, 0.0), |acc, &p| {
                            (acc.0 + p.0, acc.1 + p.1, acc.2 + p.2)
                        });
                        let count = ground_corners.len() as f64;
                        (sum.0 / count, sum.1 / count, sum.2 / count)
                    } else {
                        (self.object.position.x, self.physics_constants.ground_level, self.object.position.z)
                    };

                    // Calculate relative vector from center to impact point
                    let r: (f64, f64, f64) = (
                        impact_point.0 - self.object.position.x,
                        impact_point.1 - self.object.position.y,
                        impact_point.2 - self.object.position.z
                    );

                    // Calculate impact force (simplified as vertical impulse)
                    let impact_force = (0.0, -self.object.velocity.y * self.object.mass, 0.0);

                    // Calculate torque as cross product
                    let torque: (f64, f64, f64) = cross_product(r, impact_force);

                    // Scale torque effect
                    let torque_factor = 0.2;
                    self.apply_torque((
                                          torque.0 * torque_factor,
                                          torque.1 * torque_factor,
                                          torque.2 * torque_factor
                                      ), dt);
                }
            }
        }
    }

    // Calculate collision impulse considering both linear and angular components
    fn calculate_collision_impulse(
        obj1: &PhysicalObject3D,
        obj2: &PhysicalObject3D,
        vrel: (f64, f64, f64),
        normal: (f64, f64, f64),
        restitution: f64,
        r1: (f64, f64, f64),
        r2: (f64, f64, f64)
    ) -> f64 {
        // Normal component of relative velocity
        let vrel_n = dot_product(vrel, normal);

        // Calculate angular contribution to impulse denominator
        let inertia1 = obj1.shape.moment_of_inertia(obj1.object.mass);
        let inertia2 = obj2.shape.moment_of_inertia(obj2.object.mass);

        // Calculate r × n for both objects
        let r1_cross_n = cross_product(r1, normal);
        let r2_cross_n = cross_product(r2, normal);

        // Calculate angular terms
        let angular_term1 =
            r1_cross_n.0 * r1_cross_n.0 / inertia1[0] +
                r1_cross_n.1 * r1_cross_n.1 / inertia1[1] +
                r1_cross_n.2 * r1_cross_n.2 / inertia1[2]
        ;

        let angular_term2 =
            r2_cross_n.0 * r2_cross_n.0 / inertia2[0] +
                r2_cross_n.1 * r2_cross_n.1 / inertia2[1] +
                r2_cross_n.2 * r2_cross_n.2 / inertia2[2]
        ;

        // Calculate full impulse magnitude with rotational components
        let impulse_denom = 1.0 / obj1.object.mass + 1.0 / obj2.object.mass + angular_term1 + angular_term2;
        let impulse_mag = -(1.0 + restitution) * vrel_n / impulse_denom;

        impulse_mag
    }

    fn handle_sphere_ground_collision(&mut self, radius: f64, dt: f64) {
        // Calculate penetration depth
        let sphere_bottom = self.object.position.y - radius;
        let penetration = self.physics_constants.ground_level - sphere_bottom;

        // Adjust position to resolve penetration
        self.object.position.y += penetration;

        // Apply bounce physics if moving downward
        if self.object.velocity.y < 0.0 {
            // Bounce with energy loss
            let restitution = self.get_restitution();
            self.object.velocity.y = -self.object.velocity.y * restitution;

            // Apply rolling friction
            let friction = 0.5; // Rolling friction coefficient

            // Calculate friction force direction (opposite to velocity)
            let speed_sq = self.object.velocity.x * self.object.velocity.x +
                self.object.velocity.z * self.object.velocity.z;

            if speed_sq > 0.001 {
                let speed = speed_sq.sqrt();
                let friction_force_x = -self.object.velocity.x / speed * friction;
                let friction_force_z = -self.object.velocity.z / speed * friction;

                // Apply friction to slow down horizontal motion
                self.object.velocity.x += friction_force_x * dt;
                self.object.velocity.z += friction_force_z * dt;

                // Convert linear velocity to angular (rolling without slipping)
                self.angular_velocity.0 = -self.object.velocity.z / radius;
                self.angular_velocity.2 = self.object.velocity.x / radius;
            }
        }
    }

    fn handle_cylinder_ground_collision(&mut self, radius: f64, height: f64, dt: f64) {
        // Calculate penetration depth
        let half_height = height / 2.0;
        let bottom_y = self.object.position.y - half_height;
        let penetration = self.physics_constants.ground_level - bottom_y;

        // Adjust position to resolve penetration
        self.object.position.y += penetration;

        // Apply bounce physics if moving downward
        if self.object.velocity.y < 0.0 {
            // Bounce with energy loss
            let restitution = self.get_restitution() * 0.9; // Cylinders bounce a bit better
            self.object.velocity.y = -self.object.velocity.y * restitution;

            // Apply friction based on orientation
            // For a cylinder, the friction depends on whether it's on its side or its end

            // Calculate the up vector in world space
            let cylinder_up: (f64, f64, f64) = rotate_point((0.0, 1.0, 0.0), self.orientation);
            let up_dot_world_up = cylinder_up.1; // Dot product with world up (0,1,0)

            // If cylinder is more vertical (on its end)
            if up_dot_world_up.abs() > 0.7 {
                // Higher friction (cylinder standing on end doesn't roll well)
                let friction = 0.8;
                self.object.velocity.x *= 1.0 - friction * dt;
                self.object.velocity.z *= 1.0 - friction * dt;

                // Little angular velocity
                self.angular_velocity.0 *= 0.9;
                self.angular_velocity.2 *= 0.9;
            } else {
                // Lower friction (cylinder can roll on its side)
                let friction = 0.3;
                self.object.velocity.x *= 1.0 - friction * dt;
                self.object.velocity.z *= 1.0 - friction * dt;

                // Calculate rolling axis (perpendicular to both cylinder axis and ground normal)
                let cylinder_axis: (f64, f64, f64) = rotate_point((0.0, 1.0, 0.0), self.orientation);
                let ground_normal: (f64, f64, f64) = (0.0, 1.0, 0.0);

                let roll_axis: (f64, f64, f64) = cross_product(cylinder_axis, ground_normal);
                let roll_axis_length = (roll_axis.0*roll_axis.0 + roll_axis.1*roll_axis.1 + roll_axis.2*roll_axis.2).sqrt();

                if roll_axis_length > 0.001 {
                    // Normalize roll axis
                    let roll_dir: (f64, f64, f64) = (
                        roll_axis.0 / roll_axis_length,
                        roll_axis.1 / roll_axis_length,
                        roll_axis.2 / roll_axis_length
                    );

                    // Rolling velocity (scalar)
                    let vel_magnitude = (self.object.velocity.x*self.object.velocity.x +
                        self.object.velocity.z*self.object.velocity.z).sqrt();

                    // Set angular velocity for rolling (proportional to linear velocity)
                    let angular_speed = vel_magnitude / radius;
                    self.angular_velocity.0 = roll_dir.0 * angular_speed;
                    self.angular_velocity.1 = roll_dir.1 * angular_speed;
                    self.angular_velocity.2 = roll_dir.2 * angular_speed;
                }
            }
        }
    }

    fn handle_polyhedron_ground_collision(&mut self, vertices: &[(f64, f64, f64)], dt: f64) {
        // Find vertices that are in contact with the ground
        let ground_vertices: Vec<(f64, f64, f64)> = vertices.iter()
            .filter(|(_, y, _)| (*y - self.physics_constants.ground_level).abs() < 0.01)
            .cloned()
            .collect();

        // Find lowest vertex
        let min_y = vertices.iter().map(|(_,y,_)| *y).fold(f64::MAX, f64::min);

        // Calculate penetration depth
        let penetration = self.physics_constants.ground_level - min_y;

        // Adjust position to resolve penetration
        self.object.position.y += penetration;

        // Only apply collision response if moving downward
        if self.object.velocity.y < 0.0 && !ground_vertices.is_empty() {
            // Bounce with energy loss
            let restitution = self.get_restitution() * 0.7; // Polyhedra tend to bounce less
            self.object.velocity.y = -self.object.velocity.y * restitution;

            // Apply friction
            let friction = 0.6;
            self.object.velocity.x *= 1.0 - friction * dt;
            self.object.velocity.z *= 1.0 - friction * dt;

            // Apply angular damping
            let angular_damping = 0.5 * dt;
            self.angular_velocity.0 *= 1.0 - angular_damping;
            self.angular_velocity.1 *= 1.0 - angular_damping;
            self.angular_velocity.2 *= 1.0 - angular_damping;

            // Calculate impact torque if we have ground contact
            if !ground_vertices.is_empty() && self.object.velocity.y.abs() > 0.1 {
                // Calculate impact point (average of ground vertices)
                let sum = ground_vertices.iter().fold((0.0, 0.0, 0.0), |acc, &p| {
                    (acc.0 + p.0, acc.1 + p.1, acc.2 + p.2)
                });
                let count = ground_vertices.len() as f64;
                let impact_point: (f64, f64, f64) = (sum.0 / count, sum.1 / count, sum.2 / count);

                // Calculate relative vector from center to impact
                let r: (f64, f64, f64) = (
                    impact_point.0 - self.object.position.x,
                    impact_point.1 - self.object.position.y,
                    impact_point.2 - self.object.position.z
                );

                // Impact force (simplified vertical impulse)
                let impact_force = (0.0, -self.object.velocity.y * self.object.mass, 0.0);

                // Calculate torque
                let torque = cross_product(r, impact_force);

                // Apply with scaling factor
                let torque_factor = 0.15; // Slightly less torque for polyhedra
                self.apply_torque((
                                      torque.0 * torque_factor,
                                      torque.1 * torque_factor,
                                      torque.2 * torque_factor
                                  ), dt);
            }
        }
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
        let inertia: [f64; 6] = self.shape.moment_of_inertia(self.object.mass);

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

    /// Transform point from local to world space
    fn transform_point_to_world(&self, local_point: (f64, f64, f64)) -> (f64, f64, f64) {
        // Rotate using current orientation
        let rotated: (f64, f64, f64) = rotate_point(local_point, self.orientation);

        // Translate to world position
        (
            rotated.0 + self.object.position.x,
            rotated.1 + self.object.position.y,
            rotated.2 + self.object.position.z
        )
    }
    /// Helper method to get all corner positions in world space
    pub fn get_corner_positions(&self) -> [(f64, f64, f64); 8] {
        if let Shape3D::BeveledCuboid(width, height, depth, _) = self.shape {
            let half_w = width / 2.0;
            let half_h = height / 2.0;
            let half_d = depth / 2.0;

            // Local corner positions (object space)
            let local_corners: [(f64, f64, f64); 8] = [
                (-half_w, -half_h, -half_d),
                (half_w, -half_h, -half_d),
                (half_w, half_h, -half_d),
                (-half_w, half_h, -half_d),
                (-half_w, -half_h, half_d),
                (half_w, -half_h, half_d),
                (half_w, half_h, half_d),
                (-half_w, half_h, half_d)
            ];

            let mut corners = [(0.0, 0.0, 0.0); 8];
            // Transform to world space
            local_corners
                .iter()
                .enumerate()
                .for_each(|(i, &local_pos)| {
                    if i > 7 {
                        eprintln!("Index out of bounds at 'get_corner_positions' -> local_corners");
                        eprintln!("Bailing out before panic can crash the program");
                        return;
                    }
                    let world_pos = self.transform_point_to_world(local_pos);
                    corners[i] = world_pos;
                });
            corners
        } else {
            [(0.0, 0.0, 0.0); 8]
        }
    }

    /// Checks for collision with another physical object
    pub fn collides_with(&self, other: &PhysicalObject3D) -> bool {
        let pos1: (f64, f64, f64) = self.object.position.to_coord();
        let pos2: (f64, f64, f64) = other.object.position.to_coord();

        // Check actual shape collision instead of just bounding sphere
        match (&self.shape, &other.shape) {
            (Shape3D::BeveledCuboid(w1, h1, d1, _), Shape3D::BeveledCuboid(w2, h2, d2, _)) => {
                // Use oriented bounding box (OBB) collision detection
                // This is more complex but more accurate

                // For now, use a more conservative AABB approach
                let w1_half = w1 / 2.0;
                let h1_half = h1 / 2.0;
                let d1_half = d1 / 2.0;

                let w2_half = w2 / 2.0;
                let h2_half = h2 / 2.0;
                let d2_half = d2 / 2.0;

                // Get the vector between centers
                let dx = pos2.0 - pos1.0;
                let dy = pos2.1 - pos1.1;
                let dz = pos2.2 - pos1.2;

                // Calculate overlap
                let overlap_x = (w1_half + w2_half) - dx.abs();
                let overlap_y = (h1_half + h2_half) - dy.abs();
                let overlap_z = (d1_half + d2_half) - dz.abs();

                // There's a collision if all axes have overlap
                overlap_x > 0.0 && overlap_y > 0.0 && overlap_z > 0.0
            },
            // Fall back to default implementation for other shapes
            _ => self.shape.check_collision(pos1, &other.shape, pos2)
        }
    }

    /// Handles collision between this object and another
    pub fn handle_collision(&mut self, other: &mut PhysicalObject3D, dt: f64) -> bool {
        let pos1: (f64, f64, f64) = self.object.position.to_coord();
        let pos2: (f64, f64, f64) = other.object.position.to_coord();

        // First perform an accurate collision detection
        if !self.shape.check_collision(pos1, &other.shape, pos2) {
            return false;
        }

        // Get collision normal
        if let Some(normal) = self.shape.collision_normal(pos1, &other.shape, pos2) {
            // Calculate combined coefficient of restitution
            let restitution = (self.get_restitution() + other.get_restitution()) / 2.0;

            // Calculate better impact point based on shape types
            let impact_point: (f64, f64, f64) = PhysicalObject3D::calculate_impact_point(&self, &other, normal);

            // Calculate relative vectors from centers to impact point
            let r1: (f64, f64, f64) = (
                impact_point.0 - pos1.0,
                impact_point.1 - pos1.1,
                impact_point.2 - pos1.2
            );

            let r2: (f64, f64, f64) = (
                impact_point.0 - pos2.0,
                impact_point.1 - pos2.1,
                impact_point.2 - pos2.2
            );

            // Calculate point velocities including rotation
            let v1: (f64, f64, f64) = PhysicalObject3D::calculate_point_velocity(self, r1);
            let v2: (f64, f64, f64) = PhysicalObject3D::calculate_point_velocity(other, r2);

            // Relative velocity at impact point
            let vrel: (f64, f64, f64) = (
                v1.0 - v2.0,
                v1.1 - v2.1,
                v1.2 - v2.2
            );

            // Calculate normal component of relative velocity
            let vrel_n = dot_product(vrel, normal);

            // Only proceed with collision response if objects are moving toward each other
            if vrel_n < 0.0 {
                // Use the existing calculate_collision_impulse function
                let impulse_mag = PhysicalObject3D::calculate_collision_impulse(
                    self, other, vrel, normal, restitution, r1, r2
                );

                // Apply impulse to linear velocities
                PhysicalObject3D::apply_linear_impulse(self, other, normal, impulse_mag);

                // Apply impulse to angular velocities
                PhysicalObject3D::apply_angular_impulse(self, other, normal, impulse_mag, r1, r2, dt);

                // Resolve penetration
                PhysicalObject3D::resolve_penetration(self, other, normal);

                return true;
            }
        }

        false
    }

    fn calculate_impact_point(
        obj1: &PhysicalObject3D,
        obj2: &PhysicalObject3D,
        normal: (f64, f64, f64)
    ) -> (f64, f64, f64) {
        let pos1: (f64, f64, f64) = obj1.object.position.to_coord();
        let pos2: (f64, f64, f64) = obj2.object.position.to_coord();

        // Default to midpoint if we can't determine better point
        let mut impact: (f64, f64, f64) = (
            (pos1.0 + pos2.0) / 2.0,
            (pos1.1 + pos2.1) / 2.0,
            (pos1.2 + pos2.2) / 2.0
        );

        match (&obj1.shape, &obj2.shape) {
            // For sphere-sphere, impact point is along the line connecting centers
            (Shape3D::Sphere(r1), Shape3D::Sphere(_r2)) => {
                // Calculate distance between centers
                let dx = pos2.0 - pos1.0;
                let dy = pos2.1 - pos1.1;
                let dz = pos2.2 - pos1.2;
                let dist = (dx*dx + dy*dy + dz*dz).sqrt();

                if dist > 0.001 {
                    // Impact point is at surface of first sphere along line to second sphere
                    impact = (
                        pos1.0 + dx/dist * r1,
                        pos1.1 + dy/dist * r1,
                        pos1.2 + dz/dist * r1
                    );
                }
            },

            // For cuboid collisions, find the closest points on each cuboid
            (Shape3D::Cuboid(_, _, _), Shape3D::Cuboid(_, _, _)) |
            (Shape3D::BeveledCuboid(_, _, _, _), Shape3D::BeveledCuboid(_, _, _, _)) |
            (Shape3D::Cuboid(_, _, _), Shape3D::BeveledCuboid(_, _, _, _)) |
            (Shape3D::BeveledCuboid(_, _, _, _), Shape3D::Cuboid(_, _, _)) => {
                // For cuboids, the closest corners can give a better impact point
                // Get world corners of both shapes
                let corners1: [(f64, f64, f64); 8] = obj1.get_corner_positions();
                let corners2: [(f64, f64, f64); 8] = obj2.get_corner_positions();

                if !corners1.is_empty() && !corners2.is_empty() {
                    // Find pair of corners with smallest distance
                    let mut min_dist = f64::MAX;
                    let mut closest_pair: ((f64, f64, f64), (f64, f64, f64)) = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0));

                    for c1 in &corners1 {
                        for c2 in &corners2 {
                            let dx = c2.0 - c1.0;
                            let dy = c2.1 - c1.1;
                            let dz = c2.2 - c1.2;
                            let dist_sq = dx*dx + dy*dy + dz*dz;

                            if dist_sq < min_dist {
                                min_dist = dist_sq;
                                closest_pair = (*c1, *c2);
                            }
                        }
                    }

                    // Impact point is midway between closest corners
                    impact = (
                        (closest_pair.0.0 + closest_pair.1.0) / 2.0,
                        (closest_pair.0.1 + closest_pair.1.1) / 2.0,
                        (closest_pair.0.2 + closest_pair.1.2) / 2.0
                    );
                }
            },

            // For sphere-cuboid, impact is on sphere surface nearest cuboid
            (Shape3D::Sphere(radius), Shape3D::Cuboid(_, _, _)) |
            (Shape3D::Sphere(radius), Shape3D::BeveledCuboid(_, _, _, _)) => {
                // Impact point is along normal direction from sphere center
                impact = (
                    pos1.0 + normal.0 * radius,
                    pos1.1 + normal.1 * radius,
                    pos1.2 + normal.2 * radius
                );
            },

            (Shape3D::Cuboid(_, _, _), Shape3D::Sphere(radius)) |
            (Shape3D::BeveledCuboid(_, _, _, _), Shape3D::Sphere(radius)) => {
                // Impact point is along normal direction from sphere center (opposite direction)
                impact = (
                    pos2.0 - normal.0 * radius,
                    pos2.1 - normal.1 * radius,
                    pos2.2 - normal.2 * radius
                );
            },

            // Default case - use midpoint between centers
            _ => { /* impact already set to midpoint */ }
        }

        impact
    }

    // Helper function to calculate velocity at a point on object
    fn calculate_point_velocity(
        obj: &PhysicalObject3D,
        r: (f64, f64, f64)
    ) -> (f64, f64, f64) {
        // Linear velocity
        let v_linear = (
            obj.object.velocity.x,
            obj.object.velocity.y,
            obj.object.velocity.z
        );

        // Angular contribution: v_angular = ω × r
        let v_angular = (
            obj.angular_velocity.1 * r.2 - obj.angular_velocity.2 * r.1,
            obj.angular_velocity.2 * r.0 - obj.angular_velocity.0 * r.2,
            obj.angular_velocity.0 * r.1 - obj.angular_velocity.1 * r.0
        );

        // Total velocity at point
        (
            v_linear.0 + v_angular.0,
            v_linear.1 + v_angular.1,
            v_linear.2 + v_angular.2
        )
    }

    // Helper function to apply linear impulse
    fn apply_linear_impulse(
        obj1: &mut PhysicalObject3D,
        obj2: &mut PhysicalObject3D,
        normal: (f64, f64, f64),
        impulse_mag: f64
    ) {
        let impulse = (
            normal.0 * impulse_mag,
            normal.1 * impulse_mag,
            normal.2 * impulse_mag
        );

        // Apply to first object
        obj1.object.velocity.x += impulse.0 / obj1.object.mass;
        obj1.object.velocity.y += impulse.1 / obj1.object.mass;
        obj1.object.velocity.z += impulse.2 / obj1.object.mass;

        // Apply to second object (opposite direction)
        obj2.object.velocity.x -= impulse.0 / obj2.object.mass;
        obj2.object.velocity.y -= impulse.1 / obj2.object.mass;
        obj2.object.velocity.z -= impulse.2 / obj2.object.mass;
    }

    // Helper function to apply angular impulse
    fn apply_angular_impulse(
        obj1: &mut PhysicalObject3D,
        obj2: &mut PhysicalObject3D,
        normal: (f64, f64, f64),
        impulse_mag: f64,
        r1: (f64, f64, f64),
        r2: (f64, f64, f64),
        dt: f64
    ) {
        let impulse = (
            normal.0 * impulse_mag,
            normal.1 * impulse_mag,
            normal.2 * impulse_mag
        );

        // Calculate torque from impulse
        let torque1 = cross_product(r1, impulse);
        let torque2 = cross_product(r2, (-impulse.0, -impulse.1, -impulse.2));

        // Get moments of inertia
        let inertia1 = obj1.shape.moment_of_inertia(obj1.object.mass);
        let inertia2 = obj2.shape.moment_of_inertia(obj2.object.mass);

        // Apply angular impulse with appropriate scaling
        let angular_response_factor = 0.8;

        obj1.angular_velocity.0 += torque1.0 / inertia1[0] * dt * angular_response_factor;
        obj1.angular_velocity.1 += torque1.1 / inertia1[1] * dt * angular_response_factor;
        obj1.angular_velocity.2 += torque1.2 / inertia1[2] * dt * angular_response_factor;

        obj2.angular_velocity.0 += torque2.0 / inertia2[0] * dt * angular_response_factor;
        obj2.angular_velocity.1 += torque2.1 / inertia2[1] * dt * angular_response_factor;
        obj2.angular_velocity.2 += torque2.2 / inertia2[2] * dt * angular_response_factor;

        // Add small random perturbation to prevent "stuck" scenarios
        let random_factor = 0.05;
        let mut rng = rand::rng();

        if impulse_mag > 0.1 {  // Only add randomness for significant collisions
            obj1.angular_velocity.0 += rng.random_range(-random_factor..random_factor);
            obj1.angular_velocity.1 += rng.random_range(-random_factor..random_factor);
            obj1.angular_velocity.2 += rng.random_range(-random_factor..random_factor);
        }
    }

    // Helper function to resolve penetration
    fn resolve_penetration(
        obj1: &mut PhysicalObject3D,
        obj2: &mut PhysicalObject3D,
        normal: (f64, f64, f64)
    ) {
        let pos1 = obj1.object.position.to_coord();
        let pos2 = obj2.object.position.to_coord();

        // Calculate penetration based on shape types
        let penetration = match (&obj1.shape, &obj2.shape) {
            (Shape3D::BeveledCuboid(w1, h1, d1, _), Shape3D::BeveledCuboid(w2, h2, d2, _)) |
            (Shape3D::Cuboid(w1, h1, d1), Shape3D::Cuboid(w2, h2, d2)) |
            (Shape3D::BeveledCuboid(w1, h1, d1, _), Shape3D::Cuboid(w2, h2, d2)) |
            (Shape3D::Cuboid(w1, h1, d1), Shape3D::BeveledCuboid(w2, h2, d2, _)) => {
                let w1_half = w1 / 2.0;
                let h1_half = h1 / 2.0;
                let d1_half = d1 / 2.0;

                let w2_half = w2 / 2.0;
                let h2_half = h2 / 2.0;
                let d2_half = d2 / 2.0;

                // Vector from obj1 to obj2
                let dx = pos2.0 - pos1.0;
                let dy = pos2.1 - pos1.1;
                let dz = pos2.2 - pos1.2;

                // Calculate overlap in each axis
                let overlap_x = (w1_half + w2_half) - dx.abs();
                let overlap_y = (h1_half + h2_half) - dy.abs();
                let overlap_z = (d1_half + d2_half) - dz.abs();

                // Find minimum overlap
                overlap_x.min(overlap_y).min(overlap_z)
            },

            (Shape3D::Sphere(r1), Shape3D::Sphere(r2)) => {
                // Calculate distance between centers
                let dx = pos2.0 - pos1.0;
                let dy = pos2.1 - pos1.1;
                let dz = pos2.2 - pos1.2;
                let distance = (dx*dx + dy*dy + dz*dz).sqrt();

                // Penetration is overlap amount
                (r1 + r2) - distance
            },

            // For other shape combinations, use a small default value
            _ => 0.01
        };

        // Apply correction only if there is actual penetration
        let correction_threshold = 0.001;
        let correction_percentage = 0.8;

        if penetration > correction_threshold {
            let correction = penetration * correction_percentage;
            let correction_vector = (
                normal.0 * correction,
                normal.1 * correction,
                normal.2 * correction
            );

            // Apply correction inversely proportional to mass
            let total_mass = obj1.object.mass + obj2.object.mass;
            let self_ratio = obj2.object.mass / total_mass;
            let other_ratio = obj1.object.mass / total_mass;

            // Move both objects apart
            obj1.object.position.x -= correction_vector.0 * self_ratio;
            obj1.object.position.y -= correction_vector.1 * self_ratio;
            obj1.object.position.z -= correction_vector.2 * self_ratio;

            obj2.object.position.x += correction_vector.0 * other_ratio;
            obj2.object.position.y += correction_vector.1 * other_ratio;
            obj2.object.position.z += correction_vector.2 * other_ratio;
        }
    }

    /// Gets the vertices of the shape in world space
    pub fn world_vertices(&self) -> Vec<(f64, f64, f64)> {
        let local_vertices: Vec<(f64, f64, f64)> = self.shape.create_vertices();
        let position: (f64, f64, f64) = self.object.position.to_coord();

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
            let up: (f64, f64, f64) = (0.0, 1.0, 0.0);

            // Transform the up vector to object space (inverse of orientation)
            let inverted_orientation: (f64, f64, f64) = (
                -self.orientation.0,
                -self.orientation.1,
                -self.orientation.2
            );

            let obj_up: (f64, f64, f64) = rotate_point(up, inverted_orientation);

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

        // Apply stronger angular damping
        let angular_factor = (1.0 - angular_damping * dt).max(0.0);
        self.angular_velocity.0 *= angular_factor;
        self.angular_velocity.1 *= angular_factor;
        self.angular_velocity.2 *= angular_factor;

        // Add threshold damping to stop very slow rotations
        let min_angular_velocity = 0.05;
        if self.angular_velocity.0.abs() < min_angular_velocity {
            self.angular_velocity.0 = 0.0;
        }
        if self.angular_velocity.1.abs() < min_angular_velocity {
            self.angular_velocity.1 = 0.0;
        }
        if self.angular_velocity.2.abs() < min_angular_velocity {
            self.angular_velocity.2 = 0.0;
        }

        // Apply additional damping when close to ground to simulate rolling resistance
        let ground_proximity = self.object.position.y - self.physics_constants.ground_level;
        if ground_proximity < 0.1 {
            // Increase damping when close to ground
            let ground_damping = 0.6 * dt;
            self.angular_velocity.0 *= 1.0 - ground_damping;
            self.angular_velocity.2 *= 1.0 - ground_damping;
        }
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
