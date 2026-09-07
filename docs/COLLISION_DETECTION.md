# Collision Detection

This document provides detailed documentation of the collision detection systems in rs_physics, including the GJK algorithm, EPA algorithm, and Continuous Collision Detection (CCD).

## Table of Contents

1. [Overview](#overview)
2. [Broad Phase](#broad-phase)
3. [GJK Algorithm](#gjk-algorithm)
4. [EPA Algorithm](#epa-algorithm)
5. [Continuous Collision Detection](#continuous-collision-detection)
6. [Shape-Specific Handling](#shape-specific-handling)
7. [Collision Response](#collision-response)

---

## Overview

rs_physics implements a multi-phase collision detection pipeline:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Broad Phase   │ -> │  GJK Detection  │ -> │  EPA Contact    │
│   AABB Culling  │    │  Intersection   │    │  Information    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                      │                      │
         │ ~60% eliminated      │ collision?           │ penetration
         │ early                │ yes/no               │ depth + normal
         ▼                      ▼                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Collision Response                            │
│              Impulse calculation and application                 │
└─────────────────────────────────────────────────────────────────┘
```

**Key Files:**
- `src/interactions/gjk_collision_3d.rs` - GJK + EPA implementation (~780 lines)
- `src/interactions/continuous_collision_detection.rs` - CCD system (~2000 lines)
- `src/interactions/shape_collisions_3d.rs` - Shape-specific handling (~1200 lines)

---

## Broad Phase

### AABB Culling

Before running the expensive GJK algorithm, objects are first tested with Axis-Aligned Bounding Box (AABB) checks.

```rust
// Conservative bounding radius calculation
fn get_bounding_radius(shape: &Shape3D) -> f64 {
    match shape {
        Shape3D::Sphere(r) => *r,
        Shape3D::Cuboid(w, h, d) => {
            // Diagonal of cuboid
            (w*w + h*h + d*d).sqrt() / 2.0
        }
        Shape3D::Cylinder(r, h) => {
            (r*r + (h/2.0).powi(2)).sqrt()
        }
        // ...
    }
}
```

**Benefits:**
- Eliminates ~60% of impossible collision pairs
- O(1) per pair comparison
- Simple axis-aligned box overlap test

### Sphere-Sphere Fast Path

For sphere-sphere collisions (common case ~30%), a direct distance check bypasses GJK entirely:

```rust
fn spheres_colliding(s1: &PhysicalObject3D, s2: &PhysicalObject3D) -> bool {
    let dx = s2.object.position.x - s1.object.position.x;
    let dy = s2.object.position.y - s1.object.position.y;
    let dz = s2.object.position.z - s1.object.position.z;

    let dist_sq = dx*dx + dy*dy + dz*dz;
    let r1 = s1.shape.get_radius();
    let r2 = s2.shape.get_radius();

    dist_sq <= (r1 + r2).powi(2)
}
```

---

## GJK Algorithm

### Overview

The Gilbert-Johnson-Keerthi (GJK) algorithm determines if two convex shapes intersect by checking if their **Minkowski difference** contains the origin.

**Key Insight:** Two shapes A and B intersect if and only if their Minkowski difference (A - B) contains the origin.

### Minkowski Difference

The Minkowski difference of two shapes is the set of all points obtained by subtracting every point in B from every point in A:

```
A - B = { a - b | a ∈ A, b ∈ B }
```

Instead of computing this explicitly, GJK uses **support functions**.

### Support Function

The support function returns the furthest point of a shape in a given direction:

```rust
pub fn get_support_point(
    shape: &Shape3D,
    direction: (f64, f64, f64),
    orientation: &Quaternion
) -> (f64, f64, f64) {
    match shape {
        Shape3D::Sphere(radius) => {
            // Furthest point is center + radius * normalized_direction
            let mag = vector_magnitude(direction);
            if mag < EPSILON {
                return (0.0, 0.0, 0.0);
            }
            let norm = (direction.0/mag, direction.1/mag, direction.2/mag);
            (norm.0 * radius, norm.1 * radius, norm.2 * radius)
        }

        Shape3D::Cuboid(w, h, d) => {
            // Transform direction to local space
            let local_dir = orientation.inverse().rotate_point(direction);

            // Support point is the corner in the direction
            let hw = w / 2.0;
            let hh = h / 2.0;
            let hd = d / 2.0;

            let local_support = (
                if local_dir.0 >= 0.0 { hw } else { -hw },
                if local_dir.1 >= 0.0 { hh } else { -hh },
                if local_dir.2 >= 0.0 { hd } else { -hd },
            );

            // Transform back to world space
            orientation.rotate_point(local_support)
        }

        // Similar for Cylinder, BeveledCuboid, Polyhedron...
    }
}
```

### GJK Algorithm Flow

```rust
pub fn gjk_collision_detection(
    obj_a: &PhysicalObject3D,
    obj_b: &PhysicalObject3D
) -> (bool, Simplex) {
    // 1. Initialize with arbitrary direction
    let mut direction = (1.0, 0.0, 0.0);

    // 2. Get first support point of Minkowski difference
    let support = get_minkowski_support(obj_a, obj_b, direction);

    // 3. Initialize simplex with first point
    let mut simplex = Simplex::new();
    simplex.push(support);

    // 4. New search direction toward origin
    direction = negate(support);

    // 5. Iterate until collision determined or max iterations
    for _ in 0..GJK_MAX_ITERATIONS {
        // Get new support point
        let new_point = get_minkowski_support(obj_a, obj_b, direction);

        // If new point didn't pass origin, no collision
        if dot_product(new_point, direction) < 0.0 {
            return (false, simplex);
        }

        simplex.push(new_point);

        // Check if simplex contains origin
        if handle_simplex(&mut simplex, &mut direction) {
            return (true, simplex);
        }
    }

    (false, simplex)
}
```

### Simplex Cases

The simplex evolves from a point to a line to a triangle to a tetrahedron:

#### Line Case (2 points)
```rust
fn handle_line(simplex: &mut Simplex, direction: &mut (f64, f64, f64)) -> bool {
    let a = simplex.points[1];  // Most recently added
    let b = simplex.points[0];

    let ab = subtract(b, a);
    let ao = negate(a);  // Vector from A toward origin

    if dot_product(ab, ao) > 0.0 {
        // Origin is in the direction of B from A
        // New direction perpendicular to AB toward origin
        *direction = triple_product(ab, ao, ab);
    } else {
        // Origin is behind A, start over with just A
        simplex.points = vec![a];
        *direction = ao;
    }

    false  // Line can't contain origin
}
```

#### Triangle Case (3 points)
```rust
fn handle_triangle(simplex: &mut Simplex, direction: &mut (f64, f64, f64)) -> bool {
    let a = simplex.points[2];
    let b = simplex.points[1];
    let c = simplex.points[0];

    let ab = subtract(b, a);
    let ac = subtract(c, a);
    let ao = negate(a);
    let abc = cross_product(ab, ac);  // Triangle normal

    // Check which region the origin is in
    if dot_product(cross_product(abc, ac), ao) > 0.0 {
        if dot_product(ac, ao) > 0.0 {
            // Region AC
            simplex.points = vec![c, a];
            *direction = triple_product(ac, ao, ac);
        } else {
            // Handle AB region
            // ...
        }
    } else if dot_product(cross_product(ab, abc), ao) > 0.0 {
        // Region AB
        simplex.points = vec![b, a];
        *direction = triple_product(ab, ao, ab);
    } else {
        // Origin is above or below triangle
        if dot_product(abc, ao) > 0.0 {
            *direction = abc;
        } else {
            simplex.points = vec![b, c, a];
            *direction = negate(abc);
        }
    }

    false
}
```

#### Tetrahedron Case (4 points)
```rust
fn handle_tetrahedron(simplex: &mut Simplex, direction: &mut (f64, f64, f64)) -> bool {
    let a = simplex.points[3];
    let b = simplex.points[2];
    let c = simplex.points[1];
    let d = simplex.points[0];

    let ab = subtract(b, a);
    let ac = subtract(c, a);
    let ad = subtract(d, a);
    let ao = negate(a);

    // Check each face to see if origin is outside
    let abc = cross_product(ab, ac);
    let acd = cross_product(ac, ad);
    let adb = cross_product(ad, ab);

    if dot_product(abc, ao) > 0.0 {
        // Origin outside ABC face, reduce to triangle
        simplex.points = vec![c, b, a];
        return handle_triangle(simplex, direction);
    }

    // Similar for ACD and ADB faces...

    // Origin is inside tetrahedron!
    true
}
```

### Constants

```rust
const EPSILON: f64 = 1e-12;           // Numerical tolerance
const GJK_MAX_ITERATIONS: usize = 32;  // Iteration limit
```

---

## EPA Algorithm

### Overview

When GJK detects a collision, the Expanding Polytope Algorithm (EPA) computes:
- **Penetration depth** - How far the objects overlap
- **Contact normal** - Direction to separate the objects
- **Contact point** - Where the collision occurs

### Algorithm Flow

```rust
pub fn epa_contact_points(
    obj_a: &PhysicalObject3D,
    obj_b: &PhysicalObject3D,
    simplex: &Simplex
) -> Option<(f64, (f64, f64, f64), (f64, f64, f64))> {
    // 1. Initialize polytope from GJK simplex (tetrahedron)
    let mut polytope = simplex.points.clone();
    let mut faces = initial_tetrahedron_faces();

    // 2. Iteratively expand polytope toward Minkowski boundary
    for _ in 0..EPA_MAX_ITERATIONS {
        // Find closest face to origin
        let (closest_face_idx, normal, distance) = find_closest_face(&polytope, &faces);

        // Get support point in direction of normal
        let support = get_minkowski_support(obj_a, obj_b, normal);
        let support_dist = dot_product(support, normal);

        // Check for convergence
        if (support_dist - distance).abs() < EPA_TOLERANCE {
            // Compute contact point using barycentric coordinates
            let contact = compute_contact_point(&polytope, &faces[closest_face_idx], normal);
            return Some((distance, normal, contact));
        }

        // Expand polytope: remove visible faces, add new faces
        expand_polytope(&mut polytope, &mut faces, support);
    }

    None  // Failed to converge
}
```

### Polytope Expansion

When a new support point is found:

1. **Find visible faces** - Faces whose normal points toward the new point
2. **Remove visible faces** - These are inside the Minkowski difference
3. **Add new faces** - Connect edges of the "horizon" to the new point

```rust
fn expand_polytope(
    polytope: &mut Vec<(f64,f64,f64)>,
    faces: &mut Vec<[usize; 3]>,
    new_point: (f64, f64, f64)
) {
    // Add new point to polytope
    let new_idx = polytope.len();
    polytope.push(new_point);

    // Find and remove faces visible from new point
    let mut edges_to_fix: Vec<(usize, usize)> = Vec::new();

    faces.retain(|face| {
        let normal = compute_face_normal(polytope, face);
        let face_point = polytope[face[0]];
        let to_new = subtract(new_point, face_point);

        if dot_product(normal, to_new) > 0.0 {
            // Face is visible, collect its edges
            edges_to_fix.push((face[0], face[1]));
            edges_to_fix.push((face[1], face[2]));
            edges_to_fix.push((face[2], face[0]));
            false  // Remove this face
        } else {
            true   // Keep this face
        }
    });

    // Find unique edges (horizon) and create new faces
    let horizon = find_horizon_edges(edges_to_fix);
    for (i, j) in horizon {
        faces.push([i, j, new_idx]);
    }
}
```

### Contact Point Calculation

Using barycentric coordinates on the closest face:

```rust
fn compute_contact_point(
    polytope: &[(f64,f64,f64)],
    face: &[usize; 3],
    normal: (f64, f64, f64)
) -> (f64, f64, f64) {
    let a = polytope[face[0]];
    let b = polytope[face[1]];
    let c = polytope[face[2]];

    // Project origin onto face plane
    let origin_proj = project_point_to_plane((0.0, 0.0, 0.0), a, normal);

    // Compute barycentric coordinates
    let (u, v, w) = barycentric_coords(origin_proj, a, b, c);

    // Interpolate contact point
    (
        u * a.0 + v * b.0 + w * c.0,
        u * a.1 + v * b.1 + w * c.1,
        u * a.2 + v * b.2 + w * c.2,
    )
}
```

### Constants

```rust
const EPA_MAX_ITERATIONS: usize = 64;
const EPA_TOLERANCE: f64 = 1e-6;
```

---

## Continuous Collision Detection

### Overview

CCD prevents **tunneling** - fast-moving objects passing through each other between frames. It computes the **Time of Impact (ToI)**.

**Location:** `src/interactions/continuous_collision_detection.rs`

### Time of Impact Calculation

```rust
pub fn check_continuous_collision(
    obj_a: &PhysicalObject3D,
    obj_b: &PhysicalObject3D,
    dt: f64
) -> CcdCollisionResult {
    // Check if relative motion is significant enough
    if !is_relative_motion_significant(obj_a, obj_b, dt) {
        return CcdCollisionResult::no_collision();
    }

    // Specialized fast path for spheres
    if matches!(obj_a.shape, Shape3D::Sphere(_))
       && matches!(obj_b.shape, Shape3D::Sphere(_)) {
        return sphere_sphere_ccd(obj_a, obj_b, dt);
    }

    // General case: binary search on time
    binary_search_toi(obj_a, obj_b, dt)
}
```

### Binary Search for ToI

```rust
fn binary_search_toi(
    obj_a: &PhysicalObject3D,
    obj_b: &PhysicalObject3D,
    dt: f64
) -> CcdCollisionResult {
    let mut t_low = 0.0;
    let mut t_high = dt;

    // Check if collision at end of timestep
    let end_a = interpolate_position(obj_a, dt);
    let end_b = interpolate_position(obj_b, dt);

    if !gjk_collision_detection(&end_a, &end_b).0 {
        return CcdCollisionResult::no_collision();
    }

    // Binary search for exact time of impact
    for _ in 0..32 {
        let t_mid = (t_low + t_high) / 2.0;

        let mid_a = interpolate_position(obj_a, t_mid);
        let mid_b = interpolate_position(obj_b, t_mid);

        if gjk_collision_detection(&mid_a, &mid_b).0 {
            t_high = t_mid;
        } else {
            t_low = t_mid;
        }

        if (t_high - t_low) < 1e-6 {
            break;
        }
    }

    // Get collision info at ToI
    let toi = (t_low + t_high) / 2.0;
    let impact_a = interpolate_position(obj_a, toi);
    let impact_b = interpolate_position(obj_b, toi);

    let (_, simplex) = gjk_collision_detection(&impact_a, &impact_b);
    let contact_info = epa_contact_points(&impact_a, &impact_b, &simplex);

    CcdCollisionResult {
        will_collide: true,
        time_of_impact: toi,
        normal: contact_info.map(|c| c.1),
        contact_points: contact_info.map(|c| vec![c.2]),
    }
}
```

### Sphere-Sphere CCD (Analytical)

For sphere-sphere, we can solve analytically:

```rust
fn sphere_sphere_ccd(
    obj_a: &PhysicalObject3D,
    obj_b: &PhysicalObject3D,
    dt: f64
) -> CcdCollisionResult {
    let r_a = obj_a.shape.get_radius();
    let r_b = obj_b.shape.get_radius();
    let r_sum = r_a + r_b;

    // Relative position and velocity
    let p = subtract(obj_b.object.position, obj_a.object.position);
    let v = subtract(obj_b.object.velocity, obj_a.object.velocity);

    // Solve: |p + t*v|^2 = r_sum^2
    // This is quadratic: a*t^2 + b*t + c = 0
    let a = dot_product(v, v);
    let b = 2.0 * dot_product(p, v);
    let c = dot_product(p, p) - r_sum * r_sum;

    let discriminant = b*b - 4.0*a*c;

    if discriminant < 0.0 {
        return CcdCollisionResult::no_collision();
    }

    let t1 = (-b - discriminant.sqrt()) / (2.0 * a);
    let t2 = (-b + discriminant.sqrt()) / (2.0 * a);

    // Find first positive t within [0, dt]
    let toi = if t1 >= 0.0 && t1 <= dt { t1 }
              else if t2 >= 0.0 && t2 <= dt { t2 }
              else { return CcdCollisionResult::no_collision(); };

    // Compute contact info
    let impact_p = add(p, scale(v, toi));
    let normal = normalize(impact_p);

    CcdCollisionResult {
        will_collide: true,
        time_of_impact: toi,
        normal: Some(normal),
        contact_points: Some(vec![/* ... */]),
    }
}
```

### CCD Result Structure

```rust
pub struct CcdCollisionResult {
    pub will_collide: bool,
    pub time_of_impact: f64,
    pub normal: Option<(f64, f64, f64)>,
    pub contact_points: Option<Vec<(f64, f64, f64)>>,
}
```

---

## Shape-Specific Handling

**Location:** `src/interactions/shape_collisions_3d.rs`

### Impact Point Calculation

Different shape pairs have specialized contact point computation:

```rust
pub fn calculate_impact_point(
    obj_a: &PhysicalObject3D,
    obj_b: &PhysicalObject3D,
    collision_normal: (f64, f64, f64)
) -> (f64, f64, f64) {
    match (&obj_a.shape, &obj_b.shape) {
        (Shape3D::Sphere(r_a), Shape3D::Sphere(_)) => {
            // Contact point on sphere A surface
            let dir = normalize(collision_normal);
            add(obj_a.object.position, scale(dir, *r_a))
        }

        (Shape3D::Sphere(r), Shape3D::Cuboid(..)) |
        (Shape3D::Cuboid(..), Shape3D::Sphere(r)) => {
            // Closest point on cuboid to sphere center
            // ...
        }

        (Shape3D::Cuboid(..), Shape3D::Cuboid(..)) => {
            // Find closest corners
            // ...
        }

        _ => {
            // Generic: midpoint along collision normal
            let mid = midpoint(obj_a.object.position, obj_b.object.position);
            mid
        }
    }
}
```

### Point Velocity Calculation

For rotating objects, the velocity at a contact point includes angular contribution:

```rust
pub fn calculate_point_velocity(
    obj: &PhysicalObject3D,
    contact_point: (f64, f64, f64)
) -> (f64, f64, f64) {
    // r = contact_point - center_of_mass
    let r = subtract(contact_point, obj.object.position);

    // v_point = v_linear + omega × r
    let angular_contribution = cross_product(obj.angular_velocity, r);

    add(obj.object.velocity, angular_contribution)
}
```

---

## Collision Response

### Impulse Calculation

```rust
pub fn calculate_collision_impulse(
    obj_a: &PhysicalObject3D,
    obj_b: &PhysicalObject3D,
    contact_point: (f64, f64, f64),
    normal: (f64, f64, f64),
    restitution: f64
) -> f64 {
    // Relative velocity at contact point
    let v_a = calculate_point_velocity(obj_a, contact_point);
    let v_b = calculate_point_velocity(obj_b, contact_point);
    let v_rel = dot_product(subtract(v_a, v_b), normal);

    // If separating, no impulse needed
    if v_rel > 0.0 {
        return 0.0;
    }

    // Impulse magnitude: j = -(1 + e) * v_rel / (1/m_a + 1/m_b)
    let inv_mass_sum = 1.0 / obj_a.object.mass + 1.0 / obj_b.object.mass;

    -(1.0 + restitution) * v_rel / inv_mass_sum
}
```

### Applying Impulse

```rust
pub fn apply_linear_impulse(
    obj: &mut PhysicalObject3D,
    impulse: (f64, f64, f64)
) {
    // delta_v = impulse / mass
    let dv = scale(impulse, 1.0 / obj.object.mass);
    obj.object.velocity = add(obj.object.velocity, dv);
}

pub fn apply_angular_impulse(
    obj: &mut PhysicalObject3D,
    impulse: (f64, f64, f64),
    contact_point: (f64, f64, f64)
) {
    // torque = r × impulse
    let r = subtract(contact_point, obj.object.position);
    let torque = cross_product(r, impulse);

    // delta_omega = I^(-1) * torque
    let inv_inertia = invert_3x3(obj.inertia_tensor);
    let delta_omega = matrix_vector_multiply(inv_inertia, torque);

    obj.angular_velocity = add(obj.angular_velocity, delta_omega);
}
```

### Complete Collision Response

```rust
pub fn apply_collision_response(
    obj_a: &mut PhysicalObject3D,
    obj_b: &mut PhysicalObject3D,
    contact: &ContactInfo,
    restitution: f64
) {
    let impulse_mag = calculate_collision_impulse(
        obj_a, obj_b,
        contact.point, contact.normal,
        restitution
    );

    let impulse = scale(contact.normal, impulse_mag);
    let neg_impulse = negate(impulse);

    // Apply to object A (receives positive impulse)
    apply_linear_impulse(obj_a, impulse);
    apply_angular_impulse(obj_a, impulse, contact.point);

    // Apply to object B (receives negative impulse)
    apply_linear_impulse(obj_b, neg_impulse);
    apply_angular_impulse(obj_b, neg_impulse, contact.point);

    // Separate objects to prevent penetration
    separate_objects(obj_a, obj_b, contact.normal, contact.penetration);
}
```

---

## Performance Summary

| Phase | Complexity | Typical Cost |
|-------|------------|--------------|
| AABB Broad Phase | O(1) | ~5 ns |
| Sphere-Sphere Check | O(1) | ~10 ns |
| GJK (per pair) | O(1) amortized | ~100-500 ns |
| EPA (per collision) | O(n) | ~500-2000 ns |
| CCD Binary Search | O(log(1/ε)) | ~1-5 μs |

**Recommendations:**
- Use broad phase when checking many pairs
- Prefer spheres when possible (fast path)
- Use CCD only for fast-moving objects
- Tune iteration limits based on accuracy needs

---

*See [API_REFERENCE.md](./API_REFERENCE.md) for function signatures and usage examples.*
