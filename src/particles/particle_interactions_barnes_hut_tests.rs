use crate::particles::{ApproxNode, BarnesHutNode, build_tree, collect_approx_nodes, compute_force_scalar, compute_force_simd_avx, ParticleData, Quad};

#[test]
fn test_quad_contains() {
    let quad = Quad { cx: 0.0, cy: 0.0, half_size: 1.0 };
    assert!(quad.contains(0.0, 0.0));
    assert!(!quad.contains(1.0, 0.0)); // upper bound is half-open
}

#[test]
fn test_quad_subdivide() {
    let quad = Quad { cx: 0.0, cy: 0.0, half_size: 1.0 };
    let (nw, _ne, _sw, _se) = quad.subdivide();
    // Check that subdivision produces correct centers.
    assert!((nw.cx + 0.5).abs() < 1e-6);
    assert!((nw.cy - 0.5).abs() < 1e-6);
}

#[test]
fn test_build_tree() {
    let particles = [
        ParticleData { x: -0.5, y: -0.5, mass: 1.0 },
        ParticleData { x: 0.5, y: 0.5, mass: 1.0 },
        ParticleData { x: -0.5, y: 0.5, mass: 1.0 },
        ParticleData { x: 0.5, y: -0.5, mass: 1.0 },
    ];
    let quad = Quad { cx: 0.0, cy: 0.0, half_size: 1.0 };
    let tree = build_tree(&particles, quad);
    if let BarnesHutNode::Internal { mass, .. } = tree {
        assert!(mass >= 4.0);
    } else {
        panic!("Expected an Internal node after inserting 4 particles");
    }
}

#[test]
fn test_collect_approx_nodes() {
    let particles = [
        ParticleData { x: 0.1, y: 0.1, mass: 1.0 },
        ParticleData { x: 0.2, y: 0.1, mass: 1.0 },
    ];
    let quad = Quad { cx: 0.0, cy: 0.0, half_size: 1.0 };
    let tree = build_tree(&particles, quad);
    let target = ParticleData { x: 0.0, y: 0.0, mass: 1.0 };
    let theta = 0.5;
    let mut worklist = Vec::new();
    collect_approx_nodes(&tree, target, theta, &mut worklist);
    assert!(!worklist.is_empty(), "Worklist should not be empty");
}

#[test]
fn test_compute_force_scalar() {
    let worklist = vec![
        ApproxNode { mass: 1.0, com_x: 0.5, com_y: 0.0 },
        ApproxNode { mass: 1.0, com_x: 0.6, com_y: 0.0 },
    ];
    let p = ParticleData { x: 0.0, y: 0.0, mass: 1.0 };
    let g = 6.67430e-11;
    let (fx, _fy) = compute_force_scalar(p, &worklist, g);
    assert!(fx.abs() > 0.0, "Scalar force x should be nonzero");
}

#[test]
fn test_compute_net_force() {
    // Use an asymmetric configuration to avoid cancellation.
    let particles = [
        ParticleData { x: 0.1, y: 0.0, mass: 1.0 },
        ParticleData { x: 0.2, y: 0.0, mass: 1.0 },
    ];
    let quad = Quad { cx: 0.0, cy: 0.0, half_size: 1.0 };
    let tree = build_tree(&particles, quad);
    let target = ParticleData { x: 0.0, y: 0.0, mass: 1.0 };
    let theta = 0.5;
    let g = 6.67430e-11;
    let (fx, _fy) = tree.compute_force(target, theta, g);
    // With both particles on the positive x-axis, the net force in x should be positive.
    assert!(fx > 0.0, "Expected positive force in x-direction");
}

#[test]
fn test_compute_force_simd_vs_scalar() {
    // Create a worklist with 8 nodes.
    let worklist: Vec<ApproxNode> = vec![
        ApproxNode { mass: 1.0, com_x: 0.5, com_y: 0.1 },
        ApproxNode { mass: 1.0, com_x: 0.6, com_y: 0.2 },
        ApproxNode { mass: 1.0, com_x: 0.7, com_y: 0.1 },
        ApproxNode { mass: 1.0, com_x: 0.8, com_y: 0.2 },
        ApproxNode { mass: 1.0, com_x: 0.9, com_y: 0.1 },
        ApproxNode { mass: 1.0, com_x: 1.0, com_y: 0.2 },
        ApproxNode { mass: 1.0, com_x: 1.1, com_y: 0.1 },
        ApproxNode { mass: 1.0, com_x: 1.2, com_y: 0.2 },
    ];
    let p = ParticleData { x: 0.0, y: 0.0, mass: 1.0 };
    let g = 6.67430e-11;
    let scalar_result = compute_force_scalar(p, &worklist, g);
    let simd_result = if std::is_x86_feature_detected!("avx") {
        unsafe { compute_force_simd_avx(p, &worklist, g) }
    } else {
        scalar_result
    };
    // The results should be nearly equal.
    assert!((scalar_result.0 - simd_result.0).abs() < 1e-10);
    assert!((scalar_result.1 - simd_result.1).abs() < 1e-10);
}