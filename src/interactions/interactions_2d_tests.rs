use crate::interactions::elastic_collision_2d;
use crate::models::ObjectIn2D;
use crate::utils::DEFAULT_PHYSICS_CONSTANTS;

#[test]
fn test_elastic_collision_2d_valid() {
    let constants = DEFAULT_PHYSICS_CONSTANTS;
    let mut obj1 = ObjectIn2D::new(1.0, 2.0, (1.0, 0.0), (0.0, 0.0));
    let mut obj2 = ObjectIn2D::new(1.0, 0.5, (-1.0, 0.0), (1.0, 0.0));

    let result = elastic_collision_2d(&constants, &mut obj1, &mut obj2, 0.0, 2.0, 0.45, 1.0);
    assert!(result.is_ok());
    assert!(obj1.velocity > 0.0);
    assert!(obj2.velocity > 0.0);
}

#[test]
fn test_elastic_collision_2d_negative_mass() {
    let constants = DEFAULT_PHYSICS_CONSTANTS;
    let mut obj1 = ObjectIn2D::new(-1.0, 1.0, (0.0, 1.0), (0.0, 0.0));
    let mut obj2 = ObjectIn2D::new(1.0, 1.0, (0.0, 1.0), (1.0, 0.0));

    let result = elastic_collision_2d(&constants, &mut obj1, &mut obj2, 0.5, 1.0, 0.0, 1.0);
    assert!(result.is_err());
}

#[test]
fn test_elastic_collision_2d_zero_duration() {
    let constants = DEFAULT_PHYSICS_CONSTANTS;
    let mut obj1 = ObjectIn2D::new(1.0, 1.0, (0.0, 1.0), (0.0, 0.0));
    let mut obj2 = ObjectIn2D::new(1.0, 1.0, (0.0, 1.0), (1.0, 0.0));

    let result = elastic_collision_2d(&constants, &mut obj1, &mut obj2, 0.5, 0.0, 0.0, 1.0);
    assert!(result.is_err());
}
