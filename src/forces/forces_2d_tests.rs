use crate::forces::PhysicsSystem2D;
use crate::models::ObjectIn2D;
use crate::utils::DEFAULT_PHYSICS_CONSTANTS;

#[test]
fn test_add_and_retrieve_object() {
    let mut system = PhysicsSystem2D::new(DEFAULT_PHYSICS_CONSTANTS);
    let obj = ObjectIn2D::default();
    system.add_object(obj);
    assert!(system.get_object(0).is_some());
}

#[test]
fn test_clear_objects() {
    let mut system = PhysicsSystem2D::new(DEFAULT_PHYSICS_CONSTANTS);
    system.add_object(ObjectIn2D::default());
    system.add_object(ObjectIn2D::default());
    system.clear_objects();
    assert!(system.get_object(0).is_none());
}

#[test]
fn test_remove_object_valid_index() {
    let mut system = PhysicsSystem2D::new(DEFAULT_PHYSICS_CONSTANTS);
    system.add_object(ObjectIn2D::default());
    let removed = system.remove_object(0);
    assert!(removed.is_some());
    assert!(system.get_object(0).is_none());
}

#[test]
fn test_remove_object_invalid_index() {
    let mut system = PhysicsSystem2D::new(DEFAULT_PHYSICS_CONSTANTS);
    let removed = system.remove_object(999);
    assert!(removed.is_none());
}

#[test]
fn test_apply_gravity() {
    let mut system = PhysicsSystem2D::new(DEFAULT_PHYSICS_CONSTANTS);
    let mut obj = ObjectIn2D::default();
    obj.clear_forces();
    system.add_object(obj);
    system.apply_gravity();
    assert_eq!(system.get_object(0).unwrap().forces.len(), 1);
}