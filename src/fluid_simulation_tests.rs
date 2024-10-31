use crate::fluid_simulation::FluidGrid;

#[test]
fn test_fluid_grid_creation() {
    let grid = FluidGrid::new(10, 10, 0.1, 0.001, 0.016);
    assert!(grid.is_ok());

    let grid = grid.unwrap();
    assert_eq!(grid.get_width(), 10);
    assert_eq!(grid.get_height(), 10);
    assert_eq!(grid.get_diffusion(), 0.1);
    assert_eq!(grid.get_viscosity(), 0.001);
    assert_eq!(grid.get_dt(), 0.016);
}

#[test]
fn test_invalid_parameters() {
    assert!(FluidGrid::new(0, 10, 0.1, 0.001, 0.016).is_err());
    assert!(FluidGrid::new(10, 0, 0.1, 0.001, 0.016).is_err());
    assert!(FluidGrid::new(10, 10, -0.1, 0.001, 0.016).is_err());
    assert!(FluidGrid::new(10, 10, 0.1, -0.001, 0.016).is_err());
    assert!(FluidGrid::new(10, 10, 0.1, 0.001, 0.0).is_err());
}

#[test]
fn test_add_density() {
    let mut grid = FluidGrid::new(10, 10, 0.1, 0.001, 0.016).unwrap();
    assert!(grid.add_density(5, 5, 1.0).is_ok());
    assert_eq!(grid.get_density(5, 5).unwrap(), 1.0);
    assert!(grid.add_density(10, 5, 1.0).is_err());
}

#[test]
fn test_add_velocity() {
    let mut grid = FluidGrid::new(10, 10, 0.1, 0.001, 0.016).unwrap();
    assert!(grid.add_velocity(5, 5, 1.0, -1.0).is_ok());
    let (vx, vy) = grid.get_velocity(5, 5).unwrap();
    assert_eq!(vx, 1.0);
    assert_eq!(vy, -1.0);
    assert!(grid.add_velocity(10, 5, 1.0, 1.0).is_err());
}

#[test]
fn test_reset() {
    let mut grid = FluidGrid::new(10, 10, 0.1, 0.001, 0.016).unwrap();
    grid.add_density(5, 5, 1.0).unwrap();
    grid.add_velocity(5, 5, 1.0, -1.0).unwrap();
    grid.reset();
    assert_eq!(grid.get_density(5, 5).unwrap(), 0.0);
    let (vx, vy) = grid.get_velocity(5, 5).unwrap();
    assert_eq!(vx, 0.0);
    assert_eq!(vy, 0.0);
}