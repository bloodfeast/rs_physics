use crate::forces::Force;
use crate::models::{Axis2D, Direction2D, FromCoordinates, ObjectIn2D, ToCoordinates};
use crate::utils::PhysicsConstants;

impl ObjectIn2D {
    pub fn new(mass: f64, velocity: f64, direction: (f64, f64), position: (f64, f64)) -> Self {
        ObjectIn2D {
            mass,
            velocity,
            direction: Direction2D::from_coord(direction),
            position: Axis2D::from_coord(position),
            forces: Vec::new(),
        }
    }

    pub fn get_directional_velocities(&self) -> (f64, f64) {
        let (x, y) = self.direction.to_coord();
        let x_velocity = (x - y.abs()) * self.velocity;
        let y_velocity = (y - x.abs()) * self.velocity;
        (x_velocity, y_velocity)
    }

    pub fn add_force(&mut self, force: Force) {
        self.forces.push(force);
    }

    pub fn clear_forces(&mut self) {
        self.forces.clear();
    }
}

/// Simulates a 2D elastic collision between two objects.
///
/// # Arguments
///
/// * \`constants\` - Structure containing gravity and air density.
/// * \`obj1\`, \`obj2\` - Mutable references to the two colliding objects.
/// * \`angle\` - The angle of collision in radians.
/// * \`duration\` - The duration of the collision in seconds (must be positive).
/// * \`drag_coefficient\` - The coefficient of drag (must be non-negative).
/// * \`cross_sectional_area\` - The cross-sectional area of the objects (must be non-negative).
///
/// # Returns
///
/// \`Ok(())\` if the collision was successfully computed, or an error message on invalid parameters.
///
/// # Examples
///
/// ```
/// use rs_physics::interactions::elastic_collision_2d;
/// use rs_physics::models::ObjectIn2D;
/// let mut obj1 = ObjectIn2D::new(1.0, 2.0, (1.0, 0.0), (0.0, 0.0));
/// let mut obj2 = ObjectIn2D::new(1.0, -1.0, (-1.0, 0.0), (1.0, 0.0));
/// elastic_collision_2d(&rs_physics::utils::DEFAULT_PHYSICS_CONSTANTS, &mut obj1, &mut obj2, 0.0, 1.0, 0.45, 1.0).unwrap();
/// assert_eq!(obj1.velocity, 0.724375);
/// assert_eq!(obj2.velocity, 1.44875);
/// ```
///
/// # Notes
///
/// This function projects each object's velocity onto axes parallel and perpendicular to the collision angle,
/// applies a 1D elastic collision formula to the parallel components, then updates velocities considering gravity and drag.
/// Finally, it recombines velocities and updates direction and speed.
pub fn elastic_collision_2d(
    constants: &PhysicsConstants,
    obj1: &mut ObjectIn2D,
    obj2: &mut ObjectIn2D,
    angle: f64,
    duration: f64,
    drag_coefficient: f64,
    cross_sectional_area: f64
) -> Result<(), &'static str> {
    if obj1.mass <= 0.0 || obj2.mass <= 0.0 {
        return Err("Mass must be positive");
    }
    if duration <= 0.0 {
        return Err("Collision duration must be positive");
    }
    if drag_coefficient < 0.0 {
        return Err("Drag coefficient must be non-negative");
    }
    if cross_sectional_area < 0.0 {
        return Err("Cross-sectional area must be non-negative");
    }

    let (vx1, vy1) = obj1.get_directional_velocities();
    let (vx2, vy2) = obj2.get_directional_velocities();

    // Project velocities onto collision axis (angle) and perpendicular axis (angle + 90)
    let v1_along = vx1 * angle.cos() + vy1 * angle.sin();
    let v1_perp = -vx1 * angle.sin() + vy1 * angle.cos();
    let v2_along = vx2 * angle.cos() + vy2 * angle.sin();
    let v2_perp = -vx2 * angle.sin() + vy2 * angle.cos();

    // Calculate final velocities along collision axis using a 1D elastic collision formula
    let m1 = obj1.mass;
    let m2 = obj2.mass;
    let total_mass = m1 + m2;
    let mass_diff = m1 - m2;

    let v1_final_along = (mass_diff * v1_along + 2.0 * m2 * v2_along) / total_mass;
    let v2_final_along = (2.0 * m1 * v1_along - mass_diff * v2_along) / total_mass;

    // Approximate gravity effect (if needed)
    let gravity_effect = constants.gravity * duration * angle.sin();

    // Simplified air resistance
    let air_res_force = 0.5 * constants.air_density * drag_coefficient * cross_sectional_area * duration;
    let v1_drag = air_res_force * v1_final_along.abs() / m1;
    let v2_drag = air_res_force * v2_final_along.abs() / m2;

    // Update velocities along collision axis with gravity and drag
    let v1_along_updated = v1_final_along - gravity_effect - v1_drag * v1_final_along.signum();
    let v2_along_updated = v2_final_along - gravity_effect - v2_drag * v2_final_along.signum();

    // Recompose velocities
    let new_vx1 = v1_along_updated * angle.cos() - v1_perp * angle.sin();
    let new_vy1 = v1_along_updated * angle.sin() + v1_perp * angle.cos();
    let new_vx2 = v2_along_updated * angle.cos() - v2_perp * angle.sin();
    let new_vy2 = v2_along_updated * angle.sin() + v2_perp * angle.cos();

    // Convert back to speed and direction
    let speed1 = (new_vx1 * new_vx1 + new_vy1 * new_vy1).sqrt();
    let dir1 = (new_vx1 / speed1, new_vy1 / speed1);

    let speed2 = (new_vx2 * new_vx2 + new_vy2 * new_vy2).sqrt();
    let dir2 = (new_vx2 / speed2, new_vy2 / speed2);

    obj1.velocity = speed1;
    obj1.direction = Direction2D::from_coord(dir1);
    obj2.velocity = speed2;
    obj2.direction = Direction2D::from_coord(dir2);

    Ok(())
}
