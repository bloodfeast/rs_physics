// src/apis/easy_physics.rs

use crate::utils::PhysicsConstants;
use crate::interactions::{elastic_collision, gravitational_force, apply_force};
use rayon::prelude::*;
use crate::models::Object;

/// A simplified interface for physics simulations.
///
/// This struct provides an easy-to-use API for common physics calculations and simulations.
/// It encapsulates the complexity of the underlying physics engine and provides intuitive
/// methods for creating objects, simulating collisions, and performing various calculations.
pub struct EasyPhysics {
    constants: PhysicsConstants,
}

impl EasyPhysics {

    /// Creates a new `EasyPhysics` instance with default physical constants.
    ///
    /// This constructor initializes an `EasyPhysics` object using the default values
    /// for physical constants such as gravity, air density, speed of sound, and
    /// atmospheric pressure.
    ///
    /// # Returns
    ///
    /// * `Self` - A new instance of `EasyPhysics` with default constants
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::apis::easy_physics::EasyPhysics;
    ///
    /// let physics = EasyPhysics::new();
    /// ```
    ///
    /// # Notes
    ///
    /// The default constants are:
    /// - Gravity: 9.80665 m/s²
    /// - Air density: 1.225 kg/m³
    /// - Speed of sound: 343.0 m/s
    /// - Atmospheric pressure: 101,325 Pa
    pub fn new() -> Self {
        Self {
            constants: PhysicsConstants::default(),
        }
    }

    /// Creates a new `EasyPhysics` instance with custom physical constants.
    ///
    /// This constructor allows you to initialize an `EasyPhysics` object with
    /// specific values for physical constants, providing more control over
    /// the simulation environment.
    ///
    /// # Arguments
    ///
    /// * `gravity` - The gravitational acceleration in m/s²
    /// * `air_density` - The air density in kg/m³
    /// * `speed_of_sound` - The speed of sound in m/s
    /// * `atmospheric_pressure` - The atmospheric pressure in Pa
    ///
    /// # Returns
    ///
    /// * `Self` - A new instance of `EasyPhysics` with the specified constants
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::apis::easy_physics::EasyPhysics;
    ///
    /// // Create a physics environment for Mars
    /// let mars_physics = EasyPhysics::with_custom_constants(
    ///     3.711,  // Mars gravity in m/s²
    ///     0.020,  // Approximate Mars atmosphere density in kg/m³
    ///     244.0,  // Approximate speed of sound on Mars in m/s
    ///     600.0,   // Approximate atmospheric pressure on Mars in Pa
    ///     0.0,  // Ground level on Mars in meters
    /// );
    /// ```
    ///
    /// # Notes
    ///
    /// This method allows you to simulate physics in different environments,
    /// such as other planets or hypothetical scenarios.
    pub fn with_custom_constants(gravity: f64, air_density: f64, speed_of_sound: f64, atmospheric_pressure: f64, ground_level: f64) -> Self {
        Self {
            constants: PhysicsConstants::new(
                Some(gravity),
                Some(air_density),
                Some(speed_of_sound),
                Some(atmospheric_pressure),
                Some(ground_level),
            ),
        }
    }

    /// Creates a new `Object` with the given mass, velocity, and position.
    ///
    /// This method allows you to create physical objects that can be used in
    /// various simulations and calculations provided by the `EasyPhysics` struct.
    ///
    /// # Arguments
    ///
    /// * `mass` - The mass of the object in kilograms (kg)
    /// * `velocity` - The initial velocity of the object in meters per second (m/s)
    /// * `position` - The initial position of the object in meters (m)
    ///
    /// # Returns
    ///
    /// * `Ok(Object)` - A new `Object` instance if the creation is successful
    /// * `Err(&'static str)` - An error message if the mass is not positive
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::apis::easy_physics::EasyPhysics;
    ///
    /// let physics = EasyPhysics::new();
    ///
    /// // Create a 5 kg object moving at 2 m/s, starting at position 0 m
    /// let obj = physics.create_object(5.0, 2.0, 0.0).unwrap();
    ///
    /// // Attempting to create an object with negative mass will result in an error
    /// let invalid_obj = physics.create_object(-1.0, 0.0, 0.0);
    /// assert!(invalid_obj.is_err());
    /// ```
    ///
    /// # Notes
    ///
    /// - The mass must be positive. Attempting to create an object with zero or
    ///   negative mass will result in an error.
    /// - The velocity can be positive (moving in the positive direction),
    ///   negative (moving in the negative direction), or zero (stationary).
    /// - The position can be any real number, representing the object's
    ///   location on a one-dimensional axis.
    pub fn create_object(&self, mass: f64, velocity: f64, position: f64) -> Result<Object, &'static str> {
        Object::new(mass, velocity, position)
    }

    /// Simulates an elastic collision between two objects.
    ///
    /// # Arguments
    ///
    /// * `obj1` - A mutable reference to the first object
    /// * `obj2` - A mutable reference to the second object
    /// * `angle` - The angle of collision in radians
    /// * `duration` - The duration of the collision in seconds
    /// * `drag_coefficient` - The drag coefficient for air resistance
    /// * `cross_sectional_area` - The cross-sectional area of the objects for air resistance calculation
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::apis::easy_physics::EasyPhysics;
    ///
    /// let physics = EasyPhysics::new();
    /// let mut obj1 = physics.create_object(1.0, 5.0, 0.0).unwrap();
    /// let mut obj2 = physics.create_object(1.0, -5.0, 10.0).unwrap();
    /// physics.simulate_collision(&mut obj1, &mut obj2, 0.0, 0.1, 0.47, 1.0).unwrap();
    /// ```
    pub fn simulate_collision(
        &self,
        obj1: &mut Object,
        obj2: &mut Object,
        angle: f64,
        duration: f64,
        drag_coefficient: f64,
        cross_sectional_area: f64
    ) -> Result<(), &'static str> {
        elastic_collision(&self.constants, obj1, obj2, angle, duration, drag_coefficient, cross_sectional_area)
    }

    pub fn simulate_multiple_collisions(
        constants: &PhysicsConstants,
        object_pairs: &mut [(Object, Object)],
        angle: f64,
        duration: f64,
        drag_coefficient: f64,
        cross_sectional_area: f64
    ) {
        object_pairs.par_iter_mut().for_each(|(obj1, obj2)| {
            let _ = elastic_collision(constants, obj1, obj2, angle, duration, drag_coefficient, cross_sectional_area);
        });
    }

    /// Calculates the gravitational force between two objects.
    ///
    /// This method uses the universal law of gravitation to compute the force of attraction
    /// between two objects based on their masses and the distance between them.
    ///
    /// # Arguments
    ///
    /// * `obj1` - A reference to the first object
    /// * `obj2` - A reference to the second object
    ///
    /// # Returns
    ///
    /// * `Ok(f64)` - The gravitational force in Newtons (N) if the calculation is successful
    /// * `Err(&'static str)` - An error message if the objects are at the same position
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::apis::easy_physics::EasyPhysics;
    ///
    /// let physics = EasyPhysics::new();
    /// let obj1 = physics.create_object(5.97e24, 0.0, 0.0).unwrap(); // Earth
    /// let obj2 = physics.create_object(7.34e22, 0.0, 3.84e8).unwrap(); // Moon
    /// let force = physics.calculate_gravity_force(&obj1, &obj2).unwrap();
    /// println!("Gravitational force between Earth and Moon: {} N", force);
    /// ```
    ///
    /// # Notes
    ///
    /// The gravitational force is always attractive and acts along the line joining the centers of the two objects.
    /// The force is calculated using the formula: F = G * (m1 * m2) / r^2, where G is the gravitational constant,
    /// m1 and m2 are the masses of the objects, and r is the distance between their centers.
    pub fn calculate_gravity_force(&self, obj1: &Object, obj2: &Object) -> Result<f64, &'static str> {
        gravitational_force(&self.constants, obj1, obj2)
    }

    /// Applies a force to an object for a given time and updates its velocity and position.
    ///
    /// This method simulates the effect of a constant force applied to an object over a specified duration.
    /// It updates the object's velocity and position based on the applied force and the laws of motion.
    ///
    /// # Arguments
    ///
    /// * `obj` - A mutable reference to the object to which the force is applied
    /// * `force` - The magnitude of the force in Newtons (N)
    /// * `time` - The duration for which the force is applied in seconds (s)
    ///
    /// # Returns
    ///
    /// * `Ok(())` - If the force was successfully applied and the object's state updated
    /// * `Err(&'static str)` - An error message if the time is negative
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::apis::easy_physics::EasyPhysics;
    ///
    /// let physics = EasyPhysics::new();
    /// let mut obj = physics.create_object(1.0, 0.0, 0.0).unwrap();
    /// physics.apply_force_to_object(&mut obj, 10.0, 2.0).unwrap();
    /// println!("New velocity: {} m/s", obj.velocity);
    /// println!("New position: {} m", obj.position);
    /// ```
    ///
    /// # Notes
    ///
    /// This method uses the equation F = ma to calculate the acceleration, and then updates
    /// the velocity and position using the equations of motion for constant acceleration.
    /// It assumes that the force is constant over the given time interval.
    pub fn apply_force_to_object(&self, obj: &mut Object, force: f64, time: f64) -> Result<(), &'static str> {
        apply_force(&self.constants, obj, force, time)
    }

    /// Calculates the kinetic energy of an object.
    ///
    /// This method computes the kinetic energy of an object based on its mass and velocity.
    ///
    /// # Arguments
    ///
    /// * `obj` - A reference to the object
    ///
    /// # Returns
    ///
    /// * `f64` - The kinetic energy of the object in Joules (J)
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::apis::easy_physics::EasyPhysics;
    ///
    /// let physics = EasyPhysics::new();
    /// let obj = physics.create_object(2.0, 3.0, 0.0).unwrap();
    /// let energy = physics.calculate_kinetic_energy(&obj);
    /// println!("Kinetic energy: {} J", energy);
    /// ```
    ///
    /// # Notes
    ///
    /// The kinetic energy is calculated using the formula: KE = (1/2) * m * v^2,
    /// where m is the mass of the object and v is its velocity.
    pub fn calculate_kinetic_energy(&self, obj: &Object) -> f64 {
        0.5 * obj.mass * obj.velocity * obj.velocity
    }

    /// Calculates the momentum of an object.
    ///
    /// This method computes the linear momentum of an object based on its mass and velocity.
    ///
    /// # Arguments
    ///
    /// * `obj` - A reference to the object
    ///
    /// # Returns
    ///
    /// * `f64` - The momentum of the object in kilogram-meters per second (kg⋅m/s)
    ///
    /// # Examples
    ///
    /// ```
    /// use rs_physics::apis::easy_physics::EasyPhysics;
    ///
    /// let physics = EasyPhysics::new();
    /// let obj = physics.create_object(2.0, 3.0, 0.0).unwrap();
    /// let momentum = physics.calculate_momentum(&obj);
    /// println!("Momentum: {} kg⋅m/s", momentum);
    /// ```
    ///
    /// # Notes
    ///
    /// The momentum is calculated using the formula: p = m * v,
    /// where m is the mass of the object and v is its velocity.
    /// Momentum is a vector quantity, but this method returns its magnitude.
    pub fn calculate_momentum(&self, obj: &Object) -> f64 {
        obj.mass * obj.velocity
    }
}
