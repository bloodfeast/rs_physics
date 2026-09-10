use std::fmt;
use std::error::Error;

/// Represents errors that can occur during physics calculations.
/// Comparable, so a test — or a caller routing on the failure — can name the variant it
/// expected rather than matching on a `Debug` string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PhysicsError {
    /// Indicates an invalid mass value\
    /// (e.g., negative or zero mass).
    InvalidMass,
    /// Indicates an invalid dimension value\
    /// (e.g., negative dimension).
    InvalidDimension,
    /// Indicates that the shape is not supported for the operation.
    UnsupportedShape,
    /// Indicates a division by zero error.
    DivisionByZero,
    /// Indicates an invalid velocity value.
    InvalidVelocity,
    /// Indicates an invalid time value\
    /// (e.g., unexpectedly negative time).
    InvalidTime,
    /// Indicates an invalid angle value\
    /// (e.g., angle outside the expected range).
    InvalidAngle,
    /// Indicates an invalid force value.
    InvalidForce,
    /// Indicates an invalid radius value\
    /// (e.g., negative radius).
    InvalidRadius,
    /// Indicates an invalid distance value\
    /// (e.g., negative distance).
    InvalidDistance,
    /// Indicates an invalid area value\
    /// (e.g., negative area).
    InvalidArea,
    /// Indicates an invalid volume value\
    /// (e.g., negative volume).
    InvalidVolume,
    /// Indicates an invalid coefficient value\
    /// (e.g., negative drag coefficient).
    InvalidCoefficient,
    /// Indicates that two objects are at the same position\
    /// (e.g., when calculating gravitational force).
    ObjectsAtSamePosition,
    /// Indicates an inertia tensor that does not describe a real mass distribution.\
    /// Either it is not positive definite (some axis has zero or negative moment, so
    /// the inverse does not exist) or its principal moments violate the triangle
    /// inequality `I₁ + I₂ ≥ I₃`, which no physical body can do.
    ///
    /// The two causes are distinguishable before the fact via
    /// [`crate::rotational_dynamics::InertiaTensor::is_positive_definite`] and
    /// [`crate::rotational_dynamics::InertiaTensor::satisfies_triangle_inequality`];
    /// they are one variant here because neither is recoverable at the call site.
    InvalidInertiaTensor,
    /// A general error for calculations that produce invalid results.\
    /// The associated string is intended to add context.
    CalculationError(String),
}

impl fmt::Display for PhysicsError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            PhysicsError::InvalidMass => write!(f, "Invalid mass value"),
            PhysicsError::InvalidDimension => write!(f, "Dimension must be greater than zero."),
            PhysicsError::UnsupportedShape => write!(f, "The specified shape is not supported for moment of inertia calculations."),
            PhysicsError::DivisionByZero => write!(f, "Division by zero"),
            PhysicsError::InvalidVelocity => write!(f, "Invalid velocity value"),
            PhysicsError::InvalidTime => write!(f, "Invalid time value"),
            PhysicsError::InvalidAngle => write!(f, "Invalid angle value"),
            PhysicsError::InvalidForce => write!(f, "Invalid force value"),
            PhysicsError::InvalidRadius => write!(f, "Invalid radius value"),
            PhysicsError::InvalidDistance => write!(f, "Invalid distance value"),
            PhysicsError::InvalidArea => write!(f, "Invalid area value"),
            PhysicsError::InvalidVolume => write!(f, "Invalid volume value"),
            PhysicsError::InvalidCoefficient => write!(f, "Invalid coefficient value"),
            PhysicsError::ObjectsAtSamePosition => write!(f, "Objects are at the same position"),
            PhysicsError::InvalidInertiaTensor => write!(f, "Inertia tensor does not describe a real mass distribution (not positive definite, or its principal moments violate the triangle inequality)"),
            PhysicsError::CalculationError(msg) => write!(f, "Calculation error: {}", msg),
        }
    }
}


impl Error for PhysicsError {}