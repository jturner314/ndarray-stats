//! Custom errors returned from our methods and functions.
use std::error::Error;
use std::fmt;

/// An error that indicates that the input array was empty.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EmptyInput;

impl fmt::Display for EmptyInput {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Empty input.")
    }
}

impl Error for EmptyInput {}

/// An error computing a minimum/maximum value.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MinMaxError {
    /// The input was empty.
    EmptyInput,
    /// The ordering between a tested pair of values was undefined.
    UndefinedOrder,
}

impl fmt::Display for MinMaxError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            MinMaxError::EmptyInput => write!(f, "Empty input."),
            MinMaxError::UndefinedOrder => {
                write!(f, "Undefined ordering between a tested pair of values.")
            }
        }
    }
}

impl Error for MinMaxError {}

impl From<EmptyInput> for MinMaxError {
    fn from(_: EmptyInput) -> MinMaxError {
        MinMaxError::EmptyInput
    }
}

/// An error used by methods and functions that take two arrays as argument and
/// expect them to have exactly the same shape
/// (e.g. `ShapeMismatch` is raised when `a.shape() == b.shape()` evaluates to `False`).
#[derive(Clone, Debug)]
pub struct ShapeMismatch {
    pub first_shape: Vec<usize>,
    pub second_shape: Vec<usize>,
}

impl fmt::Display for ShapeMismatch {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Array shapes do not match: {:?} and {:?}.",
            self.first_shape, self.second_shape
        )
    }
}

impl Error for ShapeMismatch {}

/// An error for methods that take multiple non-empty array inputs.
#[derive(Clone, Debug)]
pub enum MultiInputError {
    /// One or more of the arrays were empty.
    EmptyInput,
    /// The arrays did not have the same shape.
    ShapeMismatch(ShapeMismatch),
}

impl MultiInputError {
    /// Returns whether `self` is the `EmptyInput` variant.
    pub fn is_empty_input(&self) -> bool {
        match self {
            MultiInputError::EmptyInput => true,
            _ => false,
        }
    }

    /// Returns whether `self` is the `ShapeMismatch` variant.
    pub fn is_shape_mismatch(&self) -> bool {
        match self {
            MultiInputError::ShapeMismatch(_) => true,
            _ => false,
        }
    }
}

impl fmt::Display for MultiInputError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            MultiInputError::EmptyInput => write!(f, "Empty input."),
            MultiInputError::ShapeMismatch(e) => write!(f, "Shape mismatch: {}", e),
        }
    }
}

impl Error for MultiInputError {}

impl From<EmptyInput> for MultiInputError {
    fn from(_: EmptyInput) -> Self {
        MultiInputError::EmptyInput
    }
}

impl From<ShapeMismatch> for MultiInputError {
    fn from(err: ShapeMismatch) -> Self {
        MultiInputError::ShapeMismatch(err)
    }
}
