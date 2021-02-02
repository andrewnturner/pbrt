use std::cmp::PartialOrd;

use num_traits::Num as NumBase;

mod point;
mod vector;

pub use point::{Point2, Point3};
pub use vector::{Vector2, Vector2i, Vector2f, Vector3, Vector3i, Vector3f};

pub trait Num: NumBase + PartialOrd + Copy {
    fn is_nan(&self) -> bool;
}

impl<T: NumBase + PartialOrd + Copy> Num for T {
    #[inline]
    fn is_nan(&self) -> bool {
        self != self
    }
}
