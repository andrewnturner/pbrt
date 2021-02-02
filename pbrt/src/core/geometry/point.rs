use std::ops::{Add, Sub, Mul};

use num_traits::{NumCast, ToPrimitive, Signed, Float};

use super::{Num, Vector2, Vector3};
use super::util::{min, max};

#[derive(Debug, PartialEq)]
pub struct Point2<T: Num> {
    pub x: T,
    pub y: T,
}

type Point2f = Point2<f64>;
type Point2i = Point2<isize>;

#[derive(Debug, PartialEq)]
pub struct Point3<T: Num> {
    pub x: T,
    pub y: T,
    pub z: T,
}

type Point3f = Point3<f64>;
type Point3i = Point3<isize>;

impl<T: Num> Point2<T> {
    pub fn min(&self, other: &Self) -> Self {
        Self {
            x: min(self.x, other.x),
            y: min(self.y, other.y),
        }
    }

    pub fn max(&self, other: &Self) -> Self {
        Self {
            x: max(self.x, other.x),
            y: max(self.y, other.y),
        }
    }

    pub fn index(&self, i: usize) -> T {
        match i {
            0 => self.x,
            1 => self.y,
            _ => panic!("Invalid index"),
        }
    }

    pub fn permute(&self, x_index: usize, y_index: usize) -> Self {
        Self {
            x: self.index(x_index),
            y: self.index(y_index),
        }
    }
}

impl<T: Num + ToPrimitive> Point2<T> {
    pub fn as_point2<S: Num + NumCast>(&self) -> Point2<S> {
        Point2 {
            x: S::from(self.x).unwrap(),
            y: S::from(self.y).unwrap(),
        }
    }

    pub fn lerp(&self, other: &Self, t: f64) -> Point2f {
        let self_f = self.as_point2::<f64>();
        let other_f = other.as_point2::<f64>();

        (self_f * (1.0 - t)) + (other_f * t)
    }
}

impl<T: Num + Signed> Point2<T> {
    pub fn abs(&self) -> Self {
        Self {
            x: self.x.abs(),
            y: self.y.abs(),
        }
    }
}

impl<T: Num + Float> Point2<T> {
    pub fn floor(&self) -> Self {
        Self {
            x: self.x.floor(),
            y: self.y.floor(),
        }
    }

    pub fn ceil(&self) -> Self {
        Self {
            x: self.x.ceil(),
            y: self.y.ceil(),
        }
    }
}

impl<T: Num> Add for Point2<T> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self { x: self.x + other.x, y: self.y + other.y }
    }
}

impl <T: Num> Add<Vector2<T>> for Point2<T> {
    type Output = Self;

    fn add(self, v: Vector2<T>) -> Self {
        Self { x: self.x + v.x, y: self.y + v.y }
    }
}

impl <T: Num> Sub for Point2<T> {
    type Output = Vector2<T>;

    fn sub(self, other: Self) -> Vector2<T> {
        Vector2 { x: self.x - other.x, y: self.y - other.y }
    }
}

impl <T: Num> Sub<Vector2<T>> for Point2<T> {
    type Output = Self;

    fn sub(self, v: Vector2<T>) -> Self {
        Self { x: self.x - v.x, y: self.y - v.y }
    }
}

impl<T: Num> Mul<T> for Point2<T> {
    type Output = Self;

    fn mul(self, scalar: T) -> Self {
        Self { x: self.x * scalar, y: self.y * scalar }
    }
}

impl<T: Num> Point3<T> {
    pub fn project_point2(&self) -> Point2<T> {
        Point2 {
            x: self.x,
            y: self.y,
        }
    }

    pub fn min(&self, other: &Self) -> Self {
        Self {
            x: min(self.x, other.x),
            y: min(self.y, other.y),
            z: min(self.z, other.z),
        }
    }

    pub fn max(&self, other: &Self) -> Self {
        Self {
            x: max(self.x, other.x),
            y: max(self.y, other.y),
            z: max(self.z, other.z),
        }
    }

    pub fn index(&self, i: usize) -> T {
        match i {
            0 => self.x,
            1 => self.y,
            2 => self.z,
            _ => panic!("Invalid index"),
        }
    }

    pub fn permute(&self, x_index: usize, y_index: usize, z_index: usize) -> Self {
        Self {
            x: self.index(x_index),
            y: self.index(y_index),
            z: self.index(z_index),
        }
    }
}

impl<T: Num + ToPrimitive> Point3<T> {
    pub fn as_point3<S: Num + NumCast>(&self) -> Point3<S> {
        Point3 {
            x: S::from(self.x).unwrap(),
            y: S::from(self.y).unwrap(),
            z: S::from(self.z).unwrap(),
        }
    }

    pub fn lerp(&self, other: &Self, t: f64) -> Point3f {
        let self_f = self.as_point3::<f64>();
        let other_f = other.as_point3::<f64>();

        (self_f * (1.0 - t)) + (other_f * t)
    }
}

impl<T: Num + Signed> Point3<T> {
    pub fn abs(&self) -> Self {
        Self {
            x: self.x.abs(),
            y: self.y.abs(),
            z: self.z.abs(),
        }
    }
}

impl<T: Num + Float> Point3<T> {
    pub fn floor(&self) -> Self {
        Self {
            x: self.x.floor(),
            y: self.y.floor(),
            z: self.z.floor(),
        }
    }

    pub fn ceil(&self) -> Self {
        Self {
            x: self.x.ceil(),
            y: self.y.ceil(),
            z: self.z.ceil(),
        }
    }
}

impl<T: Num> Add for Point3<T> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self { x: self.x + other.x, y: self.y + other.y, z: self.z + other.z }
    }
}
impl <T: Num> Add<Vector3<T>> for Point3<T> {
    type Output = Self;

    fn add(self, v: Vector3<T>) -> Self {
        Self { x: self.x + v.x, y: self.y + v.y, z: self.z + v.z }
    }
}

impl <T: Num> Sub for Point3<T> {
    type Output = Vector3<T>;

    fn sub(self, other: Self) -> Vector3<T> {
        Vector3 { x: self.x - other.x, y: self.y - other.y, z: self.z - other.z }
    }
}

impl <T: Num> Sub<Vector3<T>> for Point3<T> {
    type Output = Self;

    fn sub(self, v: Vector3<T>) -> Self {
        Self { x: self.x - v.x, y: self.y - v.y, z: self.z - v.z }
    }
}

impl<T: Num> Mul<T> for Point3<T> {
    type Output = Self;

    fn mul(self, scalar: T) -> Self {
        Self { x: self.x * scalar, y: self.y * scalar, z: self.z * scalar }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::{Vector2i, Vector3f};

    #[test]
    fn project_point2() {
        let p = Point3f { x: 1.0, y: 2.0, z: 3.0};

        assert_eq!(p.project_point2(), Point2f { x: 1.0, y: 2.0 });
    }

    #[test]
    fn as_point2_integer() {
        let p = Point2f { x: 1.0, y: 2.0 };

        assert_eq!(p.as_point2::<isize>(), Point2i { x: 1, y: 2 });
    }

    #[test]
    fn as_point2_float() {
        let p = Point2i { x: 1, y: 2 };

        assert_eq!(p.as_point2::<f64>(), Point2f { x: 1.0, y: 2.0 });
    }

    #[test]
    fn as_point3_integer() {
        let p = Point3f { x: 1.0, y: 2.0, z: 3.0 };

        assert_eq!(p.as_point3::<isize>(), Point3i { x: 1, y: 2, z: 3 });
    }

    #[test]
    fn as_point3_float() {
        let p = Point3i { x: 1, y: 2, z: 3 };

        assert_eq!(p.as_point3::<f64>(), Point3f { x: 1.0, y: 2.0, z: 3.0 });
    }

    #[test]
    fn add_vector2_to_point2() {
        let p = Point2i { x: 1, y: 2 };
        let v = Vector2i { x: 1, y: 2 };

        assert_eq!(p + v, Point2i { x: 2, y: 4 });
    }

    #[test]
    fn add_vector3_to_point3() {
        let p = Point3f { x: 1.0, y: 2.0, z: 3.0 };
        let v = Vector3f { x: 1.0, y: 2.0, z: 1.0 };

        assert_eq!(p + v, Point3f { x: 2.0, y: 4.0, z: 4.0 });
    }

    #[test]
    fn subtract_point_point2() {
        let p = Point2i { x: 1, y: 2 };
        let q = Point2i { x: 2, y: 1 };

        assert_eq!(p - q, Vector2i { x: -1, y: 1 });
    }

    #[test]
    fn subtract_point_point3() {
        let p = Point3f { x: 1.0, y: 2.0, z: 3.0 };
        let q = Point3f { x: 2.0, y: 1.0, z: -1.0 };

        assert_eq!(p - q, Vector3f { x: -1.0, y: 1.0, z: 4.0 });
    }

    #[test]
    fn subtract_vector_point2() {
        let p = Point2i { x: 1, y: 2 };
        let v = Vector2i { x: 2, y: 1 };

        assert_eq!(p - v, Point2i { x: -1, y: 1 });
    }

    #[test]
    fn subtract_vector_point3() {
        let p = Point3f { x: 1.0, y: 2.0, z: 3.0 };
        let v = Vector3f { x: 2.0, y: 1.0, z: -1.0 };

        assert_eq!(p - v, Point3f { x: -1.0, y: 1.0, z: 4.0 });
    }

    #[test]
    fn add_point2() {
        let v = Point2i { x: 1, y: 2 };
        let w = Point2i { x: 2, y: 3, };

        assert_eq!(v + w, Point2i { x: 3, y: 5 });
    }

    #[test]
    fn add_point3() {
        let v = Point3f { x: 1.0, y: 2.0, z: 3.0 };
        let w = Point3f { x: 2.0, y: 3.0, z: 4.0 };

        assert_eq!(v + w, Point3f { x: 3.0, y: 5.0, z: 7.0 });
    }

    #[test]
    fn right_multiply_point2() {
        let v = Point2i { x: 1, y: 2 };

        assert_eq!(v * 2, Point2i { x: 2, y: 4 });
    }

    #[test]
    fn right_multiply_point3() {
        let v = Point3f { x: 1.0, y: 2.0, z: 3.0 };

        assert_eq!(v * 2.0, Point3f { x: 2.0, y: 4.0, z: 6.0 });
    }

    #[test]
    fn lerp_point2() {
        let v = Point2i { x: 1, y: 2 };
        let w = Point2i { x: 3, y: 4, };

        assert_eq!(v.lerp(&w, 0.5), Point2f { x: 2.0, y: 3.0 });
    }

    #[test]
    fn lerp_point3() {
        let v = Point3f { x: 1.0, y: 2.0, z: 3.0 };
        let w = Point3f { x: 3.0, y: 4.0, z: 5.0 };

        assert_eq!(v.lerp(&w, 0.5), Point3f { x: 2.0, y: 3.0, z: 4.0 });
    }

    #[test]
    fn min_point2() {
        let p = Point2i { x: 1, y: 2 };
        let q = Point2i { x: 3, y: 0 };

        assert_eq!(p.min(&q), Point2i { x: 1, y: 0 });
    }

    #[test]
    fn max_point2() {
        let p = Point2i { x: 1, y: 2 };
        let q = Point2i { x: 3, y: 0 };

        assert_eq!(p.max(&q), Point2i { x: 3, y: 2 });
    }

    #[test]
    fn min_point3() {
        let p = Point3f { x: 1.0, y: 2.0, z: 3.0 };
        let q = Point3f { x: 3.0, y: 0.0, z: 1.0 };

        assert_eq!(p.min(&q), Point3f { x: 1.0, y: 0.0, z: 1.0 });
    }

    #[test]
    fn max_point3() {
        let p = Point3f { x: 1.0, y: 2.0, z: 3.0 };
        let q = Point3f { x: 3.0, y: 0.0, z: 1.0 };

        assert_eq!(p.max(&q), Point3f { x: 3.0, y: 2.0, z: 3.0 });
    }

    #[test]
    fn abs_point2() {
        let v = Point2i { x: 1, y: 2 };

        assert_eq!(v.abs(),Point2i { x: 1, y: 2 });
    }

    #[test]
    fn abs_point3() {
        let v = Point3f { x: 1.0, y: -2.0, z: 3.0 };

        assert_eq!(v.abs(),Point3f { x: 1.0, y: 2.0, z: 3.0 });
    }

    #[test]
    fn floor_point2() {
        let v = Point2f { x: 1.5, y: -1.5 };

        assert_eq!(v.floor(),Point2f { x: 1.0, y: -2.0 });
    }

    #[test]
    fn floor_point3() {
        let v = Point3f { x: 1.5, y: -1.5, z: 2.5 };

        assert_eq!(v.floor(), Point3f { x: 1.0, y: -2.0, z: 2.0 });
    }

    #[test]
    fn ceil_point2() {
        let v = Point2f { x: 1.5, y: -1.5 };

        assert_eq!(v.ceil(), Point2f { x: 2.0, y: -1.0 });
    }

    #[test]
    fn ceil_point3() {
        let v = Point3f { x: 1.5, y: -1.5, z: 2.5 };

        assert_eq!(v.ceil(), Point3f { x: 2.0, y: -1.0, z: 3.0 });
    }

    #[test]
    fn index_point2() {
        let v = Point2i { x: 1, y: 2 };

        assert_eq!(v.index(1), 2);
    }

    #[test]
    fn permute_point2() {
        let v = Point2i { x: 1, y: 2 };

        assert_eq!(v.permute(1, 0), Point2i { x: 2, y: 1 });
    }

    #[test]
    fn index_point3() {
        let v = Point3f { x: 1.0, y: 2.0, z: 3.0 };

        assert_eq!(v.index(1), 2.0);
    }

    #[test]
    fn permute_point3() {
        let v = Point3f { x: 1.0, y: 2.0, z: 3.0 };

        assert_eq!(v.permute(1, 2, 0), Point3f { x: 2.0, y: 3.0, z: 1.0 });
    }
}
