use std::ops::{Add, Sub, Mul, Div, Neg};

use num_traits::{NumCast, ToPrimitive};

use super::{Num, Vector2, Vector3};

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

impl<T: Num + ToPrimitive> Point2<T> {
    pub fn as_point2<S: Num + NumCast>(&self) -> Point2<S> {
        Point2 {
            x: S::from(self.x).unwrap(),
            y: S::from(self.y).unwrap(),
        }
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

impl<T: Num> Point3<T> {
    pub fn project_point2(&self) -> Point2<T> {
        Point2 {
            x: self.x,
            y: self.y,
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
    fn add_point2() {
        let p = Point2i { x: 1, y: 2 };
        let v = Vector2i { x: 1, y: 2 };

        assert_eq!(p + v, Point2i { x: 2, y: 4 });
    }

    #[test]
    fn add_point3() {
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
}
