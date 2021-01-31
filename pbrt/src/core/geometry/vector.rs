use std::ops::{Add, AddAssign, Sub, SubAssign,  Mul, MulAssign, Div, DivAssign, Neg};

use num_traits::{Num as NumBase, Float, Signed};

pub trait Num: NumBase + Copy {
    fn is_nan(&self) -> bool;
}

impl<T: NumBase + Copy> Num for T {
    #[inline]
    fn is_nan(&self) -> bool {
        self != self
    }
}

#[derive(Debug, PartialEq)]
pub struct Vector2<T: Num> {
    pub x: T,
    pub y: T,
}

type Vector2f = Vector2<f64>;
type Vector2i = Vector2<isize>;

#[derive(Debug, PartialEq)]
pub struct Vector3<T: Num> {
    pub x: T,
    pub y: T,
    pub z: T,
}

type Vector3f = Vector3<f64>;
type Vector3i = Vector3<isize>;

impl<T: Num> Vector2<T> {
    pub fn zero() -> Self {
        Vector2 {
            x: T::zero(),
            y: T::zero(),
        }
    }

    pub fn has_nans(&self) -> bool {
        self.x.is_nan() || self.y.is_nan()
    }
}

impl<T: Signed + Copy> Vector2<T> {
    pub fn abs(&self) -> Self {
        Self {
            x: self.x.abs(),
            y: self.y.abs(),
        }
    }
}

impl<T: Num> Add for Vector2<T> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self { x: self.x + other.x, y: self.y + other.y }
    }
}

impl<T: Num> AddAssign for Vector2<T> {
    fn add_assign(&mut self, other: Self) {
        self.x = self.x + other.x;
        self.y = self.y + other.y;
    }
}

impl<T: Num> Sub for Vector2<T> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self { x: self.x - other.x, y: self.y - other.y }
    }
}

impl<T: Num> SubAssign for Vector2<T> {
    fn sub_assign(&mut self, other: Self) {
        self.x = self.x - other.x;
        self.y = self.y - other.y;
    }
}

impl<T: Num> Mul<T> for Vector2<T> {
    type Output = Self;

    fn mul(self, scalar: T) -> Self {
        Self { x: self.x * scalar, y: self.y * scalar }
    }
}

impl<T: Num> MulAssign<T> for Vector2<T> {
    fn mul_assign(&mut self, scalar: T) {
        self.x = self.x * scalar;
        self.y = self.y * scalar;
    }
}

impl<T: Float> Div<T> for Vector2<T> {
    type Output = Vector2<T>;

    fn div(self, scalar: T) -> Vector2<T> {
        let inv = scalar.recip();

        Self { x: self.x * inv, y: self.y * inv }
    }
}

impl<T: Float> DivAssign<T> for Vector2<T> {
    fn div_assign(&mut self, scalar: T) {
        let inv = scalar.recip();

        self.x = self.x * inv;
        self.y = self.y * inv;
    }
}

impl<T: Signed + Copy> Neg for Vector2<T> {
    type Output = Self;

    fn neg(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
        }
    }
}

impl<T: Num> Vector3<T> {
    pub fn zero() -> Self {
        Vector3 {
            x: T::zero(),
            y: T::zero(),
            z: T::zero(),
        }
    }

    pub fn has_nans(&self) -> bool {
        self.x.is_nan() || self.y.is_nan() || self.z.is_nan()
    }
}

impl<T: Signed + Copy> Vector3<T> {
    pub fn abs(&self) -> Self {
        Self {
            x: self.x.abs(),
            y: self.y.abs(),
            z: self.z.abs(),
        }
    }
}

impl<T: Num> Add for Vector3<T> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self { x: self.x + other.x, y: self.y + other.y, z: self.z + other.z }
    }
}

impl<T: Num> AddAssign for Vector3<T> {
    fn add_assign(&mut self, other: Self) {
        self.x = self.x + other.x;
        self.y = self.y + other.y;
        self.z = self.z + other.z;
    }
}

impl<T: Num> Sub for Vector3<T> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self { x: self.x - other.x, y: self.y - other.y, z: self.z - other.z }
    }
}

impl<T: Num> SubAssign for Vector3<T> {
    fn sub_assign(&mut self, other: Self) {
        self.x = self.x - other.x;
        self.y = self.y - other.y;
        self.z = self.z - other.z;
    }
}

impl<T: Num> Mul<T> for Vector3<T> {
    type Output = Self;

    fn mul(self, scalar: T) -> Self {
        Self { x: self.x * scalar, y: self.y * scalar, z: self.z * scalar }
    }
}

impl<T: Num> MulAssign<T> for Vector3<T> {
    fn mul_assign(&mut self, scalar: T) {
        self.x = self.x * scalar;
        self.y = self.y * scalar;
        self.z = self.z * scalar;
    }
}

impl<T: Float> Div<T> for Vector3<T> {
    type Output = Vector3<T>;

    fn div(self, scalar: T) -> Vector3<T> {
        debug_assert!(scalar != T::zero());
        let inv = scalar.recip();

        Self { x: self.x * inv, y: self.y * inv, z: self.z * inv }
    }
}

impl<T: Float> DivAssign<T> for Vector3<T> {
    fn div_assign(&mut self, scalar: T) {
        debug_assert!(scalar != T::zero());
        let inv = scalar.recip();

        self.x = self.x * inv;
        self.y = self.y * inv;
        self.z = self.z * inv;
    }
}
impl<T: Signed + Copy> Neg for Vector3<T> {
    type Output = Self;

    fn neg(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn make_zero_vector2() {
        assert_eq!(
            Vector2i::zero(),
            Vector2i { x: 0, y: 0 },
        );
    }

    #[test]
    fn make_zero_vector3() {
        assert_eq!(
            Vector3f::zero(),
            Vector3f { x: 0.0, y: 0.0, z: 0.0 },
        );
    }

    #[test]
    fn check_no_nans_vector2() {
        let v = Vector2i { x: 0, y: 0 };

        assert_eq!(v.has_nans(), false);
    }

    #[test]
    fn check_nans_vector2() {
        let v = Vector2f { x: 0.0, y: f64::NAN };

        assert_eq!(v.has_nans(), true);
    }

    #[test]
    fn check_no_nans_vector3() {
        let v = Vector3f { x: 0.0, y: 0.0, z: 0.0 };

        assert_eq!(v.has_nans(), false);
    }

    #[test]
    fn check_nans_vector3() {
        let v = Vector3f { x: 0.0, y: f64::NAN, z: 0.0 };

        assert_eq!(v.has_nans(), true);
    }

    #[test]
    fn add_vector2() {
        let v = Vector2i { x: 1, y: 2 };
        let w = Vector2i { x: 2, y: 3, };

        assert_eq!(v + w, Vector2i { x: 3, y: 5 });
    }

    #[test]
    fn add_assign_vector2() {
        let mut v = Vector2i { x: 1, y: 2 };
        v += Vector2i { x: 2, y: 3, };

        assert_eq!(v, Vector2i { x: 3, y: 5 });
    }

    #[test]
    fn add_vector3() {
        let v = Vector3f { x: 1.0, y: 2.0, z: 3.0 };
        let w = Vector3f { x: 2.0, y: 3.0, z: 4.0 };

        assert_eq!(v + w, Vector3f { x: 3.0, y: 5.0, z: 7.0 });
    }

    #[test]
    fn add_assign_vector3() {
        let mut v = Vector3f { x: 1.0, y: 2.0, z: 3.0 };
        v += Vector3f { x: 2.0, y: 3.0, z: 4.0 };

        assert_eq!(v, Vector3f { x: 3.0, y: 5.0, z: 7.0 });
    }

    #[test]
    fn subtract_vector2() {
        let v = Vector2i { x: 5, y: 7 };
        let w = Vector2i { x: 2, y: 3, };

        assert_eq!(v - w, Vector2i { x: 3, y: 4 });
    }

    #[test]
    fn subtract_assign_vector2() {
        let mut v = Vector2i { x: 5, y: 7 };
        v -= Vector2i { x: 2, y: 3, };

        assert_eq!(v, Vector2i { x: 3, y: 4 });
    }

    #[test]
    fn subtract_vector3() {
        let v = Vector3f { x: 5.0, y: 7.0, z: 9.0 };
        let w = Vector3f { x: 2.0, y: 3.0, z: 4.0 };

        assert_eq!(v - w, Vector3f { x: 3.0, y: 4.0, z: 5.0 });
    }

    #[test]
    fn subtract_assign_vector3() {
        let mut v = Vector3f { x: 5.0, y: 7.0, z: 9.0 };
        v -= Vector3f { x: 2.0, y: 3.0, z: 4.0 };

        assert_eq!(v, Vector3f { x: 3.0, y: 4.0, z: 5.0 });
    }

    #[test]
    fn right_multiply_vector2() {
        let v = Vector2i { x: 1, y: 2 };

        assert_eq!(v * 2, Vector2i { x: 2, y: 4 });
    }

    #[test]
    fn multiply_assign_vector2() {
        let mut v = Vector2i { x: 1, y: 2 };
        v *= 2;

        assert_eq!(v, Vector2i { x: 2, y: 4 });
    }

    #[test]
    fn right_multiply_vector3() {
        let v = Vector3f { x: 1.0, y: 2.0, z: 3.0 };

        assert_eq!(v * 2.0, Vector3f { x: 2.0, y: 4.0, z: 6.0 });
    }

    #[test]
    fn multiply_assign_vector3() {
        let mut v = Vector3f { x: 1.0, y: 2.0, z: 3.0 };
        v *= 2.0;

        assert_eq!(v, Vector3f { x: 2.0, y: 4.0, z: 6.0 });
    }

    #[test]
    fn divide_vector2() {
        let v = Vector2f { x: 2.0, y: 4.0 };

        assert_eq!(v / 2.0, Vector2f { x: 1.0, y: 2.0 });
    }

    #[test]
    fn divide_assign_vector2() {
        let mut v = Vector2f { x: 2.0, y: 4.0 };
        v /= 2.0;

        assert_eq!(v, Vector2f { x: 1.0, y: 2.0 });
    }

    #[test]
    fn divide_vector3() {
        let v = Vector3f { x: 2.0, y: 4.0, z: 6.0 };

        assert_eq!(v / 2.0, Vector3f { x: 1.0, y: 2.0, z: 3.0 });
    }

    #[test]
    fn divide_assign_vector3() {
        let mut v = Vector3f { x: 2.0, y: 4.0, z: 6.0 };
        v /= 2.0;

        assert_eq!(v, Vector3f { x: 1.0, y: 2.0, z: 3.0 });
    }

    #[test]
    fn negate_vector2() {
        let v = Vector2i { x: 1, y: -2 };

        assert_eq!(-v, Vector2i { x: -1, y: 2 });
    }

    #[test]
    fn negate_vector3() {
        let v = Vector3f { x: 1.0, y: -2.0, z: 3.0 };

        assert_eq!(-v, Vector3f { x: -1.0, y: 2.0, z: -3.0 });
    }

    #[test]
    fn abs_vector2() {
        let v = Vector2i { x: 1, y: 2 };

        assert_eq!(v.abs(), Vector2i { x: 1, y: 2 });
    }

    #[test]
    fn abs_vector3() {
        let v = Vector3f { x: 1.0, y: -2.0, z: 3.0 };

        assert_eq!(v.abs(), Vector3f { x: 1.0, y: 2.0, z: 3.0 });
    }
}
