use std::ops::{Add, AddAssign, Sub, SubAssign,  Mul, MulAssign, Div, DivAssign, Neg};
use std::cmp::PartialOrd;

use num_traits::{Num as NumBase, Float, Signed};

#[inline]
fn min<T: Num>(a: T, b: T) -> T {
    if a < b { a } else { b }
}

#[inline]
fn max<T: Num>(a: T, b: T) -> T {
    if a > b { a } else { b }
}

pub trait Num: NumBase + PartialOrd + Copy {
    fn is_nan(&self) -> bool;
}

impl<T: NumBase + PartialOrd + Copy> Num for T {
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

    pub fn dot(&self, other: &Self) -> T {
        (self.x * other.x) + (self.y * other.y)
    }

    pub fn length_squared(&self) -> T {
        (self.x * self.x) + (self.y * self.y)
    }

    pub fn min_component(&self) -> T {
        min(self.x, self.y)
    }

    pub fn max_component(&self) -> T {
        max(self.x, self.y)
    }

    pub fn min_dimension(&self) -> usize {
        if self.x < self.y {
            0
        } else {
            1
        }
    }

    pub fn max_dimension(&self) -> usize {
        if self.x > self.y {
            0
        } else {
            1
        }
    }

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

impl<T: Num + Signed> Vector2<T> {
    pub fn abs(&self) -> Self {
        Self {
            x: self.x.abs(),
            y: self.y.abs(),
        }
    }

    #[inline]
    pub fn abs_dot(self, other: Self) -> T {
        self.dot(&other).abs()
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

impl<T: Num + Signed> Neg for Vector2<T> {
    type Output = Self;

    fn neg(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
        }
    }
}

impl<T> Vector2<T> where T: Float, Vector2<T>: Div<T> {
    pub fn length(&self) -> T {
        self.length_squared().sqrt()
    }

    pub fn normalise(self) -> <Vector2<T> as Div<T>>::Output {
        let l = self.length();

        self / l
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

    pub fn dot(&self, other: &Self) -> T {
        (self.x * other.x) + (self.y * other.y) + (self.z * other.z)
    }

    pub fn cross(&self, other: &Self) -> Self {
        Self {
            x: (self.y * other.z) - (self.z * other.y),
            y: (self.z * other.x) - (self.x * other.z),
            z: (self.x * other.y) - (self.y * other.x),
        }
    }

    pub fn length_squared(&self) -> T {
        (self.x * self.x) + (self.y * self.y) + (self.z * self.z)
    }

    pub fn min_component(&self) -> T {
        min(self.x, min(self.y, self.z))
    }

    pub fn max_component(&self) -> T {
        max(self.x, max(self.y, self.z))
    }

    pub fn min_dimension(&self) -> usize {
        if self.x < self.y {
            if self.x < self.z {
                0
            } else {
                2
            }
        } else {
            if self.y < self.z {
                1
            } else {
                2
            }
        }
    }

    pub fn max_dimension(&self) -> usize {
        if self.x > self.y {
            if self.x > self.z {
                0
            } else {
                2
            }
        } else {
            if self.y > self.z {
                1
            } else {
                2
            }
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

impl<T: Num + Signed> Vector3<T> {
    pub fn abs(&self) -> Self {
        Self {
            x: self.x.abs(),
            y: self.y.abs(),
            z: self.z.abs(),
        }
    }

        #[inline]
    pub fn abs_dot(self, other: Self) -> T {
        self.dot(&other).abs()
    }
}

impl<T: Float> Vector3<T> where T: Float, Vector3<T>: Div<T, Output=Vector3<T>> {
    pub fn length(&self) -> T {
        self.length_squared().sqrt()
    }

    pub fn normalise(self) -> <Vector3<T> as Div<T>>::Output {
        let l = self.length();

        self / l
    }

    pub fn coordinate_system(&self) -> (Self, Self) {
        debug_assert!((self.length_squared() - T::one()).abs() < T::epsilon());

        let w: Self = if self.x.abs() > self.y.abs() {
            let a: Self = Self { x: -self.z, y: T::zero(), z: self.x };
            let b: T = ((self.x * self.x) + (self.z * self.z)).sqrt();
                a / b
        } else {
            Self { x: T::zero(), y: self.z, z: -self.y }
                / ((self.y * self.y) + (self.z * self.z)).sqrt()
        };

        let u = self.cross(&w);

        (w, u)
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
impl<T: Num + Signed> Neg for Vector3<T> {
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

    macro_rules! assert_delta {
        ($x:expr, $y:expr, $d:expr) => {
            if !($x - $y < $d || $y - $x < $d) { panic!(); }
        }
    }

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

    #[test]
    fn dot_vector2() {
        let v = Vector2i { x: -1, y: -2 };
        let w = Vector2i { x: 3, y: 4 };

        assert_eq!(v.dot(&w), -11);
    }

    #[test]
    fn dot_vector3() {
        let v = Vector3f { x: -1.0, y: -2.0, z: -3.0 };
        let w = Vector3f { x: 3.0, y: 4.0, z: 5.0 };

        assert_eq!(v.dot(&w), -26.0);
    }

    #[test]
    fn abs_dot_vector2() {
        let v = Vector2i { x: -1, y: -2 };
        let w = Vector2i { x: 3, y: 4 };

        assert_eq!(v.abs_dot(w), 11);
    }

    #[test]
    fn abs_dot_vector3() {
        let v = Vector3f { x: -1.0, y: -2.0, z: -3.0 };
        let w = Vector3f { x: 3.0, y: 4.0, z: 5.0 };

        assert_eq!(v.abs_dot(w), 26.0);
    }

    #[test]
    fn cross_vector3() {
        let v = Vector3f { x: -1.0, y: 2.0, z: -3.0 };
        let w = Vector3f { x: 4.0, y: 5.0, z: 6.0 };

        assert_eq!(v.cross(&w), Vector3f { x: 27.0, y: -6.0, z: -13.0 });
    }

    #[test]
    fn length_squared_vector2() {
        let v = Vector2i { x: 3, y: -4 };

        assert_eq!(v.length_squared(), 25);
    }

    #[test]
    fn length_vector2() {
        let v = Vector2f { x: 3.0, y: -4.0 };

        assert_eq!(v.length(), 5.0);
    }

    #[test]
    fn length_squared_vector3() {
        let v = Vector3f { x: 2.0, y: -10.0, z: 11.0 };

        assert_eq!(v.length_squared(), 225.0);
    }

    #[test]
    fn length_vector3() {
        let v = Vector3f { x: 2.0, y: -10.0, z: 11.0 };

        assert_eq!(v.length(), 15.0);
    }

    #[test]
    fn normalise_vector2() {
        let v = Vector2f { x: 2.0, y: 0.0 };

        assert_eq!(v.normalise(), Vector2f { x: 1.0, y: 0.0 });
    }

    #[test]
    fn normalise_vector3() {
        let v = Vector3f { x: 1.0, y: 1.0, z: 1.0 };

        let n = 1.0 / (3.0).sqrt();
        assert_eq!(v.normalise(), Vector3f { x: n, y: n, z: n });
    }

    #[test]
    fn min_component_vector2() {
        let v = Vector2i { x: 1, y: 2 };

        assert_eq!(v.min_component(), 1);
    }

    #[test]
    fn max_component_vector2() {
        let v = Vector2i { x: 1, y: 2 };

        assert_eq!(v.max_component(), 2);
    }

    #[test]
    fn min_component_vector3() {
        let v = Vector3f { x: 1.0, y: 2.0, z: 3.0 };

        assert_eq!(v.min_component(), 1.0);
    }

    #[test]
    fn max_component_vector3() {
        let v = Vector3f { x: 1.0, y: 2.0, z: 3.0 };

        assert_eq!(v.max_component(), 3.0);
    }

    #[test]
    fn min_dimension_vector2() {
        let v = Vector2i { x: 1, y: 2 };

        assert_eq!(v.min_dimension(), 0);
    }

    #[test]
    fn max_dimension_vector2() {
        let v = Vector2i { x: 1, y: 2 };

        assert_eq!(v.max_dimension(), 1);
    }

    #[test]
    fn min_dimension_vector3() {
        let v = Vector3f { x: 1.0, y: 2.0, z: 3.0 };

        assert_eq!(v.min_dimension(), 0);
    }

    #[test]
    fn max_dimension_vector3() {
        let v = Vector3f { x: 1.0, y: 2.0, z: 3.0 };

        assert_eq!(v.max_dimension(), 2);
    }

    #[test]
    fn min_vector2() {
        let v = Vector2i { x: 1, y: 2 };
        let w = Vector2i { x: 3, y: 0 };

        assert_eq!(v.min(&w), Vector2i { x: 1, y: 0 });
    }

    #[test]
    fn max_vector2() {
        let v = Vector2i { x: 1, y: 2 };
        let w = Vector2i { x: 3, y: 0 };

        assert_eq!(v.max(&w), Vector2i { x: 3, y: 2 });
    }

    #[test]
    fn min_vector3() {
        let v = Vector3f { x: 1.0, y: 2.0, z: 3.0 };
        let w = Vector3f { x: 3.0, y: 0.0, z: 1.0 };

        assert_eq!(v.min(&w), Vector3f { x: 1.0, y: 0.0, z: 1.0 });
    }

    #[test]
    fn max_vector3() {
        let v = Vector3f { x: 1.0, y: 2.0, z: 3.0 };
        let w = Vector3f { x: 3.0, y: 0.0, z: 1.0 };

        assert_eq!(v.max(&w), Vector3f { x: 3.0, y: 2.0, z: 3.0 });
    }

    #[test]
    fn index_vector2() {
        let v = Vector2i { x: 1, y: 2 };

        assert_eq!(v.index(1), 2);
    }

    #[test]
    fn permute_vector2() {
        let v = Vector2i { x: 1, y: 2 };

        assert_eq!(v.permute(1, 0), Vector2i { x: 2, y: 1 });
    }

    #[test]
    fn index_vector3() {
        let v = Vector3f { x: 1.0, y: 2.0, z: 3.0 };

        assert_eq!(v.index(1), 2.0);
    }

    #[test]
    fn permute_vector3() {
        let v = Vector3f { x: 1.0, y: 2.0, z: 3.0 };

        assert_eq!(v.permute(1, 2, 0), Vector3f { x: 2.0, y: 3.0, z: 1.0 });
    }

    #[test]
    fn coordinate_system_vector3() {
        let v = Vector3f { x: 1.0, y: 2.0, z: 3.0 }
            .normalise();

        let (w, u) = v.coordinate_system();

        assert_delta!(w.length_squared(), 1.0, 0.000001);
        assert_delta!(u.length_squared(), 1.0, 0.000001);
        assert_delta!(v.dot(&w), 0.0, 0.000001);
        assert_delta!(v.dot(&u), 0.0, 0.000001);
    }
}
