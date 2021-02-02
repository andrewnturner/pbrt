use super::Num;

#[inline]
pub fn min<T: Num>(a: T, b: T) -> T {
    if a < b { a } else { b }
}

#[inline]
pub fn max<T: Num>(a: T, b: T) -> T {
    if a > b { a } else { b }
}
