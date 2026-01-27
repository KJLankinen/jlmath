use crate::{impl_assign_op_permutations, impl_bin_op_permutations, FloatType};
use num::traits::FloatConst;
use paste::paste;
use serde::{Deserialize, Serialize};
use std::{
    cmp::{Eq, Ordering, PartialEq, PartialOrd},
    fmt::Display,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

#[derive(Debug, Serialize, Deserialize, Copy, Clone)]
pub enum Angle<T> {
    Degrees(T),
    Radians(T),
    Turns(T),
}

impl<T> Angle<T>
where
    T: FloatType,
{
    pub fn new() -> Self {
        Angle::Radians(T::zero())
    }

    pub fn inner(&self) -> T {
        match self {
            Angle::Degrees(a) | Angle::Turns(a) | Angle::Radians(a) => *a,
        }
    }
}

// ===============================
// Free functions
// ===============================
fn normalize<T>(angle: Angle<T>) -> Angle<T>
where
    T: FloatType,
{
    fn normalization<S: FloatType>(t: S, full_round: S) -> S {
        // First from (possibly) negative to positive
        let turns = (t / full_round).abs().ceil();
        let t = t + turns * full_round;
        // 0 < t < full_round
        t - full_round * (t / full_round).floor()
    }

    match angle {
        Angle::Radians(t) => Angle::Radians(normalization(t, T::two_pi())),
        Angle::Degrees(t) => Angle::Degrees(normalization(t, T::three_sixty())),
        Angle::Turns(t) => Angle::Turns(normalization(t, T::one())),
    }
}

pub fn radians<T>(angle: Angle<T>) -> Angle<T>
where
    T: FloatType,
{
    match normalize(angle) {
        Angle::Degrees(t) => Angle::Degrees(t * FloatConst::PI() / T::one_eighty()),
        Angle::Radians(t) => Angle::Radians(t),
        Angle::Turns(t) => Angle::Turns(t * T::two_pi()),
    }
}

pub fn degrees<T>(angle: Angle<T>) -> Angle<T>
where
    T: FloatType,
{
    match normalize(angle) {
        Angle::Degrees(t) => Angle::Degrees(t),
        Angle::Radians(t) => Angle::Radians(t * T::one_eighty() / FloatConst::PI()),
        Angle::Turns(t) => Angle::Turns(t * T::three_sixty()),
    }
}

pub fn turns<T>(angle: Angle<T>) -> Angle<T>
where
    T: FloatType,
{
    match normalize(angle) {
        Angle::Degrees(t) => Angle::Degrees(t / T::three_sixty()),
        Angle::Radians(t) => Angle::Radians(t / T::two_pi()),
        Angle::Turns(t) => Angle::Turns(t),
    }
}

pub fn cos<T>(angle: Angle<T>) -> T
where
    T: FloatType,
{
    T::cos(radians(angle).inner())
}

pub fn sin<T>(angle: Angle<T>) -> T
where
    T: FloatType,
{
    T::sin(radians(angle).inner())
}

pub fn sin_cos<T>(angle: Angle<T>) -> (T, T)
where
    T: FloatType,
{
    T::sin_cos(radians(angle).inner())
}

pub fn tan<T>(angle: Angle<T>) -> T
where
    T: FloatType,
{
    T::tan(radians(angle).inner())
}

// ===============================
// Traits
// ===============================
impl<T> Default for Angle<T>
where
    T: FloatType,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Display for Angle<T>
where
    T: FloatType,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Angle::Radians(value) => {
                write!(f, "{} radians", value)
            }
            Angle::Degrees(value) => {
                write!(f, "{} degrees", value)
            }
            Angle::Turns(value) => {
                write!(f, "{} turns", value)
            }
        }
    }
}

impl<T> PartialEq for Angle<T>
where
    T: FloatType + PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        match *self {
            Angle::Degrees(a) => a == degrees(*other).inner(),
            Angle::Radians(a) => a == radians(*other).inner(),
            Angle::Turns(a) => a == turns(*other).inner(),
        }
    }
}

impl<T> Eq for Angle<T> where T: FloatType + Eq {}

impl<T> PartialOrd for Angle<T>
where
    T: FloatType + PartialOrd,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match *self {
            Angle::Degrees(a) => a.partial_cmp(&degrees(*other).inner()),
            Angle::Radians(a) => a.partial_cmp(&radians(*other).inner()),
            Angle::Turns(a) => a.partial_cmp(&turns(*other).inner()),
        }
    }

    fn lt(&self, other: &Self) -> bool {
        match *self {
            Angle::Degrees(a) => a.lt(&degrees(*other).inner()),
            Angle::Radians(a) => a.lt(&radians(*other).inner()),
            Angle::Turns(a) => a.lt(&turns(*other).inner()),
        }
    }

    fn le(&self, other: &Self) -> bool {
        match *self {
            Angle::Degrees(a) => a.le(&degrees(*other).inner()),
            Angle::Radians(a) => a.le(&radians(*other).inner()),
            Angle::Turns(a) => a.le(&turns(*other).inner()),
        }
    }

    fn gt(&self, other: &Self) -> bool {
        match *self {
            Angle::Degrees(a) => a.gt(&degrees(*other).inner()),
            Angle::Radians(a) => a.gt(&radians(*other).inner()),
            Angle::Turns(a) => a.gt(&turns(*other).inner()),
        }
    }

    fn ge(&self, other: &Self) -> bool {
        match *self {
            Angle::Degrees(a) => a.ge(&degrees(*other).inner()),
            Angle::Radians(a) => a.ge(&radians(*other).inner()),
            Angle::Turns(a) => a.ge(&turns(*other).inner()),
        }
    }
}

impl<T> Neg for Angle<T>
where
    T: FloatType,
{
    type Output = Self;
    fn neg(self) -> Self::Output {
        match self {
            Angle::Degrees(a) => normalize(Angle::Degrees(-a)),
            Angle::Radians(a) => normalize(Angle::Radians(-a)),
            Angle::Turns(a) => normalize(Angle::Turns(-a)),
        }
    }
}

impl<T> Neg for &Angle<T>
where
    T: FloatType,
{
    type Output = Angle<T>;
    fn neg(self) -> Self::Output {
        -(*self)
    }
}

impl<T> Neg for &mut Angle<T>
where
    T: FloatType,
{
    type Output = Angle<T>;
    fn neg(self) -> Self::Output {
        -(*self)
    }
}

macro_rules! impl_bin_op {
    (impl{$($tokens:tt)*}
     $trait:ident<$trait_type:ty>
     for $for_type:ty,
     Output = $output_type:ty) => {
        paste!(
            // Binary ops
            impl<'a, $($tokens)*> $trait<&'a $trait_type> for &'a $for_type {
                type Output = $output_type;
                fn [<$trait:lower>](self, rhs: &'a $trait_type) -> Self::Output {
                    match self {
                        Angle::Degrees(lhs) => {
                            let rhs = degrees(*rhs).inner();
                            normalize(Angle::Degrees((lhs).[<$trait:lower>](rhs)))
                        }
                        Angle::Radians(lhs) => {
                            let rhs = radians(*rhs).inner();
                            normalize(Angle::Radians((lhs).[<$trait:lower>](rhs)))
                        }
                        Angle::Turns(lhs) => {
                            let rhs = turns(*rhs).inner();
                            normalize(Angle::Turns((lhs).[<$trait:lower>](rhs)))
                        }
                    }
                }
            }

            // Assign ops
            impl<'a, $($tokens)*> [<$trait Assign>]<&'a $trait_type> for $for_type {
                fn [<$trait:lower _assign>](&mut self, rhs: &'a $trait_type) {
                    *self = self.[<$trait:lower>](rhs);
                }
            }
        );

        impl_bin_op_permutations!(
            impl{$($tokens)*}
            $trait<$trait_type>
            for $for_type,
            Output = $output_type);

        impl_assign_op_permutations!(
            impl{$($tokens)*}
            $trait<$trait_type>
            for $for_type);
    };
    (impl{$($tokens:tt)*}
     @scalar
     $trait:ident<$trait_type:ty>
     for $for_type:ty,
     Output = $output_type:ty) => {
        paste!(
            impl<'a, $($tokens)*> $trait<&'a $trait_type> for &'a $for_type {
                type Output = $output_type;
                fn [<$trait:lower>](self, rhs: &'a $trait_type) -> Self::Output {
                    match self {
                        Angle::Degrees(lhs) => {
                            normalize(Angle::Degrees((lhs).[<$trait:lower>](*rhs)))
                        }
                        Angle::Radians(lhs) => {
                            normalize(Angle::Radians((lhs).[<$trait:lower>](*rhs)))
                        }
                        Angle::Turns(lhs) => {
                            normalize(Angle::Turns((lhs).[<$trait:lower>](*rhs)))
                        }
                    }
                }
            }

            // Assign ops
            impl<'a, $($tokens)*> [<$trait Assign>]<&'a $trait_type> for $for_type {
                fn [<$trait:lower _assign>](&mut self, rhs: &'a $trait_type) {
                    *self = self.[<$trait:lower>](rhs);
                }
            }
        );

        impl_bin_op_permutations!(
            impl{$($tokens)*}
            $trait<$trait_type>
            for $for_type,
            Output = $output_type);

        impl_assign_op_permutations!(
            impl{$($tokens)*}
            $trait<$trait_type>
            for $for_type);
    };
    (impl{$($tokens:tt)*}
     @specialization
     $trait:ident<$trait_type:ty>
     for $for_type:ty,
     Output = $output_type:ty) => {
        paste!(
            impl<'a, $($tokens)*> $trait<&'a $trait_type> for &'a $for_type {
                type Output = $output_type;
                fn [<$trait:lower>](self, rhs: &'a $trait_type) -> Self::Output {
                    match rhs {
                        Angle::Degrees(rhs) => {
                            normalize(Angle::Degrees((self).[<$trait:lower>](*rhs)))
                        }
                        Angle::Radians(rhs) => {
                            normalize(Angle::Radians((self).[<$trait:lower>](*rhs)))
                        }
                        Angle::Turns(rhs) => {
                            normalize(Angle::Turns((self).[<$trait:lower>](*rhs)))
                        }
                    }
                }
            }
        );

        impl_bin_op_permutations!(
            impl{$($tokens)*}
            $trait<$trait_type>
            for $for_type,
            Output = $output_type);
    };
}

// Since angles have units, some operations are not sensible.
// Thus, Add and Sub are only implemented between angles, not between angles and unitless numbers.
// Similarly, Mul and Div are only implemented between angles and unitless numbers.

// Angle<T> + Angle<T>
// Angle<T> - Angle<T>
impl_bin_op!(impl{T: FloatType} Add<Angle<T>> for Angle<T>, Output = Angle<T>);
impl_bin_op!(impl{T: FloatType} Sub<Angle<T>> for Angle<T>, Output = Angle<T>);

// Angle<T> * T
// Angle<T> / T
impl_bin_op!(impl{T: FloatType} @scalar Mul<T> for Angle<T>, Output = Angle<T>);
impl_bin_op!(impl{T: FloatType} @scalar Div<T> for Angle<T>, Output = Angle<T>);

// f32 * Angle<T>
// f64 * Angle<T>
impl_bin_op!(impl{} @specialization Mul<Angle<f32>> for f32, Output = Angle<f32>);
impl_bin_op!(impl{} @specialization Mul<Angle<f64>> for f64, Output = Angle<f64>);

#[cfg(test)]
mod tests {
    use super::*;
    const LIMIT: f64 = 1e-6;

    #[test]
    fn new() {
        let angle = Angle::new();
        assert!(angle == Angle::Radians(0.0));
    }

    #[test]
    fn inner_test() {
        let value = 5.0;
        let angle = Angle::Radians(value);
        assert!(angle.inner() == value);
    }

    #[test]
    fn normalize_test() {
        let value = 8.0;
        let angle = Angle::Radians(value);
        let angle = normalize(angle);
        let inner = angle.inner();
        assert!((inner - (value - 2.0 * f64::PI())).abs() < LIMIT);
    }

    #[test]
    fn radians_test1() {
        let value = 8.0;
        let angle = Angle::Radians(value);
        let rads = radians(angle);
        assert!((rads.inner() - (value - 2.0 * f64::PI())).abs() < LIMIT);
    }

    #[test]
    fn radians_test2() {
        let value = 8.0;
        let angle = Angle::Turns(value);
        let rads = radians(angle);
        assert!(rads.inner() == 0.0);
    }

    #[test]
    fn radians_test3() {
        let value = 405.0;
        let angle = Angle::Degrees(value);
        let rads = radians(angle);
        assert!((rads.inner() - f64::PI() / 4.0).abs() < LIMIT);
    }

    #[test]
    fn degrees_test1() {
        let value = 8.0;
        let angle = Angle::Radians(value);
        let degs = degrees(angle);
        assert!((degs.inner() - 180.0 * (value - 2.0 * f64::PI()) / f64::PI()).abs() < LIMIT);
    }

    #[test]
    fn degrees_test2() {
        let value = 8.5;
        let angle = Angle::Turns(value);
        let degs = degrees(angle);
        assert!((degs.inner() - 180.0f64).abs() < LIMIT);
    }

    #[test]
    fn degrees_test3() {
        let value = 405.0;
        let angle = Angle::Degrees(value);
        let degs = degrees(angle);
        assert!((degs.inner() - 45.0f64).abs() < LIMIT);
    }

    #[test]
    fn turns_test1() {
        let value = 2.0 * f64::PI();
        let angle = Angle::Radians(value);
        let trns = turns(angle);
        assert!(trns.inner() == 0.0);
    }

    #[test]
    fn turns_test2() {
        let value = 8.5;
        let angle = Angle::Turns(value);
        let trns = turns(angle);
        assert!((trns.inner() - 0.5f64).abs() < LIMIT);
    }

    #[test]
    fn turns_test3() {
        let value = 405.0;
        let angle = Angle::Degrees(value);
        let trns = turns(angle);
        assert!((trns.inner() - 1.0 / 8.0f64).abs() < LIMIT);
    }

    #[test]
    fn add_test() {
        let d = 180.0;
        let r = f64::PI();
        let t = 0.5;
        let angles = [Angle::Degrees(d), Angle::Radians(r), Angle::Turns(t)];
        let results = [Angle::Degrees(0.0), Angle::Radians(0.0), Angle::Turns(0.0)];

        for (i, ai) in angles.iter().enumerate() {
            for aj in angles.iter() {
                println!("{}", ai + aj);
                println!("{}", results[i]);
                assert!(ai + aj == results[i]);
            }
        }
    }

    #[test]
    fn sub_test() {
        let d = 180.0;
        let r = f64::PI();
        let t = 0.5;
        let angles = [Angle::Degrees(d), Angle::Radians(r), Angle::Turns(t)];
        let results = [Angle::Degrees(0.0), Angle::Radians(0.0), Angle::Turns(0.0)];

        for (i, ai) in angles.iter().enumerate() {
            for aj in angles.iter() {
                println!("{}", ai - aj);
                println!("{}", results[i]);
                assert!(ai - aj == results[i]);
            }
        }
    }

    #[test]
    fn mul_test() {
        let d = 180.0;
        let r = f64::PI();
        let t = 0.5;
        let angles = [Angle::Degrees(d), Angle::Radians(r), Angle::Turns(t)];
        let results = [Angle::Degrees(0.0), Angle::Radians(0.0), Angle::Turns(0.0)];

        for (i, ai) in angles.iter().enumerate() {
            println!("{}", 2.0 * ai);
            println!("{}", ai * 2.0);
            println!("{}", results[i]);
            assert!(2.0 * ai == results[i]);
            assert!(ai * 2.0 == results[i]);
        }
    }

    #[test]
    fn div_test1() {
        let d = 180.0;
        let r = f64::PI();
        let t = 0.5;
        let angles = [Angle::Degrees(d), Angle::Radians(r), Angle::Turns(t)];
        let results = [
            Angle::Degrees(90.0),
            Angle::Radians(f64::FRAC_PI_2()),
            Angle::Turns(0.25),
        ];

        for (i, ai) in angles.iter().enumerate() {
            println!("{}", ai / 2.0);
            println!("{}", results[i]);
            assert!(ai / 2.0 == results[i]);
        }
    }

    #[test]
    fn test_neg_radians() {
        let angle = Angle::Radians(1.0);
        assert!((-angle - Angle::Radians(f64::two_pi() - 1.0)).inner().abs() < LIMIT);
        println!("{}", -angle);

        let angle = Angle::Radians(0.0);
        assert!(-angle == angle);
    }

    #[test]
    fn test_neg_degrees() {
        let angle = Angle::Degrees(90.0);
        println!("{}", -angle);
        assert!(-angle == Angle::Degrees(270.0));

        let angle = Angle::Degrees(0.0);
        assert!(-angle == angle);
    }

    #[test]
    fn test_neg_turns() {
        let angle = Angle::Turns(0.25);
        assert!(-angle == Angle::Turns(0.75));

        let angle = Angle::Turns(0.5);
        assert!(-angle == angle);

        let angle = Angle::Turns(0.0);
        assert!(-angle == angle);
    }

    #[test]
    fn ord_test() {
        let d = 188.0;
        let r = f64::PI() * 1.5;
        let t = 0.1321;
        let angles = [Angle::Degrees(d), Angle::Radians(r), Angle::Turns(t)];

        assert!(angles[0] < angles[1]);
        assert!(angles[0] <= angles[1]);
        assert!(angles[1] >= angles[0]);
        assert!(angles[1] > angles[0]);

        assert!(angles[0] > angles[2]);
        assert!(angles[0] >= angles[2]);
        assert!(angles[2] < angles[0]);
        assert!(angles[2] <= angles[0]);

        assert!(angles[1] > angles[2]);
        assert!(angles[1] >= angles[2]);
        assert!(angles[2] < angles[1]);
        assert!(angles[2] <= angles[1]);
    }

    #[test]
    fn eq_test() {
        let d = 180.0;
        let r = f64::PI();
        let t = 0.5;
        let angles = [Angle::Degrees(d), Angle::Radians(r), Angle::Turns(t)];

        assert!(angles[0] == angles[1]);
        assert!(angles[0] == angles[2]);
        assert!(angles[1] == angles[2]);
    }

    #[test]
    fn test_sin() {
        let d = 18.0;
        let r = f64::PI() * 123.0;
        let t = 0.5;
        let angles = [Angle::Degrees(d), Angle::Radians(r), Angle::Turns(t)];

        assert!((sin(angles[0]) - f64::sin(d * f64::PI() / 180.0)).abs() < LIMIT);
        assert!((sin(angles[1]) - f64::sin(r)).abs() < LIMIT);
        assert!((sin(angles[2]) - f64::sin(t * f64::two_pi())).abs() < LIMIT);
    }

    #[test]
    fn test_cos() {
        let d = 18.0;
        let r = f64::PI() * 123.0;
        let t = 0.5;
        let angles = [Angle::Degrees(d), Angle::Radians(r), Angle::Turns(t)];

        assert!((cos(angles[0]) - f64::cos(d * f64::PI() / 180.0)).abs() < LIMIT);
        assert!((cos(angles[1]) - f64::cos(r)).abs() < LIMIT);
        assert!((cos(angles[2]) - f64::cos(t * f64::two_pi())).abs() < LIMIT);
    }

    #[test]
    fn test_tan() {
        let d = 18.0;
        let r = f64::PI() * 123.0;
        let t = 0.5;
        let angles = [Angle::Degrees(d), Angle::Radians(r), Angle::Turns(t)];

        assert!((tan(angles[0]) - f64::tan(d * f64::PI() / 180.0)).abs() < LIMIT);
        assert!((tan(angles[1]) - f64::tan(r)).abs() < LIMIT);
        assert!((tan(angles[2]) - f64::tan(t * f64::two_pi())).abs() < LIMIT);
    }

    #[test]
    fn test_sin_cos() {
        let d = 18.0;
        let r = f64::PI() * 123.0;
        let t = 0.5;
        let angles = [Angle::Degrees(d), Angle::Radians(r), Angle::Turns(t)];

        let sc = sin_cos(angles[0]);
        let sc2 = f64::sin_cos(d * f64::PI() / 180.0);
        assert!((sc.0 - sc2.0).abs() < LIMIT);
        assert!((sc.1 - sc2.1).abs() < LIMIT);

        let sc = sin_cos(angles[1]);
        let sc2 = f64::sin_cos(r);
        assert!((sc.0 - sc2.0).abs() < LIMIT);
        assert!((sc.1 - sc2.1).abs() < LIMIT);

        let sc = sin_cos(angles[2]);
        let sc2 = f64::sin_cos(t * f64::two_pi());
        assert!((sc.0 - sc2.0).abs() < LIMIT);
        assert!((sc.1 - sc2.1).abs() < LIMIT);
    }
}
