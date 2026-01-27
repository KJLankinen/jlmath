use crate::{
    angle::*, impl_assign_op_permutations, impl_bin_op_permutations, length_sq, normalize,
    vector::*, FloatType, Inverse, Length, LengthSq, Normalize,
};
use paste::paste;
use serde::{Deserialize, Serialize};
use std::fmt::Display;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

#[allow(dead_code)]
pub type Quat = Quaternion<f32>;

#[derive(Default, Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "T: FloatType")]
pub struct Quaternion<T>
where
    T: FloatType,
{
    #[serde(flatten)]
    components: Vector<T, 4>,
}

impl<T> Quaternion<T>
where
    T: FloatType,
{
    pub fn new(components: impl Into<Vector<T, 4>>) -> Self {
        Self {
            components: components.into(),
        }
    }

    pub fn identity() -> Self {
        Self::new([T::zero(), T::zero(), T::zero(), T::one()])
    }

    pub fn from_axis_angle(axis: &Vector<T, 3>, angle: Angle<T>) -> Self {
        let angle = angle * T::half();
        let v = normalize!(axis) * sin(angle);
        Self::new([v.x(), v.y(), v.z(), cos(angle)])
    }

    pub fn rotator_from_to(from: &Vector<T, 3>, to: &Vector<T, 3>) -> Self {
        let from = normalize!(from);
        let to = normalize!(to);
        let dp = dot(&from, &to);
        if dp == T::one() {
            // Already the same vectors
            Self::identity()
        } else if dp == -T::one() {
            // Opposite vectors, choose arbitrary perpendicular vector as the axis
            let axis = normalize!(&cross(&from, &Vector::<T, 3>::new([-to[2], to[0], to[1]])));
            let angle = Angle::Radians(num::traits::FloatConst::PI());
            Self::from_axis_angle(&axis, angle)
        } else if length_sq!(&to) == T::zero() || length_sq!(&from) == T::zero() {
            // One or both are the zero vector
            panic!("There is no quaternion that can convert a vector to the zero vector");
        } else {
            let axis = normalize!(&cross(&from, &to));
            let angle = Angle::Radians(T::acos(dp));
            Self::from_axis_angle(&axis, angle)
        }
    }

    pub fn x(&self) -> T {
        self.components.x()
    }

    pub fn y(&self) -> T {
        self.components.y()
    }

    pub fn z(&self) -> T {
        self.components.z()
    }

    pub fn w(&self) -> T {
        self.components.w()
    }

    pub fn axis(&self) -> Vector<T, 3> {
        normalize!(&Vector::<T, 3>::from_vec(self.components))
    }

    pub fn angle(&self) -> Angle<T> {
        Angle::Radians(T::two() * T::acos(self.w()))
    }

    pub fn rotate(&self, rhs: &Vector<T, 3>) -> Vector<T, 3> {
        let two = T::from(2.0).unwrap();
        let v = Vector::<T, 3>::new([self.x(), self.y(), self.z()]);
        let mut rotated = rhs * (self.w() * self.w() - dot(&v, &v));
        rotated += v * dot(&v, rhs) * two;
        rotated += cross(&v, rhs) * self.w() * two;
        rotated
    }
}

// ===================================================
// Free functions
// ===================================================
pub fn conjugate<T>(lhs: &Quaternion<T>) -> Quaternion<T>
where
    T: FloatType,
{
    let mul_vec = Vector::<T, 2>::new([-T::one(), T::one()]);
    Quaternion::<T>::new(lhs.components * mul_vec.xxxy())
}

// ===================================================
// Traits
// ===================================================
impl<T> LengthSq for Quaternion<T>
where
    T: FloatType,
{
    type Output = T;
    fn length_sq(&self) -> Self::Output {
        length_sq!(&self.components)
    }
}

impl<T> Length for Quaternion<T>
where
    T: FloatType,
{
    type Output = T;
    fn length(&self) -> Self::Output {
        T::sqrt(length_sq!(self))
    }
}

impl<T> Normalize for Quaternion<T>
where
    T: FloatType,
{
    fn normalize(&self) -> Self {
        Self::new(normalize!(&self.components))
    }
}

impl<T> Inverse for Quaternion<T>
where
    T: FloatType,
{
    fn inverse(&self) -> Self {
        let inv_l_sq = T::one() / length_sq!(self);
        conjugate(self) * inv_l_sq
    }
}

impl<T> Display for Quaternion<T>
where
    T: FloatType,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.components)
    }
}

// Slice to quaternion
impl<T> From<&[T; 4]> for Quaternion<T>
where
    T: FloatType,
{
    fn from(data: &[T; 4]) -> Self {
        Self {
            components: Into::into(data),
        }
    }
}

impl<T> From<[T; 4]> for Quaternion<T>
where
    T: FloatType,
{
    fn from(data: [T; 4]) -> Self {
        Quaternion::<T>::from(&data)
    }
}

impl<T> From<&mut [T; 4]> for Quaternion<T>
where
    T: FloatType,
{
    fn from(data: &mut [T; 4]) -> Self {
        Quaternion::<T>::from(&*data)
    }
}

// Quaternion to slice
impl<T> From<&Quaternion<T>> for [T; 4]
where
    T: FloatType,
{
    fn from(quat: &Quaternion<T>) -> Self {
        Into::into(quat.components)
    }
}

impl<T> From<Quaternion<T>> for [T; 4]
where
    T: FloatType,
{
    fn from(quat: Quaternion<T>) -> Self {
        <[T; 4]>::from(&quat)
    }
}

impl<T> From<&mut Quaternion<T>> for [T; 4]
where
    T: FloatType,
{
    fn from(quat: &mut Quaternion<T>) -> Self {
        <[T; 4]>::from(&*quat)
    }
}

impl<'a, T: FloatType> Mul<&'a Quaternion<T>> for &'a Quaternion<T> {
    type Output = Quaternion<T>;
    fn mul(self, rhs: &'a Quaternion<T>) -> Self::Output {
        Self::Output::new([
            self.w() * rhs.x() + self.x() * rhs.w() + self.y() * rhs.z() - self.z() * rhs.y(),
            self.w() * rhs.y() - self.x() * rhs.z() + self.y() * rhs.w() + self.z() * rhs.x(),
            self.w() * rhs.z() + self.x() * rhs.y() - self.y() * rhs.x() + self.z() * rhs.w(),
            self.w() * rhs.w() - self.x() * rhs.x() - self.y() * rhs.y() - self.z() * rhs.z(),
        ])
    }
}

impl<'a, T: FloatType> Mul<&'a Vector<T, 3>> for &'a Quaternion<T> {
    type Output = Quaternion<T>;
    fn mul(self, rhs: &'a Vector<T, 3>) -> Self::Output {
        self * Quaternion::<T>::new([rhs.x(), rhs.y(), rhs.z(), T::zero()])
    }
}

impl<'a, T: FloatType> Mul<&'a Quaternion<T>> for &'a Vector<T, 3> {
    type Output = Quaternion<T>;
    fn mul(self, rhs: &'a Quaternion<T>) -> Self::Output {
        Quaternion::<T>::new([self.x(), self.y(), self.z(), T::zero()]) * rhs
    }
}

impl_bin_op_permutations!(impl{T: FloatType} Mul<Quaternion<T>> for Quaternion<T>, Output = Quaternion<T>);
impl_bin_op_permutations!(impl{T: FloatType} Mul<Vector<T, 3>> for Quaternion<T>, Output = Quaternion<T>);
impl_bin_op_permutations!(impl{T: FloatType} Mul<Quaternion<T>> for Vector<T, 3>, Output = Quaternion<T>);

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
                    Self::Output{
                        components: self.components.[<$trait:lower>](rhs.components),
                    }
                }
            }

            // Assign ops
            impl<'a, $($tokens)*> [<$trait Assign>]<&'a $trait_type> for $for_type {
                fn [<$trait:lower _assign>](&mut self, rhs: &'a $trait_type) {
                    self.components = self.components.[<$trait:lower>](rhs.components);
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
                    Self::Output{
                        components: self.components.[<$trait:lower>](rhs),
                    }
                }
            }

            // Assign ops
            impl<'a, $($tokens)*> [<$trait Assign>]<&'a $trait_type> for $for_type {
                fn [<$trait:lower _assign>](&mut self, rhs: &'a $trait_type) {
                    self.components = self.components.[<$trait:lower>](rhs);
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
                    rhs.[<$trait:lower>](self)
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

// Quaternion<T> op Quaternion<T>
impl_bin_op!(impl{T: FloatType} Add<Quaternion<T>> for Quaternion<T>, Output = Quaternion<T>);
impl_bin_op!(impl{T: FloatType} Sub<Quaternion<T>> for Quaternion<T>, Output = Quaternion<T>);

// Quaternion<T> op T
impl_bin_op!(impl{T: FloatType} @scalar Add<T> for Quaternion<T>, Output = Quaternion<T>);
impl_bin_op!(impl{T: FloatType} @scalar Sub<T> for Quaternion<T>, Output = Quaternion<T>);
impl_bin_op!(impl{T: FloatType} @scalar Mul<T> for Quaternion<T>, Output = Quaternion<T>);
impl_bin_op!(impl{T: FloatType} @scalar Div<T> for Quaternion<T>, Output = Quaternion<T>);

// These two cases must be explicitly specialized for the concrete types f32 & f64 due to Rust's orphan
// rules. Trying to use generics here fails, since neither type T nor trait Add etc. are local to
// this crate.
// f32 op Quaternion<f32>
impl_bin_op!(impl{} @specialization Add<Quaternion<f32>> for f32, Output = Quaternion<f32>);
impl_bin_op!(impl{} @specialization Sub<Quaternion<f32>> for f32, Output = Quaternion<f32>);
impl_bin_op!(impl{} @specialization Mul<Quaternion<f32>> for f32, Output = Quaternion<f32>);
impl_bin_op!(impl{} @specialization Div<Quaternion<f32>> for f32, Output = Quaternion<f32>);

// f64 op Quaternion<f64>
impl_bin_op!(impl{} @specialization Add<Quaternion<f64>> for f64, Output = Quaternion<f64>);
impl_bin_op!(impl{} @specialization Sub<Quaternion<f64>> for f64, Output = Quaternion<f64>);
impl_bin_op!(impl{} @specialization Mul<Quaternion<f64>> for f64, Output = Quaternion<f64>);
impl_bin_op!(impl{} @specialization Div<Quaternion<f64>> for f64, Output = Quaternion<f64>);

#[cfg(test)]
mod tests {
    use crate::{inverse, length, normalize};

    use super::*;
    const LIMIT: f32 = 1e-6;

    #[test]
    fn new() {
        let q = Quat::new([0.0, 0.0, 0.0, 0.0]);
        assert!(q.x() == 0.0);
        assert!(q.y() == 0.0);
        assert!(q.z() == 0.0);
        assert!(q.w() == 0.0);
    }

    #[test]
    fn identity() {
        let q = Quat::identity();
        assert!(q.x() == 0.0);
        assert!(q.y() == 0.0);
        assert!(q.z() == 0.0);
        assert!(q.w() == 1.0);
    }

    #[test]
    fn length_test() {
        let q = &Quat::identity();
        assert!((length!(q) - 1.0).abs() < LIMIT);
    }

    #[test]
    fn length_sq_test() {
        let q = &Quat::new([2.0, 0.0, 0.0, 0.0]);
        assert!((length_sq!(q) - 4.0).abs() < LIMIT);
    }

    #[test]
    fn normalize_test() {
        let q = &Quat::new([2.0, 0.0, 0.0, 0.0]);
        assert!((length!(&normalize!(q)) - 1.0).abs() < LIMIT);
    }

    #[test]
    fn inverse1() {
        let q = &Quat::identity();
        let qi = inverse!(q);
        assert!(qi.x() == 0.0);
        assert!(qi.y() == 0.0);
        assert!(qi.z() == 0.0);
        assert!(qi.w() == 1.0);
    }

    #[test]
    fn inverse2() {
        let q = &Quat::new([2.0, 321.0, -10.0, 0.11777]);
        let q = normalize!(q);
        let qi = inverse!(&q);
        println!("{}", q);
        println!("{}", qi);

        // For unit quaternion, the inverse is equal to the complex conjugate
        assert!((qi.x() + q.x()).abs() < LIMIT);
        assert!((qi.y() + q.y()).abs() < LIMIT);
        assert!((qi.z() + q.z()).abs() < LIMIT);
        assert!((qi.w() - q.w()).abs() < LIMIT);
    }

    #[test]
    fn inverse3() {
        let q = &Quat::new([2.0, 321.0, -10.0, 0.11777]);
        let qi = inverse!(q);

        assert!((qi.x() * length_sq!(q) + q.x()).abs() < LIMIT);
        assert!((qi.y() * length_sq!(q) + q.y()).abs() < LIMIT);
        assert!((qi.z() * length_sq!(q) + q.z()).abs() < LIMIT);
        assert!((qi.w() * length_sq!(q) - q.w()).abs() < LIMIT);
    }

    #[test]
    fn conjugate1() {
        let q = &Quat::new([2.0, 321.0, -10.0, 0.11777]);
        let qc = conjugate(q);

        assert!((qc.x() + q.x()).abs() < LIMIT);
        assert!((qc.y() + q.y()).abs() < LIMIT);
        assert!((qc.z() + q.z()).abs() < LIMIT);
        assert!((qc.w() - q.w()).abs() < LIMIT);
    }

    #[test]
    fn mul() {
        let p = Quat::new([1.0, -2.0, 1.0, 3.0]);
        let q = Quat::new([-1.0, 2.0, 3.0, 2.0]);
        let pq = p * q;

        assert!(pq.x() + 9.0 < LIMIT);
        assert!(pq.y() + 2.0 < LIMIT);
        assert!(pq.z() - 11.0 < LIMIT);
        assert!(pq.w() - 8.0 < LIMIT);
    }

    #[test]
    fn add() {
        let p = &Quat::new([1.5, 2.5, 3.5, 1.5]);
        let q = Quat::new([2.5, 3.5, 4.5, 1.5]);
        let pq = p + q;

        assert!((pq.x() - 4.0).abs() < LIMIT);
        assert!((pq.y() - 6.0).abs() < LIMIT);
        assert!((pq.z() - 8.0).abs() < LIMIT);
        assert!((pq.w() - 3.0).abs() < LIMIT);
    }

    #[test]
    fn sub() {
        let p = &Quat::new([1.5, 2.5, 3.5, 1.5]);
        let q = Quat::new([2.5, 3.5, 4.5, 1.5]);
        let pq = p - q;

        assert!((pq.x() + 1.0).abs() < LIMIT);
        assert!((pq.y() + 1.0).abs() < LIMIT);
        assert!((pq.z() + 1.0).abs() < LIMIT);
        assert!((pq.w() + 0.0).abs() < LIMIT);
    }

    #[test]
    fn rotate1() {
        let x = Float3::new([1.0, 0.0, 0.0]);
        let angle = std::f32::consts::PI / 4.0;
        let q = Quat::new([0.0, 0.0, f32::sin(angle), f32::cos(angle)]);
        let y = q.rotate(&x);

        assert!(y.x().abs() < LIMIT);
        assert!((y.y() - 1.0).abs() < LIMIT);
        assert!(y.z().abs() < LIMIT);
    }

    #[test]
    fn rotate2() {
        let x = Float3::new([1.0, 0.0, 0.0]);
        let angle = std::f32::consts::PI / 8.0;
        let q = Quat::new([0.0, 0.0, f32::sin(angle), f32::cos(angle)]);
        let y = q.rotate(&x);
        let val = f32::sqrt(2.0) / 2.0;
        assert!((y.x() - val).abs() < LIMIT);
        assert!((y.y() - val).abs() < LIMIT);
        assert!(y.z().abs() < LIMIT);
    }

    #[test]
    fn rotate3() {
        let z = Float3::new([0.0, 0.0, 1.0]);
        let axis = Float3::new([1.0, 0.0, 0.0]);
        let angle = Angle::Radians(std::f32::consts::PI / 2.0);
        let q = Quat::from_axis_angle(&axis, angle);
        let y = q.rotate(&z);

        assert!(y.x().abs() < LIMIT);
        assert!((y.y() + 1.0).abs() < LIMIT);
        assert!(y.z().abs() < LIMIT);
    }

    #[test]
    fn rotate4() {
        let v1 = normalize!(&Float3::new([-0.132, 5.0, 1.0]));
        let axis = normalize!(&Float3::new([1.0, 1.0, -1.0]));
        let angle = Angle::Radians(std::f32::consts::PI / 4.1231);
        let q = Quat::from_axis_angle(&axis, angle);
        let qi = inverse!(&q);
        let rotated1 = q.rotate(&v1);
        let rotated2 = (q * v1 * qi).axis();
        let v2 = (qi * rotated2 * q).axis();
        let v3 = qi.rotate(&rotated1);
        let v4 = (qi * q * v1 * q * qi).axis();

        println!("{}", rotated1);
        println!("{}", rotated2);
        println!("{}", v1);
        println!("{}", v2);
        println!("{}", v3);
        println!("{}", v4);

        assert!((rotated1.x() - rotated2.x()).abs() < LIMIT);
        assert!((rotated1.y() - rotated2.y()).abs() < LIMIT);
        assert!((rotated1.z() - rotated2.z()).abs() < LIMIT);

        assert!((v1.x() - v2.x()).abs() < LIMIT);
        assert!((v1.y() - v2.y()).abs() < LIMIT);
        assert!((v1.z() - v2.z()).abs() < LIMIT);

        assert!((v1.x() - v3.x()).abs() < LIMIT);
        assert!((v1.y() - v3.y()).abs() < LIMIT);
        assert!((v1.z() - v3.z()).abs() < LIMIT);
    }

    #[test]
    fn rotate5() {
        let v1 = Float3::new([0.5, 0.5, 0.0]);
        let v2 = Float3::new([0.5, 0.0, 0.5]);
        let axis = Float3::new([1.0, 0.0, 0.0]);
        let angle = Angle::Radians(std::f32::consts::PI / 2.0);
        let q = Quat::from_axis_angle(&axis, angle);
        let v3 = q.rotate(&v1);
        let v4 = v2 - v3;

        assert!(v4.x().abs() < LIMIT);
        assert!(v4.y().abs() < LIMIT);
        assert!(v4.z().abs() < LIMIT);
    }

    #[test]
    fn rotate6() {
        let v1 = Float3::new([0.5, 0.5, 0.0]);
        let v2 = Float3::new([0.5, -0.5, 0.0]);
        let axis = Float3::new([1.0, 0.0, 0.0]);
        let angle = Angle::Radians(std::f32::consts::PI);
        let q = Quat::from_axis_angle(&axis, angle);
        let v3 = q.rotate(&v1);
        let v4 = v2 - v3;

        assert!(v4.x().abs() < LIMIT);
        assert!(v4.y().abs() < LIMIT);
        assert!(v4.z().abs() < LIMIT);
    }

    #[test]
    fn serialize_deserialize() {
        let q = Quat::new([-0.0, 1.0, -2.0, 3.0]);
        let q_str = serde_json::to_string_pretty(&q).unwrap();
        let q2: Quaternion<f32> = serde_json::from_str(&q_str).unwrap();

        assert!(q == q2);
    }

    #[test]
    fn from_to1() {
        let from = Float3::new([1.0, 0.0, 0.0]);
        let to = Float3::new([1.0, 0.0, 1.0]);
        let q = Quat::rotator_from_to(&from, &to);
        let rotated = q.rotate(&from);
        let to = normalize!(&to);
        println!("{}", to);
        println!("{}", rotated);

        assert!((to.x() - rotated.x()).abs() < LIMIT);
        assert!((to.y() - rotated.y()).abs() < LIMIT);
        assert!((to.z() - rotated.z()).abs() < LIMIT);
    }

    #[test]
    fn from_to2() {
        let from = Float3::new([1.0, 0.0, 0.0]);
        let to = Float3::new([-1.0, 0.0, 0.0]);
        let q = Quat::rotator_from_to(&from, &to);
        let rotated = q.rotate(&from);
        println!("{}", to);
        println!("{}", rotated);

        assert!((to.x() - rotated.x()).abs() < LIMIT);
        assert!((to.y() - rotated.y()).abs() < LIMIT);
        assert!((to.z() - rotated.z()).abs() < LIMIT);
    }

    #[test]
    #[should_panic(
        expected = "There is no quaternion that can convert a vector to the zero vector"
    )]
    fn from_zero_vector_to_some_invalid() {
        let from = Float3::new([0.0, 0.0, 0.0]);
        let to = Float3::new([-1.0, 0.0, 0.0]);
        let _q = Quat::rotator_from_to(&from, &to);
    }

    #[test]
    fn from_to3() {
        let from = Float3::new([1.0, 0.0, 0.0]);
        let to = Float3::new([1.0, 0.0, 0.0]);
        let q = Quat::rotator_from_to(&from, &to);
        let rotated = q.rotate(&from);
        println!("{}", to);
        println!("{}", rotated);

        assert!((to.x() - rotated.x()).abs() < LIMIT);
        assert!((to.y() - rotated.y()).abs() < LIMIT);
        assert!((to.z() - rotated.z()).abs() < LIMIT);
    }

    #[test]
    fn from_to4() {
        let from = Float3::new([1.0, 0.0, 0.0]);
        let to = Float3::new([0.0, 1.0, 0.0]);
        let q = Quat::rotator_from_to(&from, &to);
        let rotated = q.rotate(&from);
        println!("{}", to);
        println!("{}", rotated);

        assert!((to.x() - rotated.x()).abs() < LIMIT);
        assert!((to.y() - rotated.y()).abs() < LIMIT);
        assert!((to.z() - rotated.z()).abs() < LIMIT);
    }

    #[test]
    fn from_to_axis_angle() {
        let axis = normalize!(&Float3::new([1.0, 1.0, 0.0]));
        let angle = Angle::Radians(std::f32::consts::PI);
        let q = Quat::from_axis_angle(&axis, angle);

        let diff = abs(&(axis - q.axis()));
        assert!(diff.x() < LIMIT);
        assert!(diff.y() < LIMIT);
        assert!(diff.z() < LIMIT);
        assert!((angle - q.angle()).inner().abs() < LIMIT);
    }
}
