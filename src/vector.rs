#![allow(dead_code)]

use crate::*;
use paste::paste;
use serde::{Deserialize, Serialize};
use serde_with::serde_as;
use std::{
    array::IntoIter,
    fmt::Display,
    iter::{IntoIterator, Iterator},
    ops::{
        Add, AddAssign, BitAnd, BitOr, BitXor, Div, DivAssign, Index, IndexMut, Mul, MulAssign,
        Neg, Not, Sub, SubAssign,
    },
    slice::{Iter, IterMut},
};

pub type Float2 = Vector<DefaultFloat, 2>;
pub type Float3 = Vector<DefaultFloat, 3>;
pub type Float4 = Vector<DefaultFloat, 4>;
pub type Float2_32 = Vector<f32, 2>;
pub type Float3_32 = Vector<f32, 3>;
pub type Float4_32 = Vector<f32, 4>;
pub type Float2_64 = Vector<f64, 2>;
pub type Float3_64 = Vector<f64, 3>;
pub type Float4_64 = Vector<f64, 4>;

pub type Int2 = Vector<DefaultInt, 2>;
pub type Int3 = Vector<DefaultInt, 3>;
pub type Int4 = Vector<DefaultInt, 4>;
pub type Int2_32 = Vector<i32, 2>;
pub type Int3_32 = Vector<i32, 3>;
pub type Int4_32 = Vector<i32, 4>;
pub type Int2_64 = Vector<i64, 2>;
pub type Int3_64 = Vector<i64, 3>;
pub type Int4_64 = Vector<i64, 4>;

pub type Uint2 = Vector<DefaultUint, 2>;
pub type Uint3 = Vector<DefaultUint, 3>;
pub type Uint4 = Vector<DefaultUint, 4>;
pub type Uint2_32 = Vector<u32, 2>;
pub type Uint3_32 = Vector<u32, 3>;
pub type Uint4_32 = Vector<u32, 4>;
pub type Uint2_64 = Vector<u64, 2>;
pub type Uint3_64 = Vector<u64, 3>;
pub type Uint4_64 = Vector<u64, 4>;

pub type Bool2 = Vector<bool, 2>;
pub type Bool3 = Vector<bool, 3>;
pub type Bool4 = Vector<bool, 4>;

#[serde_as]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Vector<T, const N: usize>
where
    T: VectorizedType,
{
    #[serde_as(as = "[_; N]")]
    data: [T; N],
}

macro_rules! swizzle_permutations {
    // Generate component swizzle function permutations:
    // xx(), xy(), ... ww() (total 16 permutations)
    // xxx(), ... www()     (total 64 permutations)
    // xxxx(), ... wwww()   (total 256 permutations)
    ($($components:tt),*) => {
        // two components
        swizzle_permutations!($($components),*; $($components),*);
        // three components
        swizzle_permutations!(
            $($components),*;
            $($components),*;
            $($components),*
       );
        // four components
        swizzle_permutations!(
            $($components),*;
            $($components),*;
            $($components),*;
            $($components),*
        );
    };
    // Base case for two components
    ($h1:tt; $h2:tt) => {
        paste!(
            pub fn [<$h1 $h2>](&self) -> Vector<T, 2> {
                Vector::<T, 2> {
                    data: [
                        self.[<$h1>](),
                        self.[<$h2>](),
                    ],
                }
            }
        );
    };
    ($h1:tt; $h2:tt, $($t2:tt),*) => {
        swizzle_permutations!($h1; $h2);
        swizzle_permutations!($h1; $($t2),*);
    };
    ($h1:tt, $($t1:tt),*; $($h2:tt),*) => {
        swizzle_permutations!($h1; $($h2),*);
        swizzle_permutations!($($t1),*; $($h2),*);
    };
    // Base case for three components
    ($h1:tt; $h2:tt; $h3:tt) => {
        paste!(
            pub fn [<$h1 $h2 $h3>](&self) -> Vector<T, 3> {
                Vector::<T, 3> {
                    data: [
                        self.[<$h1>](),
                        self.[<$h2>](),
                        self.[<$h3>](),
                    ],
                }
            }
        );
    };
    ($h1:tt; $h2:tt; $h3:tt, $($t3:tt),*) => {
        swizzle_permutations!($h1; $h2; $h3);
        swizzle_permutations!($h1; $h2; $($t3),*);
    };
    ($h1:tt; $h2:tt, $($t2:tt),*; $($h3:tt),*) => {
        swizzle_permutations!($h1; $h2; $($h3),*);
        swizzle_permutations!($h1; $($t2),*; $($h3),*);
    };
    ($h1:tt, $($t1:tt),*; $($h2:tt),*; $($h3:tt),*) => {
        swizzle_permutations!($h1; $($h2),*; $($h3),*);
        swizzle_permutations!($($t1),*; $($h2),*; $($h3),*);
    };
    // Base case for four components
    ($h1:tt; $h2:tt; $h3:tt; $h4:tt) => {
        paste!(
            pub fn [<$h1 $h2 $h3 $h4>](&self) -> Vector<T, 4> {
                Vector::<T, 4> {
                    data: [
                        self.[<$h1>](),
                        self.[<$h2>](),
                        self.[<$h3>](),
                        self.[<$h4>](),
                    ],
                }
            }
        );
    };
    ($h1:tt; $h2:tt; $h3:tt; $h4:tt, $($t4:tt),*) => {
        swizzle_permutations!($h1; $h2; $h3; $h4);
        swizzle_permutations!($h1; $h2; $h3; $($t4),*);
    };
    ($h1:tt; $h2:tt; $h3:tt, $($t3:tt),*; $($h4:tt),*) => {
        swizzle_permutations!($h1; $h2; $h3; $($h4),*);
        swizzle_permutations!($h1; $h2; $($t3),*; $($h4),*);
    };
    ($h1:tt; $h2:tt, $($t2:tt),*; $($h3:tt),*; $($h4:tt),*) => {
        swizzle_permutations!($h1; $h2; $($h3),*; $($h4),*);
        swizzle_permutations!($h1; $($t2),*; $($h3),*; $($h4),*);
    };
    ($h1:tt, $($t1:tt),*; $($h2:tt),*; $($h3:tt),*; $($h4:tt),*) => {
        swizzle_permutations!($h1; $($h2),*; $($h3),*; $($h4),*);
        swizzle_permutations!($($t1),*; $($h2),*; $($h3),*; $($h4),*);
    };
}

impl<T, const N: usize> Vector<T, N>
where
    T: VectorizedType,
{
    pub const fn new(data: [T; N]) -> Self {
        Vector { data }
    }

    pub fn x(&self) -> T {
        self[0]
    }

    pub fn y(&self) -> T {
        self[1]
    }

    pub fn z(&self) -> T {
        if N > 2 {
            self[2]
        } else {
            T::default()
        }
    }

    pub fn w(&self) -> T {
        if N > 3 {
            self[3]
        } else {
            T::default()
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.data.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.data.iter_mut()
    }

    pub const fn dim(&self) -> usize {
        N
    }

    // xx(), xy(), .. etc. permutations up to wwww()
    swizzle_permutations!(x, y, z, w);
}

impl<T, const N: usize> Vector<T, N>
where
    T: ArithmeticType,
{
    pub fn zero() -> Self {
        Vector {
            data: [T::zero(); N],
        }
    }

    pub fn one() -> Self {
        Vector {
            data: [T::one(); N],
        }
    }

    pub fn x_axis() -> Vector<T, 3> {
        Vector::<T, 3>::new([T::one(), T::zero(), T::zero()])
    }

    pub fn y_axis() -> Vector<T, 3> {
        Vector::<T, 3>::new([T::zero(), T::one(), T::zero()])
    }

    pub fn z_axis() -> Vector<T, 3> {
        Vector::<T, 3>::new([T::zero(), T::zero(), T::one()])
    }

    pub fn product(&self) -> T {
        let mut total = T::one();
        for i in self.data {
            total *= i;
        }
        total
    }

    pub fn sum(&self) -> T {
        dot(self, &Self::one())
    }

    // Should point be its own structure with only three componentsand
    // and an implicit one during matrix multiplication?
    pub fn to_point4(&self) -> Vector<T, 4> {
        Vector::<T, 4>::new([self.x(), self.y(), self.z(), T::one()])
    }
}

// ===================================================
// Free functions
// ===================================================
pub fn dot<T, const N: usize>(lhs: &Vector<T, N>, rhs: &Vector<T, N>) -> T
where
    T: ArithmeticType,
{
    lhs.iter().zip(rhs.iter()).map(|(&x, &y)| x * y).sum()
}

pub fn cross<T>(lhs: &Vector<T, 3>, rhs: &Vector<T, 3>) -> Vector<T, 3>
where
    T: ArithmeticType,
{
    let x = lhs[1] * rhs[2] - lhs[2] * rhs[1];
    let y = lhs[2] * rhs[0] - lhs[0] * rhs[2];
    let z = lhs[0] * rhs[1] - lhs[1] * rhs[0];
    Vector::<T, 3>::new([x, y, z])
}

pub fn min<T, const N: usize>(lhs: &Vector<T, N>, rhs: &Vector<T, N>) -> Vector<T, N>
where
    T: ArithmeticType,
{
    // Component-wise minimum
    lhs.iter()
        .zip(rhs.iter())
        .map(|(&l, &r)| if l < r { l } else { r })
        .collect()
}

pub fn max<T, const N: usize>(lhs: &Vector<T, N>, rhs: &Vector<T, N>) -> Vector<T, N>
where
    T: ArithmeticType,
{
    // Component-wise maximum
    lhs.iter()
        .zip(rhs.iter())
        .map(|(&l, &r)| if l > r { l } else { r })
        .collect()
}

pub fn ltcw<T, const N: usize>(lhs: &Vector<T, N>, rhs: &Vector<T, N>) -> Vector<bool, N>
where
    T: ArithmeticType,
{
    // Component-wise <
    lhs.iter().zip(rhs.iter()).map(|(&l, &r)| l < r).collect()
}

pub fn lecw<T, const N: usize>(lhs: &Vector<T, N>, rhs: &Vector<T, N>) -> Vector<bool, N>
where
    T: ArithmeticType,
{
    // Component-wise <=
    lhs.iter().zip(rhs.iter()).map(|(&l, &r)| l <= r).collect()
}

pub fn gtcw<T, const N: usize>(lhs: &Vector<T, N>, rhs: &Vector<T, N>) -> Vector<bool, N>
where
    T: ArithmeticType,
{
    ltcw(rhs, lhs)
}

pub fn gecw<T, const N: usize>(lhs: &Vector<T, N>, rhs: &Vector<T, N>) -> Vector<bool, N>
where
    T: ArithmeticType,
{
    lecw(rhs, lhs)
}

pub fn eqcw<T, const N: usize>(lhs: &Vector<T, N>, rhs: &Vector<T, N>) -> Vector<bool, N>
where
    T: ArithmeticType,
{
    // Component-wise ==
    lhs.iter().zip(rhs.iter()).map(|(&l, &r)| l == r).collect()
}

pub fn necw<T, const N: usize>(lhs: &Vector<T, N>, rhs: &Vector<T, N>) -> Vector<bool, N>
where
    T: ArithmeticType,
{
    !eqcw(rhs, lhs)
}

pub fn abs<T, const N: usize>(lhs: &Vector<T, N>) -> Vector<T, N>
where
    T: SignedType,
{
    lhs.iter().map(|&x| T::abs(&x)).collect()
}

pub fn all<const N: usize>(lhs: &Vector<bool, N>) -> bool {
    lhs.iter().all(|&v| v)
}

pub fn any<const N: usize>(lhs: &Vector<bool, N>) -> bool {
    lhs.iter().any(|&v| v)
}

pub fn clamp<T, const N: usize>(
    lhs: &Vector<T, N>,
    min: &Vector<T, N>,
    max: &Vector<T, N>,
) -> Vector<T, N>
where
    T: ArithmeticType,
{
    lhs.iter()
        .zip(min.iter())
        .zip(max.iter())
        .map(|((&x, &min), &max)| num::clamp(x, min, max))
        .collect()
}

// ===================================================
// Trait implementations for Vector
// ===================================================
impl<T, const N: usize> LengthSq for Vector<T, N>
where
    T: ArithmeticType,
{
    type Output = T;
    fn length_sq(&self) -> Self::Output {
        dot(self, self)
    }
}

impl<T, const N: usize> Length for Vector<T, N>
where
    T: FloatType,
{
    type Output = T;
    fn length(&self) -> Self::Output {
        T::sqrt(length_sq!(self))
    }
}

impl<T, const N: usize> Normalize for Vector<T, N>
where
    T: FloatType,
{
    fn normalize(&self) -> Self {
        // Avoid NaNs by checking if length is zero
        let l = length_sq!(self);
        if l == T::zero() {
            *self
        } else {
            *self / T::sqrt(l)
        }
    }
}

impl<T, const N: usize> Display for Vector<T, N>
where
    T: VectorizedType,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "(")?;
        for value in self.into_iter().take(N - 1) {
            write!(f, "{}, ", value)?;
        }
        write!(f, "{})", self[N - 1])?;
        Ok(())
    }
}

impl<T, const N: usize> Default for Vector<T, N>
where
    T: VectorizedType,
{
    fn default() -> Self {
        Self {
            data: [T::default(); N],
        }
    }
}

impl<T, const N: usize> Neg for &Vector<T, N>
where
    T: SignedType,
{
    type Output = Vector<T, N>;
    fn neg(self) -> Self::Output {
        self.iter().map(|&x| -x).collect()
    }
}

impl<T, const N: usize> Neg for &mut Vector<T, N>
where
    T: SignedType,
{
    type Output = Vector<T, N>;
    fn neg(self) -> Self::Output {
        -(&*self)
    }
}

impl<T, const N: usize> Neg for Vector<T, N>
where
    T: SignedType,
{
    type Output = Vector<T, N>;
    fn neg(self) -> Self::Output {
        -(&self)
    }
}

impl<const N: usize> Not for &Vector<bool, N> {
    type Output = Vector<bool, N>;
    fn not(self) -> Self::Output {
        self.iter().map(|&x| !x).collect()
    }
}

impl<const N: usize> Not for Vector<bool, N> {
    type Output = Vector<bool, N>;
    fn not(self) -> Self::Output {
        !(&self)
    }
}

impl<const N: usize> Not for &mut Vector<bool, N> {
    type Output = Vector<bool, N>;
    fn not(self) -> Self::Output {
        !(&*self)
    }
}

// These are implemented instead of From and Into. This is because Rust doesn't have specialization
// yet, and Rust's core already implements From<T> for T. This means we cannot implement
// From<Vector<T, N>> for Vector<T, M> using generics, because it's possible N == M and we cannot
// restrict that. To have a single api for changing different sized vectors to others for values,
// refs and mut refs, we're using this and not implementing From for the ref vectors either.
pub trait IntoVec<T> {
    fn into_vec(self) -> T;
}

impl<U, T> IntoVec<U> for T
where
    U: FromVec<T>,
{
    fn into_vec(self) -> U {
        U::from_vec(self)
    }
}

pub trait FromVec<T> {
    fn from_vec(value: T) -> Self;
}

impl<T, const N: usize, const M: usize> FromVec<&Vector<T, N>> for Vector<T, M>
where
    T: VectorizedType,
{
    fn from_vec(value: &Vector<T, N>) -> Self {
        value.iter().collect()
    }
}

impl<T, const N: usize, const M: usize> FromVec<&mut Vector<T, N>> for Vector<T, M>
where
    T: VectorizedType,
{
    fn from_vec(value: &mut Vector<T, N>) -> Self {
        Self::from_vec(&*value)
    }
}

impl<T, const N: usize, const M: usize> FromVec<Vector<T, N>> for Vector<T, M>
where
    T: VectorizedType,
{
    fn from_vec(value: Vector<T, N>) -> Self {
        Self::from_vec(&value)
    }
}

impl<T, const N: usize> From<&[T; N]> for Vector<T, N>
where
    T: VectorizedType,
{
    fn from(data: &[T; N]) -> Self {
        Self { data: *data }
    }
}

impl<T, const N: usize> From<[T; N]> for Vector<T, N>
where
    T: VectorizedType,
{
    fn from(data: [T; N]) -> Self {
        Vector::<T, N>::from(&data)
    }
}

impl<T, const N: usize> From<&mut [T; N]> for Vector<T, N>
where
    T: VectorizedType,
{
    fn from(data: &mut [T; N]) -> Self {
        Vector::<T, N>::from(&*data)
    }
}

impl<T, const N: usize> From<&Vector<T, N>> for [T; N]
where
    T: VectorizedType,
{
    fn from(vec: &Vector<T, N>) -> Self {
        vec.data
    }
}

impl<T, const N: usize> From<Vector<T, N>> for [T; N]
where
    T: VectorizedType,
{
    fn from(vec: Vector<T, N>) -> Self {
        <[T; N]>::from(&vec)
    }
}

impl<T, const N: usize> From<&mut Vector<T, N>> for [T; N]
where
    T: VectorizedType,
{
    fn from(vec: &mut Vector<T, N>) -> Self {
        <[T; N]>::from(&*vec)
    }
}

impl<T, const N: usize> Index<usize> for Vector<T, N>
where
    T: VectorizedType,
{
    type Output = T;
    fn index(&self, i: usize) -> &Self::Output {
        &self.data[i]
    }
}

impl<T, const N: usize> IndexMut<usize> for Vector<T, N>
where
    T: VectorizedType,
{
    fn index_mut(&mut self, i: usize) -> &mut Self::Output {
        &mut self.data[i]
    }
}

impl<T, const N: usize> FromIterator<T> for Vector<T, N>
where
    T: VectorizedType,
{
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        let mut vec = Self::default();
        for (v, i) in vec.iter_mut().zip(iter.into_iter().take(N)) {
            *v = i;
        }
        vec
    }
}

impl<'a, T, const N: usize> FromIterator<&'a T> for Vector<T, N>
where
    T: VectorizedType + 'a,
{
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = &'a T>,
    {
        Self::from_iter(iter.into_iter().copied())
    }
}

impl<'a, T, const N: usize> FromIterator<&'a mut T> for Vector<T, N>
where
    T: VectorizedType + 'a,
{
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = &'a mut T>,
    {
        Self::from_iter(iter.into_iter().map(|&mut x| x))
    }
}

impl<T, const N: usize> IntoIterator for Vector<T, N>
where
    T: VectorizedType,
{
    type Item = T;
    type IntoIter = IntoIter<T, N>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

impl<'a, T, const N: usize> IntoIterator for &'a Vector<T, N>
where
    T: VectorizedType,
{
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

impl<'a, T, const N: usize> IntoIterator for &'a mut Vector<T, N>
where
    T: VectorizedType,
{
    type Item = &'a mut T;
    type IntoIter = IterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.iter_mut()
    }
}

// ========================================
// Binary operations
// ========================================
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
                    self
                        .iter()
                        .zip(rhs.iter())
                        .map(|(&x, &y)| x.[<$trait:lower>](y))
                        .collect()
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
                    self
                        .iter()
                        .zip(std::iter::repeat(rhs).take(self.dim()))
                        .map(|(&x, &y)| x.[<$trait:lower>](y))
                        .collect()
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
                    std::iter::repeat(self)
                    .take(rhs.dim())
                    .zip(rhs)
                    .map(|(&x, &y)| x.[<$trait:lower>](y))
                    .collect()
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

// Vector<T, N> op Vector<T, N>
impl_bin_op!(impl{const N: usize, T: ArithmeticType} Add<Vector<T, N>> for Vector<T, N>, Output = Vector<T, N>);
impl_bin_op!(impl{const N: usize, T: ArithmeticType} Sub<Vector<T, N>> for Vector<T, N>, Output = Vector<T, N>);
impl_bin_op!(impl{const N: usize, T: ArithmeticType} Mul<Vector<T, N>> for Vector<T, N>, Output = Vector<T, N>);
impl_bin_op!(impl{const N: usize, T: ArithmeticType} Div<Vector<T, N>> for Vector<T, N>, Output = Vector<T, N>);

// Vector<T, N> op T
impl_bin_op!(impl{const N: usize, T: ArithmeticType} @scalar Add<T> for Vector<T, N>, Output = Vector<T, N>);
impl_bin_op!(impl{const N: usize, T: ArithmeticType} @scalar Sub<T> for Vector<T, N>, Output = Vector<T, N>);
impl_bin_op!(impl{const N: usize, T: ArithmeticType} @scalar Mul<T> for Vector<T, N>, Output = Vector<T, N>);
impl_bin_op!(impl{const N: usize, T: ArithmeticType} @scalar Div<T> for Vector<T, N>, Output = Vector<T, N>);

// T op Vector<T, N>, specialized for all concrete types
impl_bin_op!(impl{const N: usize} @specialization Add<Vector<f32, N>> for f32, Output = Vector<f32, N>);
impl_bin_op!(impl{const N: usize} @specialization Sub<Vector<f32, N>> for f32, Output = Vector<f32, N>);
impl_bin_op!(impl{const N: usize} @specialization Mul<Vector<f32, N>> for f32, Output = Vector<f32, N>);
impl_bin_op!(impl{const N: usize} @specialization Div<Vector<f32, N>> for f32, Output = Vector<f32, N>);

impl_bin_op!(impl{const N: usize} @specialization Add<Vector<f64, N>> for f64, Output = Vector<f64, N>);
impl_bin_op!(impl{const N: usize} @specialization Sub<Vector<f64, N>> for f64, Output = Vector<f64, N>);
impl_bin_op!(impl{const N: usize} @specialization Mul<Vector<f64, N>> for f64, Output = Vector<f64, N>);
impl_bin_op!(impl{const N: usize} @specialization Div<Vector<f64, N>> for f64, Output = Vector<f64, N>);

impl_bin_op!(impl{const N: usize} @specialization Add<Vector<i32, N>> for i32, Output = Vector<i32, N>);
impl_bin_op!(impl{const N: usize} @specialization Sub<Vector<i32, N>> for i32, Output = Vector<i32, N>);
impl_bin_op!(impl{const N: usize} @specialization Mul<Vector<i32, N>> for i32, Output = Vector<i32, N>);
impl_bin_op!(impl{const N: usize} @specialization Div<Vector<i32, N>> for i32, Output = Vector<i32, N>);

impl_bin_op!(impl{const N: usize} @specialization Add<Vector<i64, N>> for i64, Output = Vector<i64, N>);
impl_bin_op!(impl{const N: usize} @specialization Sub<Vector<i64, N>> for i64, Output = Vector<i64, N>);
impl_bin_op!(impl{const N: usize} @specialization Mul<Vector<i64, N>> for i64, Output = Vector<i64, N>);
impl_bin_op!(impl{const N: usize} @specialization Div<Vector<i64, N>> for i64, Output = Vector<i64, N>);

impl_bin_op!(impl{const N: usize} @specialization Add<Vector<u32, N>> for u32, Output = Vector<u32, N>);
impl_bin_op!(impl{const N: usize} @specialization Sub<Vector<u32, N>> for u32, Output = Vector<u32, N>);
impl_bin_op!(impl{const N: usize} @specialization Mul<Vector<u32, N>> for u32, Output = Vector<u32, N>);
impl_bin_op!(impl{const N: usize} @specialization Div<Vector<u32, N>> for u32, Output = Vector<u32, N>);

impl_bin_op!(impl{const N: usize} @specialization Add<Vector<u64, N>> for u64, Output = Vector<u64, N>);
impl_bin_op!(impl{const N: usize} @specialization Sub<Vector<u64, N>> for u64, Output = Vector<u64, N>);
impl_bin_op!(impl{const N: usize} @specialization Mul<Vector<u64, N>> for u64, Output = Vector<u64, N>);
impl_bin_op!(impl{const N: usize} @specialization Div<Vector<u64, N>> for u64, Output = Vector<u64, N>);

// Booleans separately
impl_bin_op!(impl{const N: usize} BitOr<Vector<bool, N>> for Vector<bool, N>, Output = Vector<bool, N>);
impl_bin_op!(impl{const N: usize} BitAnd<Vector<bool, N>> for Vector<bool, N>, Output = Vector<bool, N>);
impl_bin_op!(impl{const N: usize} BitXor<Vector<bool, N>> for Vector<bool, N>, Output = Vector<bool, N>);
impl_bin_op!(impl{const N: usize} @scalar BitOr<bool> for Vector<bool, N>, Output = Vector<bool, N>);
impl_bin_op!(impl{const N: usize} @scalar BitAnd<bool> for Vector<bool, N>, Output = Vector<bool, N>);
impl_bin_op!(impl{const N: usize} @scalar BitXor<bool> for Vector<bool, N>, Output = Vector<bool, N>);
impl_bin_op!(impl{const N: usize} @specialization BitOr<Vector<bool, N>> for bool, Output = Vector<bool, N>);
impl_bin_op!(impl{const N: usize} @specialization BitAnd<Vector<bool, N>> for bool, Output = Vector<bool, N>);
impl_bin_op!(impl{const N: usize} @specialization BitXor<Vector<bool, N>> for bool, Output = Vector<bool, N>);

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(dead_code)]
    const X: Float3 = Float3::new([1.0, 0.0, 0.0]);
    #[allow(dead_code)]
    const Y: Float3 = Float3::new([0.0, 1.0, 0.0]);
    #[allow(dead_code)]
    const Z: Float3 = Float3::new([0.0, 0.0, 1.0]);
    #[allow(dead_code)]
    const XY: Float3 = Float3::new([1.0, 1.0, 0.0]);
    #[allow(dead_code)]
    const XZ: Float3 = Float3::new([1.0, 0.0, 1.0]);
    #[allow(dead_code)]
    const YZ: Float3 = Float3::new([0.0, 1.0, 1.0]);
    #[allow(dead_code)]
    const XYZ: Float3 = Float3::new([1.0, 1.0, 1.0]);

    #[test]
    fn new() {
        let v = Vector::<f32, 10>::new([0.0; 10]);
        assert!(v == Vector::<f32, 10>::zero());
    }

    #[test]
    fn length_sq1() {
        assert!(length_sq!(&X) == 1.0);
    }

    #[test]
    fn length_sq2() {
        assert!(length_sq!(&Float3::zero()) == 0.0);
    }

    #[test]
    fn length_sq3() {
        assert!(length_sq!(&Float3::one()) == 3.0);
    }

    #[test]
    fn length1() {
        let v = Vector::<f32, 10>::new([0.0; 10]);
        assert!(length!(&v) == 0.0);
    }

    #[test]
    fn length2() {
        assert!(length!(&X) == 1.0);
    }

    #[test]
    fn normalize1() {
        let v = Float3::new([2.0, 0.0, 0.0]);
        assert!(length!(&normalize!(&v)) == 1.0);
    }

    #[test]
    fn normalize2() {
        let v = Float3::new([0.0; 3]);
        assert!(length!(&normalize!(&v)) == 0.0);
    }

    #[test]
    fn cross_xy() {
        assert!(cross(&X, &Y) == Z);
    }

    #[test]
    fn cross_yz() {
        assert!(cross(&Y, &Z) == X);
    }

    #[test]
    fn cross_xz() {
        assert!(cross(&X, &Z) == -Y);
    }

    #[test]
    fn cross_xzz() {
        assert!(cross(&cross(&X, &Z), &Z) == -X);
    }

    #[test]
    fn cross_arbitrary() {
        // Some arbitary vectors, tested with wolframalpha
        let v1 = Float3::new([17.555, 0.123, -0.366]);
        let mut v2 = Float3::new([1.0, -12.0, 0.68]);
        let v3 = cross(&v1, &mut v2);
        let ref_v3 = Float3::new([-4.30836, -12.3034, -210.783]);
        assert!(v3 == ref_v3);
    }

    #[test]
    fn dot_xy() {
        assert!(dot(&X, &Y) == 0.0);
    }

    #[test]
    fn dot_xz() {
        assert!(dot(&X, &Z) == 0.0);
    }

    #[test]
    fn dot_yz() {
        assert!(dot(&Y, &Z) == 0.0);
    }

    #[test]
    fn dot_xx() {
        assert!(dot(&X, &X) == 1.0);
    }

    #[test]
    fn negative() {
        assert!(-X == Float3::new([-1.0, 0.0, 0.0]));
    }

    #[test]
    fn add_vec1() {
        assert!(X + Y == XY);
    }

    #[test]
    fn add_vec2() {
        assert!(X + Y + Z == XYZ);
    }

    #[test]
    fn sub_vec1() {
        assert!(XY - Y == X);
    }

    #[test]
    fn sub_vec2() {
        assert!(Y - Y == Float3::new([0.0; 3]));
    }

    #[test]
    fn div_vec1() {
        assert!((XYZ + XYZ) / Float3::new([2.0, 4.0, 6.0]) == Float3::new([1.0, 0.5, 1.0 / 3.0]));
    }

    #[test]
    fn mul_vec1() {
        assert!(XYZ * XYZ == XYZ);
    }

    #[test]
    fn mul_vec2() {
        assert!(XYZ * -XYZ == -XYZ);
    }

    #[test]
    fn mul_vec3() {
        assert!(XYZ * X == X);
    }

    #[test]
    fn add_scalar() {
        assert!(Float3::new([0.0; 3]) + 1.0 == XYZ);
    }

    #[test]
    fn sub_scalar() {
        assert!(Float3::new([0.0; 3]) - 1.0 == -XYZ);
    }

    #[test]
    fn sub_scalar2() {
        assert!(1.0 - Float3::new([0.0; 3]) == XYZ);
    }

    #[test]
    fn div_scalar() {
        assert!((XYZ + XYZ) / 2.0 == XYZ);
    }

    #[test]
    fn div_scalar2() {
        assert!(2.0 / XYZ == XYZ + XYZ);
    }

    #[test]
    fn mul_scalar() {
        assert!((XYZ + XYZ) == 2.0 * XYZ);
    }

    #[test]
    fn serialize_deserialize() {
        let v = Float4::new([0.0, 1.0, 2.0, 3.0]);
        let v_str = serde_json::to_string_pretty(&v).unwrap();
        let v2: Float4 = serde_json::from_str(&v_str).unwrap();

        assert!(v == v2);
    }

    #[test]
    fn iterate_by_value() {
        let v = Float4::new([0.0, 1.0, 2.0, 3.0]);
        for (i, c) in v.into_iter().enumerate() {
            assert!(c == i as u32 as f32)
        }
    }

    #[test]
    fn iterate_by_ref() {
        let v = Float4::new([0.0, 1.0, 2.0, 3.0]);
        for (i, &c) in (&v).into_iter().enumerate() {
            assert!(c == i as u32 as f32)
        }
    }

    #[test]
    fn iterate_by_mut_ref() {
        let mut v = Float4::new([0.0, 1.0, 2.0, 3.0]);
        for (i, &mut c) in (&mut v).into_iter().enumerate() {
            assert!(c == i as u32 as f32)
        }
    }

    #[test]
    fn xx() {
        let v = Float2::new([0.0, 1.0]);
        let v = v.xx();
        assert!(v.x() == 0.0);
        assert!(v.y() == 0.0);
    }

    #[test]
    fn yy() {
        let v = Float4::new([0.0, 1.0, 0.0, 0.0]);
        let v: Float2 = v.yy();
        assert!(v.x() == 1.0);
        assert!(v.y() == 1.0);
    }

    #[test]
    fn yx() {
        let v = Float3::new([0.0, 1.0, 0.0]);
        let v: Float2 = v.yx();
        assert!(v.x() == 1.0);
        assert!(v.y() == 0.0);
    }

    #[test]
    fn xwy() {
        let v = Float2::new([-2.5, 1.0]);
        let v: Float3 = v.xwy();
        assert!(v.x() == -2.5);
        assert!(v.y() == 0.0);
        assert!(v.z() == 1.0);
    }

    #[test]
    fn xxxx() {
        let v = Float2::new([-2.5, 1.0]);
        let v: Float4 = v.xxxx();
        assert!(v.x() == -2.5);
        assert!(v.y() == -2.5);
        assert!(v.z() == -2.5);
        assert!(v.w() == -2.5);
    }

    #[test]
    fn product() {
        let v = Float3::new([1.0, 2.0, 3.0]);
        assert!(v.product() == 6.0);
    }

    #[test]
    fn sum() {
        let v = Float3::new([1.0, 2.0, 3.0]);
        assert!(v.sum() == 6.0);
    }

    #[test]
    fn min_test() {
        let v = Float3::new([1.0, 2.0, 3.0]);
        let w = Float3::new([1.0, 1.0, 1.0]);
        assert!(min(&v, &w) == Float3::one());
    }

    #[test]
    fn max_test() {
        let v = Float3::new([1.0, 2.0, 3.0]);
        let w = Float3::new([1.0, 1.0, 1.0]);
        assert!(max(&v, &w) == v);
    }

    #[test]
    fn abs_test() {
        let v = Float3::new([-1.0, 2.0, -3.0]);
        assert!(abs(&v) == Float3::new([1.0, 2.0, 3.0]));
    }

    #[test]
    fn new_boolean() {
        let bv = Vector::<bool, 3>::new([true; 3]);
        assert!(bv.x());
        assert!(bv.y());
        assert!(bv.z());
        assert!(!bv.w());
    }

    #[test]
    fn new_integer() {
        let uv = Vector::<u32, 3>::new([5; 3]);
        assert!(uv.x() == 5);
        assert!(uv.y() == 5);
        assert!(uv.z() == 5);
        assert!(uv.w() == 0);

        assert!(length_sq!(&(uv + uv)) == 300);
    }

    #[test]
    fn test_lt1() {
        let v1 = Uint3::new([1; 3]);
        let v2 = Uint3::new([5; 3]);
        let ltv = ltcw(&v1, &v2);

        assert!(ltv[0]);
        assert!(ltv[1]);
        assert!(ltv[2]);
    }

    #[test]
    fn test_lt2() {
        let v1 = Uint3::new([5; 3]);
        let v2 = Uint3::new([5; 3]);
        let ltv = ltcw(&v1, &v2);

        assert!(!ltv[0]);
        assert!(!ltv[1]);
        assert!(!ltv[2]);
    }

    #[test]
    fn test_le1() {
        let v1 = Int3::new([-1; 3]);
        let v2 = Int3::new([-5; 3]);
        let ltv = lecw(&v1, &v2);

        assert!(!ltv[0]);
        assert!(!ltv[1]);
        assert!(!ltv[2]);
    }

    #[test]
    fn test_le2() {
        let v1 = Int3::new([-1; 3]);
        let v2 = Int3::new([-1; 3]);
        let ltv = lecw(&v1, &v2);

        assert!(ltv[0]);
        assert!(ltv[1]);
        assert!(ltv[2]);
    }

    #[test]
    fn test_gt1() {
        let v1 = Float3::new([1.11; 3]);
        let v2 = Float3::new([5.321; 3]);
        let ltv = gtcw(&v1, &v2);

        assert!(!ltv[0]);
        assert!(!ltv[1]);
        assert!(!ltv[2]);
    }

    #[test]
    fn test_gt2() {
        let v1 = Float3::new([5.321, 2.22, 1.321]);
        let v2 = Float3::new([1.0, 2.0, 3.0]);
        let ltv = gtcw(&v1, &v2);

        assert!(ltv[0]);
        assert!(ltv[1]);
        assert!(!ltv[2]);
    }

    #[test]
    fn test_ge1() {
        let v1 = Int3::new([-1; 3]);
        let v2 = Int3::new([-5; 3]);
        let ltv = gecw(&v1, &v2);

        assert!(ltv[0]);
        assert!(ltv[1]);
        assert!(ltv[2]);
    }

    #[test]
    fn test_ge2() {
        let v1 = Int3::new([-1; 3]);
        let v2 = Int3::new([-1; 3]);
        let ltv = gecw(&v1, &v2);

        assert!(ltv[0]);
        assert!(ltv[1]);
        assert!(ltv[2]);
    }

    #[test]
    fn test_eq1() {
        let v1 = Int3::new([-1; 3]);
        let v2 = Int3::new([-1; 3]);
        let ltv = eqcw(&v1, &v2);

        assert!(ltv[0]);
        assert!(ltv[1]);
        assert!(ltv[2]);
    }

    #[test]
    fn test_eq2() {
        let v1 = Int3::new([-1, -5, 16]);
        let v2 = Int3::new([-1, -5, 15]);
        let ltv = eqcw(&v1, &v2);

        assert!(ltv[0]);
        assert!(ltv[1]);
        assert!(!ltv[2]);
    }

    #[test]
    fn test_ne1() {
        let v1 = Int3::new([-1; 3]);
        let v2 = Int3::new([-1; 3]);
        let ltv = necw(&v1, &v2);

        assert!(!ltv[0]);
        assert!(!ltv[1]);
        assert!(!ltv[2]);
    }

    #[test]
    fn test_ne2() {
        let v1 = Int3::new([-1, -5, 16]);
        let v2 = Int3::new([-1, -5, 15]);
        let ltv = necw(&v1, &v2);

        assert!(!ltv[0]);
        assert!(!ltv[1]);
        assert!(ltv[2]);
    }

    #[test]
    fn test_all1() {
        let v1 = Int3::new([-1, -5, 16]);
        let v2 = Int3::new([-1, -5, 16]);
        assert!(all(&eqcw(&v1, &v2)));
    }

    #[test]
    fn test_all2() {
        assert!(all(&Bool3::new([true; 3])));
    }

    #[test]
    fn test_all3() {
        assert!(!all(&Bool3::new([true, true, false])));
    }

    #[test]
    fn test_all4() {
        assert!(!all(&Bool3::new([true, false, false])));
    }

    #[test]
    fn test_any1() {
        let v1 = Int3::new([-1, -5, 16]);
        let v2 = Int3::new([-1, -5, 16]);
        assert!(any(&eqcw(&v1, &v2)));
    }

    #[test]
    fn test_any2() {
        assert!(any(&Bool3::new([true; 3])));
    }

    #[test]
    fn test_any3() {
        assert!(any(&Bool3::new([true, true, false])));
    }

    #[test]
    fn test_any4() {
        assert!(any(&Bool3::new([true, false, false])));
    }

    #[test]
    fn test_any5() {
        assert!(!any(&Bool3::new([false; 3])));
    }

    #[test]
    fn clamp1() {
        let v1 = Int3::new([0, 0, 0]);
        let min = Int3::new([-1, -1, -1]);
        let max = Int3::new([1, 1, 1]);

        assert!(all(&eqcw(&v1, &clamp(&v1, &min, &max))));
    }

    #[test]
    fn clamp2() {
        let v1 = Int3::new([-5, -5, -5]);
        let min = Int3::new([-1, -1, -1]);
        let max = Int3::new([1, 1, 1]);

        assert!(all(&eqcw(&min, &clamp(&v1, &min, &max))));
    }

    #[test]
    fn clamp3() {
        let v1 = Int3::new([5, 5, 5]);
        let min = Int3::new([-1, -1, -1]);
        let max = Int3::new([1, 1, 1]);

        assert!(all(&eqcw(&max, &clamp(&v1, &min, &max))));
    }

    #[test]
    fn clamp4() {
        let v1 = Int3::new([5, 0, 2]);
        let min = Int3::new([-1, -1, -10]);
        let max = Int3::new([1, 1, 2]);

        assert!(all(&eqcw(&Int3::new([1, 0, 2]), &clamp(&v1, &min, &max))));
    }

    #[test]
    fn test_from_iter1() {
        let vec = Int3::from_iter(0..5);
        assert!(vec[0] == 0);
        assert!(vec[1] == 1);
        assert!(vec[2] == 2);
    }

    #[test]
    fn test_from_iter2() {
        let values = [0, 1, 2];
        let iter = values.iter();
        let vec = Int3::from_iter(iter);
        assert!(vec[0] == 0);
        assert!(vec[1] == 1);
        assert!(vec[2] == 2);
    }

    #[test]
    fn test_from_iter3() {
        let mut values = [0, 1, 2];
        let iter = values.iter_mut();
        let vec = Int3::from_iter(iter);
        assert!(vec[0] == 0);
        assert!(vec[1] == 1);
        assert!(vec[2] == 2);
    }
}
