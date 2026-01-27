use num::traits::FloatConst;
use serde::{de::DeserializeOwned, Serialize};
use std::{
    fmt::{Debug, Display},
    ops::{AddAssign, BitAndAssign, BitOrAssign, BitXorAssign, DivAssign, MulAssign, SubAssign},
};

pub mod angle;
pub mod matrix;
pub mod quaternion;
pub mod vector;

// ========================================================================
// Type defs
// ========================================================================
// Change these to change defaults used in the math mods
pub type DefaultFloat = f32;
pub type DefaultInt = i32;
pub type DefaultUint = u32;

// ========================================================================
// Traits
// ========================================================================
// Traits that can be used with Vector/Quaternion/Matrix generic types:
// e.g. Vector<T> where T: FloatType
pub trait ArithmeticAssign: AddAssign + SubAssign + MulAssign + DivAssign + Sized {}
impl<T> ArithmeticAssign for T where T: AddAssign + SubAssign + MulAssign + DivAssign + Sized {}

// Use this for all common vector types that don't require knowledge of the underlying data type
pub trait VectorizedType:
    Display + Debug + Default + Copy + Clone + PartialEq + Serialize + DeserializeOwned
{
}
impl<T> VectorizedType for T where
    T: Display + Debug + Default + Copy + Clone + PartialEq + Serialize + DeserializeOwned
{
}

// Use this for all vector types that use +-/* operations
pub trait ArithmeticType:
    VectorizedType + ArithmeticAssign + num::Num + PartialOrd + std::iter::Sum
{
    #[inline]
    fn two() -> Self {
        Self::one() + Self::one()
    }

    #[inline]
    fn three() -> Self {
        Self::one() + Self::two()
    }

    #[inline]
    fn four() -> Self {
        Self::two() + Self::two()
    }

    #[inline]
    fn six() -> Self {
        Self::two() + Self::four()
    }

    #[inline]
    fn eight() -> Self {
        Self::four() + Self::four()
    }

    #[inline]
    fn ten() -> Self {
        Self::two() + Self::eight()
    }

    #[inline]
    fn one_eighty() -> Self {
        Self::three() * Self::six() * Self::ten()
    }

    #[inline]
    fn three_sixty() -> Self {
        Self::six() * Self::six() * Self::ten()
    }
}
impl<T> ArithmeticType for T where
    T: VectorizedType + ArithmeticAssign + num::Num + PartialOrd + std::iter::Sum
{
}

// Use this for all vector types that have -+ signs
pub trait SignedType: ArithmeticType + num::Signed {}
impl<T> SignedType for T where T: ArithmeticType + num::Signed {}

// Use this for f32, f64 etc.
pub trait FloatType: SignedType + num::Float + num::traits::FloatConst {
    #[inline]
    fn two_pi() -> Self {
        Self::two() * FloatConst::PI()
    }

    #[inline]
    fn half() -> Self {
        Self::from(0.5).unwrap()
    }
}
impl<T> FloatType for T where T: SignedType + num::Float + num::traits::FloatConst {}

// Use this for i32, i64 etc.
pub trait IntType: SignedType + num::Integer {}
impl<T> IntType for T where T: SignedType + num::Integer {}

// Use this for u32, u64 etc.
pub trait UintType: ArithmeticType + num::Integer {}
impl<T> UintType for T where T: ArithmeticType + num::Integer {}

// Helper traits/macros for some common math functions
pub trait Inverse {
    fn inverse(&self) -> Self;
}

macro_rules! inverse {
    ($($tokens:tt)*) => {
        Inverse::inverse($($tokens)*)
    };
}
pub(crate) use inverse;

pub trait LengthSq {
    type Output;
    fn length_sq(&self) -> Self::Output;
}
macro_rules! length_sq {
    ($($tokens:tt)*) => {
        LengthSq::length_sq($($tokens)*)
    };
}
pub(crate) use length_sq;

pub trait Length {
    type Output;
    fn length(&self) -> Self::Output;
}
macro_rules! length {
    ($($tokens:tt)*) => {
        Length::length($($tokens)*)
    };
}
pub(crate) use length;

pub trait Normalize {
    fn normalize(&self) -> Self;
}
macro_rules! normalize {
    ($($tokens:tt)*) => {
        Normalize::normalize($($tokens)*)
    };
}
pub(crate) use normalize;

// ========================================================================
// Macros
// ========================================================================
// Implements binary operators:
// based on "&T op &U"
// T op U
// T op &U
// T op &mut U
// &T op U
// &T op &mut U
// &mut T op U
// &mut T op &U
// &mut T op &mut U
macro_rules! impl_bin_op_permutations {
    (impl{$($tokens:tt)*}
     $trait:ident<$trait_type:ty>
     for $for_type:ty,
     Output = $output_type:ty) => {
        paste!(
            // T op U
            impl<$($tokens)*> $trait<$trait_type> for $for_type {
                type Output = $output_type;
                fn [<$trait:lower>](self, rhs: $trait_type) -> Self::Output {
                    (&self).[<$trait:lower>](&rhs)
                }
            }

            // T op &U
            impl<'b, $($tokens)*> $trait<&'b $trait_type> for $for_type {
                type Output = $output_type;
                fn [<$trait:lower>](self, rhs: &'b $trait_type) -> Self::Output {
                    (&self).[<$trait:lower>](rhs)
                }
            }

            // T op &mut U
            impl<'b, $($tokens)*> $trait<&'b mut $trait_type> for $for_type {
                type Output = $output_type;
                fn [<$trait:lower>](self, rhs: &'b mut $trait_type) -> Self::Output {
                    (&self).[<$trait:lower>](&*rhs)
                }
            }

            // &T op U
            impl<$($tokens)*> $trait<$trait_type> for &$for_type {
                type Output = $output_type;
                fn [<$trait:lower>](self, rhs: $trait_type) -> Self::Output {
                    self.[<$trait:lower>](&rhs)
                }
            }

            // &T op &mut U
            impl<'b, $($tokens)*> $trait<&'b mut $trait_type> for &$for_type {
                type Output = $output_type;
                fn [<$trait:lower>](self, rhs: &'b mut $trait_type) -> Self::Output {
                    self.[<$trait:lower>](&*rhs)
                }
            }

            // &mut T op U
            impl<$($tokens)*> $trait<$trait_type> for &mut $for_type {
                type Output = $output_type;
                fn [<$trait:lower>](self, rhs: $trait_type) -> Self::Output {
                    (&*self).[<$trait:lower>](&rhs)
                }
            }

            // &mut T op &U
            impl<'b, $($tokens)*> $trait<&'b $trait_type> for &'b mut $for_type {
                type Output = $output_type;
                fn [<$trait:lower>](self, rhs: &'b $trait_type) -> Self::Output {
                    (&*self).[<$trait:lower>](rhs)
                }
            }

            // &mut T op &mut U
            impl<'b, $($tokens)*> $trait<&'b mut $trait_type> for &'b mut $for_type {
                type Output = $output_type;
                fn [<$trait:lower>](self, rhs: &'b mut $trait_type) -> Self::Output {
                    (&*self).[<$trait:lower>](&*rhs)
                }
            }
        );
    };
}

// Implements assign operators:
// based on "&mut T op &U"
// &mut T op U
// &mut T op &mut U
macro_rules! impl_assign_op_permutations {
    (impl{$($tokens:tt)*}
     $trait:ident<$trait_type:ty>
     for $for_type:ty) => {
        paste!(
            // &mut T op U
            impl<$($tokens)*> [<$trait Assign>]<$trait_type> for $for_type {
                fn [<$trait:lower _assign>](&mut self, rhs: $trait_type) {
                    self.[<$trait:lower _assign>](&rhs)
                }
            }

            // &mut T op &mut U
            impl<'b, $($tokens)*> [<$trait Assign>]<&'b mut $trait_type> for $for_type {
                fn [<$trait:lower _assign>](&mut self, rhs: &'b mut $trait_type) {
                    self.[<$trait:lower _assign>](&*rhs)
                }
            }
        );
    };
}
pub(crate) use impl_assign_op_permutations;
pub(crate) use impl_bin_op_permutations;
