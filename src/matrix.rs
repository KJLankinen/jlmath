use crate::{
    angle::*, impl_assign_op_permutations, impl_bin_op_permutations, inverse, length,
    length_sq, normalize, quaternion::*, vector::*, FloatType, Inverse, Length, LengthSq,
    Normalize,
};

use paste::paste;
use serde::{Deserialize, Serialize};
use serde_with::serde_as;
use std::fmt::Display;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

pub type Mat44 = Matrix<f32, 4, 4>;
pub type Mat33 = Matrix<f32, 3, 3>;
pub type Mat22 = Matrix<f32, 2, 2>;

#[serde_as]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Matrix<T, const R: usize, const C: usize>
where
    T: FloatType,
{
    #[serde_as(as = "[[_; C]; R]")]
    data: [[T; C]; R],
}

// 2x2 matrix methods
impl<T> Matrix<T, 2, 2>
where
    T: FloatType,
{
    pub fn determinant(&self) -> T {
        Self::determinant_2x2(self[0][0], self[0][1], self[1][0], self[1][1])
    }
}

// 3x3 matrix methods
impl<T> Matrix<T, 3, 3>
where
    T: FloatType,
{
    pub fn to_quaternion(&self) -> Quaternion<T> {
        <&Matrix<T, 3, 3> as Into<Matrix<T, 4, 4>>>::into(self).to_quaternion()
    }

    pub fn determinant(&self) -> T {
        let a = self[0][0];
        let b = self[0][1];
        let c = self[0][2];
        let d = self[1][0];
        let e = self[1][1];
        let f = self[1][2];
        let g = self[2][0];
        let h = self[2][1];
        let i = self[2][2];
        Self::determinant_3x3(a, b, c, d, e, f, g, h, i)
    }

    pub fn cofactor_transpose(&self) -> Self {
        // The transpose of the cofactor matrix of self
        // Cofactor matrix: M[i][j] = determinant of N,
        // where N is M with row i and column j removed
        // In other words 9 determinants for a 3x3 matrix M
        let a = self[0][0];
        let b = self[0][1];
        let c = self[0][2];
        let d = self[1][0];
        let e = self[1][1];
        let f = self[1][2];
        let g = self[2][0];
        let h = self[2][1];
        let i = self[2][2];

        Self {
            data: [
                [
                    Self::determinant_2x2(e, f, h, i),
                    -Self::determinant_2x2(b, c, h, i),
                    Self::determinant_2x2(b, c, e, f),
                ],
                [
                    -Self::determinant_2x2(d, f, g, i),
                    Self::determinant_2x2(a, c, g, i),
                    -Self::determinant_2x2(a, c, d, f),
                ],
                [
                    Self::determinant_2x2(d, e, g, h),
                    -Self::determinant_2x2(a, b, g, h),
                    Self::determinant_2x2(a, b, d, e),
                ],
            ],
        }
    }
}

// 4x4 matrix methods
impl<T> Matrix<T, 4, 4>
where
    T: FloatType,
{
    pub fn determinant(&self) -> T {
        let a = self[0][0];
        let b = self[0][1];
        let c = self[0][2];
        let d = self[0][3];
        let e = self[1][0];
        let f = self[1][1];
        let g = self[1][2];
        let h = self[1][3];
        let i = self[2][0];
        let j = self[2][1];
        let k = self[2][2];
        let l = self[2][3];
        let m = self[3][0];
        let n = self[3][1];
        let o = self[3][2];
        let p = self[3][3];

        let inmj = i * n - m * j;
        let iomk = i * o - m * k;
        let ipml = i * p - m * l;
        let jpnl = j * p - n * l;
        let jonk = j * o - n * k;
        let kpol = k * p - o * l;

        let fgh = f * kpol - g * jpnl + h * jonk;
        let egh = e * kpol - g * ipml + h * iomk;
        let efh = e * jpnl - f * ipml + h * inmj;
        let efg = e * jonk - f * iomk + g * inmj;

        a * fgh - b * egh + c * efh - d * efg
    }

    pub fn scale(scale: &Vector<T, 3>) -> Self {
        Self {
            data: [
                [scale.x(), T::zero(), T::zero(), T::zero()],
                [T::zero(), scale.y(), T::zero(), T::zero()],
                [T::zero(), T::zero(), scale.z(), T::zero()],
                [T::zero(), T::zero(), T::zero(), T::one()],
            ],
        }
    }

    pub fn translate(translation: &Vector<T, 3>) -> Self {
        Self {
            data: [
                [T::one(), T::zero(), T::zero(), translation.x()],
                [T::zero(), T::one(), T::zero(), translation.y()],
                [T::zero(), T::zero(), T::one(), translation.z()],
                [T::zero(), T::zero(), T::zero(), T::one()],
            ],
        }
    }

    pub fn rotate(rotation: &Quaternion<T>) -> Self {
        // Quaternion doesn't have to be normalized for this
        let n = length_sq!(rotation);
        let s = if n.is_zero() {
            T::zero()
        } else {
            (T::one() + T::one()) / n
        };

        let xx = rotation.x() * rotation.x();
        let xy = rotation.x() * rotation.y();
        let xz = rotation.x() * rotation.z();
        let xw = rotation.x() * rotation.w();
        let yy = rotation.y() * rotation.y();
        let yz = rotation.y() * rotation.z();
        let yw = rotation.y() * rotation.w();
        let zz = rotation.z() * rotation.z();
        let zw = rotation.z() * rotation.w();
        Self {
            data: [
                [
                    T::one() - s * (yy + zz),
                    s * (xy - zw),
                    s * (xz + yw),
                    T::zero(),
                ],
                [
                    s * (xy + zw),
                    T::one() - s * (xx + zz),
                    s * (yz - xw),
                    T::zero(),
                ],
                [
                    s * (xz - yw),
                    s * (yz + xw),
                    T::one() - s * (xx + yy),
                    T::zero(),
                ],
                [T::zero(), T::zero(), T::zero(), T::one()],
            ],
        }
    }

    pub fn extract_trs(&self) -> (Vector<T, 3>, Quaternion<T>, Vector<T, 3>) {
        // This version only works for positive scale
        let scale = Vector::<T, 3>::new([
            length!(&self.cv::<3>(0)),
            length!(&self.cv::<3>(1)),
            length!(&self.cv::<3>(2)),
        ]);
        let translation: Vector<T, 3> = self.cv(3);

        let mut rot_mat = Self::zero();
        rot_mat.srv(0, &(self.rv::<3>(0) / scale));
        rot_mat.srv(1, &(self.rv::<3>(1) / scale));
        rot_mat.srv(2, &(self.rv::<3>(2) / scale));
        let rotation = rot_mat.to_quaternion();

        (translation, rotation, scale)
    }

    pub fn to_quaternion(&self) -> Quaternion<T> {
        // Assuming this is a rotation matrix, decompose it to a quaternion
        let (t, q) = if self[2][2] < T::zero() {
            if self[0][0] > self[1][1] {
                let t = T::one() + self[0][0] - self[1][1] - self[2][2];
                let q = Quaternion::<T>::new([
                    t,
                    self[1][0] + self[0][1],
                    self[0][2] + self[2][0],
                    self[2][1] - self[1][2],
                ]);
                (t, q)
            } else {
                let t = T::one() - self[0][0] + self[1][1] - self[2][2];
                let q = Quaternion::<T>::new([
                    self[1][0] + self[0][1],
                    t,
                    self[2][1] + self[1][2],
                    self[0][2] - self[2][0],
                ]);
                (t, q)
            }
        } else if self[0][0] < -self[1][1] {
            let t = T::one() - self[0][0] - self[1][1] + self[2][2];
            let q = Quaternion::<T>::new([
                self[0][2] + self[2][0],
                self[2][1] + self[1][2],
                t,
                self[1][0] - self[0][1],
            ]);
            (t, q)
        } else {
            let t = T::one() + self[0][0] + self[1][1] + self[2][2];
            let q = Quaternion::<T>::new([
                self[2][1] - self[1][2],
                self[0][2] - self[2][0],
                self[1][0] - self[0][1],
                t,
            ]);
            (t, q)
        };

        let half = T::one() / (T::one() + T::one());
        q * half / T::sqrt(t)
    }

    pub fn model(
        translation: &Vector<T, 3>,
        rotation: &Quaternion<T>,
        scale: &Vector<T, 3>,
    ) -> Self {
        let mut model = Self::rotate(rotation);
        model[0][0] *= scale.x();
        model[1][0] *= scale.x();
        model[2][0] *= scale.x();
        model[0][1] *= scale.y();
        model[1][1] *= scale.y();
        model[2][1] *= scale.y();
        model[0][2] *= scale.z();
        model[1][2] *= scale.z();
        model[2][2] *= scale.z();
        model[0][3] = translation.x();
        model[1][3] = translation.y();
        model[2][3] = translation.z();
        model
    }

    pub fn camera(position: &Vector<T, 3>, look_at: &Vector<T, 3>, up: &Vector<T, 3>) -> Self {
        // Make a model matrix for camera from the given data
        // Right handed coordinates, +y is up, camera looks at -z in it's own frame of reference
        let camera_z = -normalize!(&(look_at - position));
        let camera_y = normalize!(&(up - camera_z * dot(up, &camera_z)));
        let camera_x = normalize!(&cross(&camera_y, &camera_z));
        Self {
            data: [
                [camera_x.x(), camera_y.x(), camera_z.x(), position.x()],
                [camera_x.y(), camera_y.y(), camera_z.y(), position.y()],
                [camera_x.z(), camera_y.z(), camera_z.z(), position.z()],
                [T::zero(), T::zero(), T::zero(), T::one()],
            ],
        }
    }

    pub fn view(position: &Vector<T, 3>, look_at: &Vector<T, 3>, up: &Vector<T, 3>) -> Self {
        inverse!(&Self::camera(position, look_at, up))
    }

    pub fn perspective(aspect_ratio: T, vertical_fov: Angle<T>, near: T, far: T) -> Self {
        // Based on Vulkan spec:
        // rigth handed coordinates with +y down, +z towards the screen.
        // Assumes the view direction specified at view()
        let inverse_tan_fov_per_2 =
            T::one() / T::tan(T::from(0.5).unwrap() * radians(vertical_fov).inner());
        let far_per_far_minus_near = far / (far - near);

        Self {
            data: [
                [
                    inverse_tan_fov_per_2 / aspect_ratio,
                    T::zero(),
                    T::zero(),
                    T::zero(),
                ],
                [T::zero(), -inverse_tan_fov_per_2, T::zero(), T::zero()],
                [
                    T::zero(),
                    T::zero(),
                    -far_per_far_minus_near,
                    -near * far_per_far_minus_near,
                ],
                [T::zero(), T::zero(), -T::one(), T::zero()],
            ],
        }
    }

    pub fn orthographic(aspect_ratio: T, vertical_fov: Angle<T>, near: T, far: T) -> Self {
        let inverse_tan_fov_per_2 =
            T::one() / T::tan(T::from(0.5).unwrap() * radians(vertical_fov).inner());
        let inverse_far_minus_near = T::one() / (far - near);
        let inverse_near = T::one() / near;

        Self {
            data: [
                [
                    inverse_near * inverse_tan_fov_per_2 / aspect_ratio,
                    T::zero(),
                    T::zero(),
                    T::zero(),
                ],
                [
                    T::zero(),
                    -inverse_near * inverse_tan_fov_per_2,
                    T::zero(),
                    T::zero(),
                ],
                [
                    T::zero(),
                    T::zero(),
                    -inverse_far_minus_near,
                    -near * inverse_far_minus_near,
                ],
                [T::zero(), T::zero(), T::zero(), T::one()],
            ],
        }
    }

    pub fn cofactor_transpose(&self) -> Self {
        // The transpose of the cofactor matrix of self
        // Cofactor matrix: M[i][j] = determinant of N,
        // where N is M with row i and column j removed
        // In other words 16 determinants for a 4x4 matrix M
        let a = self[0][0];
        let b = self[0][1];
        let c = self[0][2];
        let d = self[0][3];
        let e = self[1][0];
        let f = self[1][1];
        let g = self[1][2];
        let h = self[1][3];
        let i = self[2][0];
        let j = self[2][1];
        let k = self[2][2];
        let l = self[2][3];
        let m = self[3][0];
        let n = self[3][1];
        let o = self[3][2];
        let p = self[3][3];

        Self {
            data: [
                [
                    Self::determinant_3x3(f, g, h, j, k, l, n, o, p),
                    -Self::determinant_3x3(b, c, d, j, k, l, n, o, p),
                    Self::determinant_3x3(b, c, d, f, g, h, n, o, p),
                    -Self::determinant_3x3(b, c, d, f, g, h, j, k, l),
                ],
                [
                    -Self::determinant_3x3(e, g, h, i, k, l, m, o, p),
                    Self::determinant_3x3(a, c, d, i, k, l, m, o, p),
                    -Self::determinant_3x3(a, c, d, e, g, h, m, o, p),
                    Self::determinant_3x3(a, c, d, e, g, h, i, k, l),
                ],
                [
                    Self::determinant_3x3(e, f, h, i, j, l, m, n, p),
                    -Self::determinant_3x3(a, b, d, i, j, l, m, n, p),
                    Self::determinant_3x3(a, b, d, e, f, h, m, n, p),
                    -Self::determinant_3x3(a, b, d, e, f, h, i, j, l),
                ],
                [
                    -Self::determinant_3x3(e, f, g, i, j, k, m, n, o),
                    Self::determinant_3x3(a, b, c, i, j, k, m, n, o),
                    -Self::determinant_3x3(a, b, c, e, f, g, m, n, o),
                    Self::determinant_3x3(a, b, c, e, f, g, i, j, k),
                ],
            ],
        }
    }
}

// Generic matrices
impl<'a, const R: usize, const C: usize, T> Matrix<T, R, C>
where
    T: FloatType + 'a,
{
    pub const fn new(data: [[T; C]; R]) -> Self {
        Self { data }
    }

    pub fn zero() -> Self {
        Self {
            data: [[T::zero(); C]; R],
        }
    }

    pub fn identity() -> Self {
        // This is relatively meaningless without a square matrix but oh well
        let mut m = Self::zero();
        for r in 0..usize::min(R, C) {
            m[r][r] = T::one();
        }
        m
    }

    pub fn transpose(&self) -> Matrix<T, C, R> {
        let mut m = Matrix::<T, C, R>::zero();
        for i in 0..R {
            for j in 0..C {
                m[j][i] = self[i][j];
            }
        }
        m
    }

    pub fn rv<const N: usize>(&self, i: usize) -> Vector<T, N> {
        let mut v = Vector::<T, N>::zero();
        for j in 0..usize::min(N, C) {
            v[j] = self[i][j];
        }
        v
    }

    pub fn srv<const N: usize>(&mut self, i: usize, v: &Vector<T, N>) {
        for j in 0..usize::min(N, C) {
            self[i][j] = v[j];
        }
    }

    pub fn cv<const N: usize>(&self, j: usize) -> Vector<T, N> {
        let mut v = Vector::<T, N>::zero();
        for i in 0..usize::min(N, R) {
            v[i] = self[i][j];
        }
        v
    }

    pub fn scv<const N: usize>(&mut self, j: usize, v: &Vector<T, N>) {
        for i in 0..usize::min(N, R) {
            self[i][j] = v[i];
        }
    }

    pub fn col(&self, j: usize) -> [T; R] {
        let mut c = [T::zero(); R];
        for i in 0..R {
            c[i] = self[i][j];
        }
        c
    }

    pub fn set_col(&mut self, j: usize, col: &[T; R]) {
        for i in 0..R {
            self[i][j] = col[i];
        }
    }

    #[inline]
    pub fn determinant_2x2(a: T, b: T, c: T, d: T) -> T {
        a * d - c * b
    }

    #[inline]
    pub fn determinant_3x3(a: T, b: T, c: T, d: T, e: T, f: T, g: T, h: T, i: T) -> T {
        a * Self::determinant_2x2(e, f, h, i) - b * Self::determinant_2x2(d, f, g, i)
            + c * Self::determinant_2x2(d, e, g, h)
    }

    pub fn iter(&'a self) -> impl Iterator<Item = &'a T> {
        self.data.iter().flatten()
    }
}

// Generic square matrices
impl<T, const N: usize> Matrix<T, N, N>
where
    T: FloatType,
{
    pub fn gj_inverse(&self) -> Self {
        let mut index_col: [usize; N] = [0; N];
        let mut index_row: [usize; N] = [0; N];
        let mut index_pivot: [usize; N] = [0; N];

        let mut matrix = *self;

        // Main loop over columns
        for i in 0..N {
            let mut row = 0;
            let mut col = 0;
            let mut big = T::zero();

            // Searching for the pivot element
            for r in 0..N {
                if index_pivot[r] != 1 {
                    for (c, ip) in index_pivot.iter().enumerate() {
                        if ip == &0 {
                            let elem = matrix[r][c].abs();
                            if elem >= big {
                                big = elem;
                                row = r;
                                col = c;
                            }
                        }
                    }
                }
            }

            index_pivot[col] += 1;

            // Exchange rows so the pivot element is on the diagonal
            if row != col {
                for c in 0..N {
                    let temp = matrix[row][c];
                    matrix[row][c] = matrix[col][c];
                    matrix[col][c] = temp;
                }
            }
            // Store the original location of the pivot element
            index_row[i] = row;
            index_col[i] = col;
            if matrix[col][col] == T::zero() {
                // Singular matrix
                return Self::zero() + T::nan();
            }

            // Divide the pivot row by the pivot element
            let pivot_inverse = T::one() / matrix[col][col];
            matrix[col][col] = T::one();
            for c in 0..N {
                matrix[col][c] = matrix[col][c] * pivot_inverse;
            }

            // Reduce other rows
            for r in 0..N {
                if r != col {
                    let temp = matrix[r][col];
                    matrix[r][col] = T::zero();
                    for c in 0..N {
                        matrix[r][c] = matrix[r][c] - matrix[col][c] * temp;
                    }
                }
            }
        }

        // Unscrambling the columns
        for i in (0..N).rev() {
            let ir = index_row[i];
            let ic = index_col[i];
            if ir != ic {
                for j in 0..N {
                    matrix[j].swap(ir, ic);
                }
            }
        }

        matrix
    }
}

//------------------------------------------------
// Trait implementations
//------------------------------------------------
impl<T> Inverse for Matrix<T, 2, 2>
where
    T: FloatType,
{
    fn inverse(&self) -> Self {
        let inv_det = T::one() / self.determinant();
        Matrix::<T, 2, 2> {
            data: [[self[1][1], -self[0][1]], [-self[1][0], self[0][0]]],
        } * inv_det
    }
}

impl<T> Inverse for Matrix<T, 3, 3>
where
    T: FloatType,
{
    fn inverse(&self) -> Self {
        // Adjugate inverse
        let inv_det = T::one() / self.determinant();
        self.cofactor_transpose() * inv_det
    }
}

impl<T> Inverse for Matrix<T, 4, 4>
where
    T: FloatType,
{
    fn inverse(&self) -> Self {
        // Adjugate inverse
        let inv_det = T::one() / self.determinant();
        self.cofactor_transpose() * inv_det
    }
}

// Rusts const generic support is bad, so cannot implement
// Inverse for Matrix<T, N, N> where N > 4, because there're
// conflicting implementations with 2, 3 and 4
macro_rules! impl_inverse {
    ($N:tt) => {
        impl<T> Inverse for Matrix<T, $N, $N>
        where
            T: FloatType,
        {
            fn inverse(&self) -> Self {
                self.gj_inverse()
            }
        }
    };
}
// Add macro call for each N that's needed
impl_inverse!(5);

impl<T> From<&Matrix<T, 4, 4>> for Matrix<T, 3, 3>
where
    T: FloatType,
{
    fn from(mat44: &Matrix<T, 4, 4>) -> Self {
        let mut m = Self::zero();
        m.srv(0, &mat44.rv::<3>(0));
        m.srv(1, &mat44.rv::<3>(1));
        m.srv(2, &mat44.rv::<3>(2));
        m
    }
}

impl<T> From<&mut Matrix<T, 4, 4>> for Matrix<T, 3, 3>
where
    T: FloatType,
{
    fn from(mat44: &mut Matrix<T, 4, 4>) -> Self {
        Self::from(&*mat44)
    }
}

impl<T> From<Matrix<T, 4, 4>> for Matrix<T, 3, 3>
where
    T: FloatType,
{
    fn from(mat44: Matrix<T, 4, 4>) -> Self {
        Self::from(&mat44)
    }
}

impl<T> From<&Matrix<T, 3, 3>> for Matrix<T, 4, 4>
where
    T: FloatType,
{
    fn from(mat33: &Matrix<T, 3, 3>) -> Self {
        let mut m = Self::zero();
        m.srv(0, &mat33.rv::<4>(0));
        m.srv(1, &mat33.rv::<4>(1));
        m.srv(2, &mat33.rv::<4>(2));
        m[3][3] = T::one();
        m
    }
}

impl<T> From<&mut Matrix<T, 3, 3>> for Matrix<T, 4, 4>
where
    T: FloatType,
{
    fn from(mat33: &mut Matrix<T, 3, 3>) -> Self {
        Self::from(&*mat33)
    }
}

impl<T> From<Matrix<T, 3, 3>> for Matrix<T, 4, 4>
where
    T: FloatType,
{
    fn from(mat33: Matrix<T, 3, 3>) -> Self {
        Self::from(&mat33)
    }
}

impl<T, const R: usize, const C: usize> Display for Matrix<T, R, C>
where
    T: FloatType,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for r in (0..R).take(R - 1) {
            for c in (0..C).take(C - 1) {
                write!(f, "{}, ", self[r][c])?;
            }
            writeln!(f, "{}", self[r][C - 1])?;
        }

        let r = R - 1;
        for c in (0..C).take(C - 1) {
            write!(f, "{}, ", self[r][c])?;
        }
        write!(f, "{}", self[r][C - 1])?;
        Ok(())
    }
}

impl<T, const R: usize, const C: usize> Default for Matrix<T, R, C>
where
    T: FloatType,
{
    fn default() -> Self {
        Self::zero()
    }
}

impl<T, const R: usize, const C: usize> Neg for &Matrix<T, R, C>
where
    T: FloatType,
{
    type Output = Matrix<T, R, C>;
    fn neg(self) -> Self::Output {
        let mut out_mat = Self::Output::zero();
        for r in 0..R {
            for c in 0..C {
                out_mat[r][c] = -self[r][c];
            }
        }
        out_mat
    }
}

impl<T, const R: usize, const C: usize> Neg for &mut Matrix<T, R, C>
where
    T: FloatType,
{
    type Output = Matrix<T, R, C>;
    fn neg(self) -> Self::Output {
        (&*self).neg()
    }
}

impl<T, const R: usize, const C: usize> Neg for Matrix<T, R, C>
where
    T: FloatType,
{
    type Output = Matrix<T, R, C>;
    fn neg(self) -> Self::Output {
        (&self).neg()
    }
}

impl<T, const R: usize, const C: usize> Index<usize> for Matrix<T, R, C>
where
    T: FloatType,
{
    type Output = [T; C];
    fn index(&self, i: usize) -> &Self::Output {
        &self.data[i]
    }
}

impl<T, const R: usize, const C: usize> IndexMut<usize> for Matrix<T, R, C>
where
    T: FloatType,
{
    fn index_mut(&mut self, i: usize) -> &mut Self::Output {
        &mut self.data[i]
    }
}

impl<'a, const A: usize, const B: usize, const C: usize, T> Mul<&'a Matrix<T, B, C>>
    for &'a Matrix<T, A, B>
where
    T: FloatType,
{
    type Output = Matrix<T, A, C>;
    fn mul(self, rhs: &'a Matrix<T, B, C>) -> Self::Output {
        let mut output = Self::Output::zero();
        for i in 0..A {
            for j in 0..C {
                for k in 0..B {
                    output[i][j] += self[i][k] * rhs[k][j];
                }
            }
        }
        output
    }
}

impl<'a, const R: usize, const C: usize, T> Mul<&'a Vector<T, C>> for &'a Matrix<T, R, C>
where
    T: FloatType,
{
    type Output = Vector<T, R>;
    fn mul(self, rhs: &'a Vector<T, C>) -> Self::Output {
        let mut output = Self::Output::zero();
        for row in 0..R {
            for col in 0..C {
                output[row] += self[row][col] * rhs[col];
            }
        }
        output
    }
}

impl<'a, const R: usize, const C: usize, T> Mul<&'a Matrix<T, R, C>> for &'a Vector<T, R>
where
    T: FloatType,
{
    type Output = Vector<T, C>;
    fn mul(self, rhs: &'a Matrix<T, R, C>) -> Self::Output {
        let mut output = Self::Output::zero();
        for col in 0..C {
            for row in 0..R {
                output[col] += rhs[row][col] * self[row];
            }
        }
        output
    }
}

impl_bin_op_permutations!(
    impl{const A: usize, const B: usize, const C: usize, T: FloatType}
    Mul<Matrix<T, B, C>>
    for Matrix<T, A, B>,
    Output = Matrix<T, A, C>);
impl_bin_op_permutations!(
    impl{const R: usize, const C: usize, T: FloatType}
    Mul<Vector<T, C>>
    for Matrix<T, R, C>,
    Output = Vector<T, R>);
impl_bin_op_permutations!(
    impl{const R: usize, const C: usize, T: FloatType}
    Mul<Matrix<T, R, C>>
    for Vector<T, R>,
    Output = Vector<T, C>);

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
                    let mut out_mat = Matrix::<T, R, C>::zero();
                    for r in 0..R {
                        for c in 0..C {
                            out_mat[r][c] = self[r][c].[<$trait:lower>](rhs[r][c]);
                        }
                    }
                    out_mat
                }
            }

            // Assign ops
            impl<'a, $($tokens)*> [<$trait Assign>]<&'a $trait_type> for $for_type {
                fn [<$trait:lower _assign>](&mut self, rhs: &'a $trait_type) {
                    for r in 0..R {
                        for c in 0..C {
                            self[r][c] = self[r][c].[<$trait:lower>](rhs[r][c]);
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
                    let mut out_mat = Matrix::<T, R, C>::zero();
                    for r in 0..R {
                        for c in 0..C {
                            out_mat[r][c] = self[r][c].[<$trait:lower>](*rhs);
                        }
                    }
                    out_mat
                }
            }

            // Assign ops
            impl<'a, $($tokens)*> [<$trait Assign>]<&'a $trait_type> for $for_type {
                fn [<$trait:lower _assign>](&mut self, rhs: &'a $trait_type) {
                    for r in 0..R {
                        for c in 0..C {
                            self[r][c] = self[r][c].[<$trait:lower>](*rhs);
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

        impl_assign_op_permutations!(
            impl{$($tokens)*}
            $trait<$trait_type>
            for $for_type);
    };
}

// Matrix<T, R, C> op Matrix<T, R, C>
impl_bin_op!(
    impl{const R: usize, const C: usize, T: FloatType}
    Add<Matrix<T, R, C>>
    for Matrix<T, R, C>,
    Output = Matrix<T, R, C>);
impl_bin_op!(
    impl{const R: usize, const C: usize, T: FloatType}
    Sub<Matrix<T, R, C>>
    for Matrix<T, R, C>,
    Output = Matrix<T, R, C>);

// Matrix<T, R, C> op T
impl_bin_op!(
    impl{const R: usize, const C: usize, T: FloatType}
    @scalar Add<T>
    for Matrix<T, R, C>,
    Output = Matrix<T, R, C>);
impl_bin_op!(
    impl{const R: usize, const C: usize, T: FloatType}
    @scalar Sub<T>
    for Matrix<T, R, C>,
    Output = Matrix<T, R, C>);
impl_bin_op!(
    impl{const R: usize, const C: usize, T: FloatType}
    @scalar Mul<T>
    for Matrix<T, R, C>,
    Output = Matrix<T, R, C>);
impl_bin_op!(
    impl{const R: usize, const C: usize, T: FloatType}
    @scalar Div<T>
    for Matrix<T, R, C>,
    Output = Matrix<T, R, C>);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        inverse, length_sq, normalize,
        quaternion::Quat,
        vector::{Float2, Float3, Float4, FromVec},
        Inverse, LengthSq, Normalize,
    };

    const M1: Mat22 = Mat22 {
        data: [[1.0, 2.0], [2.0, 1.0]],
    };
    const M2: Mat22 = Mat22 {
        data: [[3.0, 4.0], [4.0, 3.0]],
    };
    const LIMIT: f32 = 1e-6;

    #[test]
    fn zero() {
        let m = Mat44::zero();
        assert!(
            m == Mat44 {
                data: [[0.0; 4]; 4]
            }
        );
    }

    #[test]
    fn identity() {
        let m = Mat44::identity();

        for r in 0..4 {
            for c in 0..4 {
                if r == c {
                    assert!(m[r][c] == 1.0);
                } else {
                    assert!(m[r][c] == 0.0);
                }
            }
        }
    }

    #[test]
    fn transpose1() {
        let m = Matrix::<f32, 4, 10>::identity();
        let mt = m.transpose();

        for r in 0..10 {
            for c in 0..4 {
                if r == c {
                    assert!(m[c][r] == 1.0);
                    assert!(mt[r][c] == 1.0);
                } else {
                    assert!(m[c][r] == 0.0);
                    assert!(mt[r][c] == 0.0);
                }
            }
        }
    }

    #[test]
    fn transpose2() {
        let m = Mat44 {
            data: [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0, 16.0],
            ],
        };

        for (i, item) in m.transpose().iter().enumerate() {
            let j = i / 4;
            let k = i & 3;
            assert!(item == &m[k][j]);
        }
    }

    #[test]
    fn transpose3() {
        let m = Mat44 {
            data: [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0, 16.0],
            ],
        };
        let mtt = m.transpose().transpose();
        assert!(m == mtt);
    }

    #[test]
    fn transpose_mul_vec() {
        let m = Mat44 {
            data: [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0, 16.0],
            ],
        };
        let v = Float4::one();
        let v2 = m.transpose() * v;
        assert!(v2.x() == 28.0);
        assert!(v2.y() == 32.0);
        assert!(v2.z() == 36.0);
        assert!(v2.w() == 40.0);
    }

    #[test]
    fn vec_mul_transpose() {
        let m = Mat44 {
            data: [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0, 16.0],
            ],
        };
        let v = Float4::one();
        let v2 = v * m.transpose();
        assert!(v2.x() == 10.0);
        assert!(v2.y() == 26.0);
        assert!(v2.z() == 42.0);
        assert!(v2.w() == 58.0);
    }

    #[test]
    fn set_row() {
        let mut m = Mat44 {
            data: [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0, 16.0],
            ],
        };
        let m2 = m;
        m[0] = [0.0; 4];
        assert!(m[0] == [0.0; 4]);
        assert!(m[1] == m2[1]);
        assert!(m[2] == m2[2]);
        assert!(m[3] == m2[3]);
    }

    #[test]
    fn set_col() {
        let mut m = Mat44 {
            data: [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0, 16.0],
            ],
        };
        m.set_col(0, &[0.0; 4]);
        let m2: Mat44 = m.transpose();
        let row = Float4::new([2.0, 6.0, 10.0, 14.0]);
        assert!(m2.rv(0) == Float4::new([0.0; 4]));
        assert!(m2.rv(1) == row);
        assert!(m2.rv(2) == row + 1.0);
        assert!(m2.rv(3) == row + 2.0);
    }

    #[test]
    fn mat44_rv() {
        let mut m = Mat44 {
            data: [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0, 16.0],
            ],
        };
        let m2 = m;

        let vec = Float4::new([4.0, 3.0, 2.0, 1.0]);
        m.srv(0, &vec);
        assert!(m.rv(0) == vec);
        assert!(m.rv::<4>(1) == m2.rv(1));
        assert!(m.rv::<4>(2) == m2.rv(2));
        assert!(m.rv::<4>(3) == m2.rv(3));

        let m2 = Mat44 {
            data: [
                [4.0, 3.0, 2.0, 1.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0, 16.0],
            ],
        };

        assert!(m == m2);
    }

    #[test]
    fn mat44_cv() {
        let mut m = Mat44 {
            data: [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0, 16.0],
            ],
        };
        let m2 = m;

        let vec = Float4::new([4.0, 3.0, 2.0, 1.0]);
        println!("{:#?}", m);
        m.scv(0, &vec);
        println!("{:#?}", m);

        assert!(m.cv(0) == vec);
        assert!(m.cv::<4>(1) == m2.cv(1));
        assert!(m.cv::<4>(2) == m2.cv(2));
        assert!(m.cv::<4>(3) == m2.cv(3));

        let m2 = Mat44 {
            data: [
                [4.0, 2.0, 3.0, 4.0],
                [3.0, 6.0, 7.0, 8.0],
                [2.0, 10.0, 11.0, 12.0],
                [1.0, 14.0, 15.0, 16.0],
            ],
        };

        assert!(m == m2);
    }

    #[test]
    fn index() {
        let m = Mat44 {
            data: [
                [1.0, -1.0, -1.0, 1.0],
                [1.0, 2.0, -1.0, -1.0],
                [-1.0, 1.0, 3.0, -1.0],
                [1.0, -1.0, 1.0, 4.0],
            ],
        };

        for r in 0..4 {
            for c in 0..4 {
                assert!(m[r][c] == m.data[r][c]);
            }
        }
    }

    #[test]
    fn index_mut() {
        let mut m = Mat44::zero();
        m[3][3] = 666.0;

        for r in 0..4 {
            for c in 0..4 {
                if r == 3 && c == 3 {
                    assert!(m[r][c] == 666.0);
                } else {
                    assert!(m[r][c] == 0.0);
                }
            }
        }
    }

    #[test]
    fn add() {
        assert!(
            M1 + M2
                == Mat22 {
                    data: [[4.0, 6.0], [6.0, 4.0],],
                }
        );
    }

    #[test]
    fn sub() {
        assert!(
            M1 - M2
                == Mat22 {
                    data: [[-2.0, -2.0], [-2.0, -2.0],],
                }
        );
    }

    #[test]
    fn mul() {
        assert!(
            M1 * M2
                == Mat22 {
                    data: [[11.0, 10.0], [10.0, 11.0],],
                }
        );
    }

    #[test]
    fn add_scalar() {
        assert!(
            M1 + 1.0
                == Mat22 {
                    data: [[2.0, 3.0], [3.0, 2.0],],
                }
        );
    }

    #[test]
    fn sub_scalar() {
        assert!(
            M1 - 1.0
                == Mat22 {
                    data: [[0.0, 1.0], [1.0, 0.0],],
                }
        );
    }

    #[test]
    fn mul_scalar() {
        assert!(
            M1 * 2.0
                == Mat22 {
                    data: [[2.0, 4.0], [4.0, 2.0],],
                }
        );
    }

    #[test]
    fn div_scalar() {
        assert!(
            M1 / 2.0
                == Mat22 {
                    data: [[0.5, 1.0], [1.0, 0.5],],
                }
        );
    }

    #[test]
    fn mul_vector1() {
        let v1 = Float2::new([4.0, 5.0]);
        assert!(M1 * v1 == Float2::new([14.0, 13.0]));
    }

    #[test]
    fn mul_vector2() {
        let v1 = Float2::new([4.0, 5.0]);
        assert!(v1 * M1.transpose() == Float2::new([14.0, 13.0]));
    }

    #[test]
    fn mul_vector3() {
        let v1 = Float2::new([4.0, 5.0]);
        assert!(v1 * Mat22::identity() == v1);
        assert!(Mat22::identity() * v1 == v1);
    }

    #[test]
    fn neg() {
        assert!(
            -Mat22 {
                data: [[-1.0, -2.0], [-2.0, -1.0],]
            } == M1
        );
    }

    #[test]
    fn scale() {
        let v1 = Float4::new([1.0, 0.5, -1.0, 0.0]);
        let v2 = Float4::new([10.0, 5.0, -10.0, 0.0]);
        let scale = Float3::new([10.0, 10.0, 10.0]);
        let m = Mat44::scale(&scale);
        assert!(m * v1 == v2);
    }

    #[test]
    fn translate1() {
        let v1 = Float3::new([1.0, 0.5, -1.0]);
        let v2 = v1;
        let translation = Float3::new([10.0, 10.0, 10.0]);
        let m = Mat44::translate(&translation);
        assert!(m * Float4::from_vec(v1) == Float4::from_vec(v2));
    }

    #[test]
    fn translate2() {
        let v1 = Float3::new([1.0, 0.5, -1.0]);
        let v2 = Float3::new([11.0, 10.5, 9.0]);
        let translation = Float3::new([10.0, 10.0, 10.0]);
        let m = Mat44::translate(&translation);
        assert!(m * v1.to_point4() == v2.to_point4());
    }

    #[test]
    fn rotate1() {
        let v1 = Float3::new([1.0, 0.5, -1.0]);
        let axis = Float3::new([1.0, 0.0, 0.0]);
        let angle = Angle::Radians(std::f32::consts::FRAC_PI_2);
        let q = Quat::from_axis_angle(&axis, angle);
        let m = Mat44::rotate(&q);
        let rotated1 = q.rotate(&v1);
        let rotated2 = Float3::from_vec(m * Float4::from_vec(v1));
        println!("{}", rotated1);
        println!("{}", rotated2);
        let v3: Float3 = rotated1 - rotated2;
        assert!(v3.x().abs() < LIMIT);
        assert!(v3.y().abs() < LIMIT);
        assert!(v3.z().abs() < LIMIT);
    }

    #[test]
    fn serialize_deserialize() {
        let m = Mat44::identity();
        let m_str = serde_json::to_string_pretty(&m).unwrap();
        let m2: Mat44 = serde_json::from_str(&m_str).unwrap();

        assert!(m == m2);
    }

    #[test]
    fn model_from_trs1() {
        let t = Float3::new([-1.0, 2.0, 666.0]);
        let axis = Float3::new([0.0, 0.0, 1.0]);
        let angle = Angle::Radians(std::f32::consts::FRAC_PI_4);
        let r = Quat::from_axis_angle(&axis, angle);
        let s = Float3::new([15.0, 1.0, 1.0]);
        let m = Mat44::model(&t, &r, &s);

        let starting_point = Float3::new([1.0, 0.0, 0.0]);
        let end_point = Float3::from_vec(m * starting_point.to_point4());
        let ref_end = Float3::new([9.6066, 12.6066, 666.0]);
        println!("{}, {}", end_point, ref_end);
        assert!(length_sq!(&(ref_end - end_point)) < LIMIT);
    }

    #[test]
    fn model_from_trs2() {
        let t = Float3::new([0.0, 0.0, 0.0]);
        let axis = Float3::new([0.0, 0.0, 1.0]);
        let angle = Angle::Radians(0.0);
        let r = Quat::from_axis_angle(&axis, angle);
        let s = Float3::new([1.0, 1.0, 1.0]);
        let m = Mat44::model(&t, &r, &s);

        let starting_point = Float3::new([1.0, 0.0, 0.0]);
        let end_point = Float3::from_vec(m * starting_point.to_point4());
        let ref_end = Float3::new([1.0, 0.0, 0.0]);
        println!("{}, {}", end_point, ref_end);
        assert!(length_sq!(&(ref_end - end_point)) < LIMIT);
    }

    #[test]
    fn model_from_trs3() {
        let t = Float3::new([10.0, 10.0, 0.0]);
        let axis = Float3::new([0.0, 0.0, 1.0]);
        let angle = Angle::Radians(0.0);
        let r = Quat::from_axis_angle(&axis, angle);
        let s = Float3::new([1.0, 1.0, 1.0]);
        let m = Mat44::model(&t, &r, &s);

        let starting_point = Float3::new([1.0, 0.0, 0.0]);
        let end_point = Float3::from_vec(m * starting_point.to_point4());
        let ref_end = Float3::new([11.0, 10.0, 0.0]);
        println!("{}, {}", end_point, ref_end);
        assert!(length_sq!(&(ref_end - end_point)) < LIMIT);
    }

    #[test]
    fn model_from_trs4() {
        let t = Float3::new([0.0, 0.0, 0.0]);
        let axis = Float3::new([0.0, 0.0, 1.0]);
        let angle = Angle::Radians(std::f32::consts::FRAC_PI_2);
        let r = Quat::from_axis_angle(&axis, angle);
        let s = Float3::new([1.0, 1.0, 1.0]);
        let m = Mat44::model(&t, &r, &s);

        let starting_point = Float3::new([1.0, 0.0, 0.0]);
        let end_point = Float3::from_vec(m * starting_point.to_point4());
        let ref_end = Float3::new([0.0, 1.0, 0.0]);
        println!("{}, {}", end_point, ref_end);
        assert!(length_sq!(&(ref_end - end_point)) < LIMIT);
    }

    #[test]
    fn model_from_trs5() {
        let t = Float3::new([0.0, 0.0, 0.0]);
        let axis = Float3::new([0.0, 0.0, 1.0]);
        let angle = Angle::Radians(std::f32::consts::FRAC_PI_2);
        let r = Quat::from_axis_angle(&axis, angle);
        let s = Float3::new([1.0, 1.0, 1.0]);
        let m = Mat44::model(&t, &r, &s);

        let starting_point = Float3::new([0.0, 0.0, 1.0]);
        let end_point = Float3::from_vec(m * starting_point.to_point4());
        let ref_end = Float3::new([0.0, 0.0, 1.0]);
        println!("{}, {}", end_point, ref_end);
        assert!(length_sq!(&(ref_end - end_point)) < LIMIT);
    }

    #[test]
    fn model_from_trs6() {
        let t = Float3::new([0.0, 0.0, 0.0]);
        let axis = Float3::new([0.0, 0.0, 1.0]);
        let angle = Angle::Radians(std::f32::consts::FRAC_PI_2);
        let r = Quat::from_axis_angle(&axis, angle);
        let s = Float3::new([5.0, 1.0, 1.0]);
        let m = Mat44::model(&t, &r, &s);

        let starting_point = Float3::new([0.0, 0.0, 1.0]);
        let end_point = Float3::from_vec(m * starting_point.to_point4());
        let ref_end = Float3::new([0.0, 0.0, 1.0]);
        println!("{}, {}", end_point, ref_end);
        assert!(length_sq!(&(ref_end - end_point)) < LIMIT);
    }

    #[test]
    fn model_from_trs7() {
        let t = Float3::new([0.0, 0.0, 0.0]);
        let axis = Float3::new([0.0, 0.0, 1.0]);
        let angle = Angle::Radians(std::f32::consts::FRAC_PI_2);
        let r = Quat::from_axis_angle(&axis, angle);
        let s = Float3::new([5.0, 1.0, 5.0]);
        let m = Mat44::model(&t, &r, &s);

        let starting_point = Float3::new([0.0, 0.0, 1.0]);
        let end_point = Float3::from_vec(m * starting_point.to_point4());
        let ref_end = Float3::new([0.0, 0.0, 5.0]);
        println!("{}, {}", end_point, ref_end);
        assert!(length_sq!(&(ref_end - end_point)) < LIMIT);
    }

    #[test]
    fn matrix_to_quat_first_branch() {
        let axis = normalize!(&Float3::new([1.0, 0.0, 0.0]));
        let angle = Angle::Radians(4.0);
        let q_orig = Quat::from_axis_angle(&axis, angle);
        let m = Mat44::rotate(&q_orig);
        let q_decomp = m.to_quaternion();
        println!("{}", q_orig);
        println!("{}", q_decomp);

        assert!((q_orig.x() - q_decomp.x()).abs() < LIMIT);
        assert!((q_orig.y() - q_decomp.y()).abs() < LIMIT);
        assert!((q_orig.z() - q_decomp.z()).abs() < LIMIT);
        assert!((q_orig.w() - q_decomp.w()).abs() < LIMIT);
    }

    #[test]
    fn matrix_to_quat_second_branch() {
        let axis = normalize!(&Float3::new([1.0, 1.0, 0.0]));
        let angle = Angle::Radians(2.0);
        let q_orig = Quat::from_axis_angle(&axis, angle);
        let m = Mat44::rotate(&q_orig);
        let q_decomp = m.to_quaternion();
        println!("{}", q_orig);
        println!("{}", q_decomp);

        assert!((q_orig.x() - q_decomp.x()).abs() < LIMIT);
        assert!((q_orig.y() - q_decomp.y()).abs() < LIMIT);
        assert!((q_orig.z() - q_decomp.z()).abs() < LIMIT);
        assert!((q_orig.w() - q_decomp.w()).abs() < LIMIT);
    }

    #[test]
    fn matrix_to_quat_third_branch() {
        let axis = Float3::new([1.0, 0.0, 1.0]);
        let angle = Angle::Radians(3.0);
        let q_orig = Quat::from_axis_angle(&axis, angle);
        let m = Mat44::rotate(&q_orig);
        let q_decomp = m.to_quaternion();
        println!("{}", q_orig);
        println!("{}", q_decomp);

        assert!((q_orig.x() - q_decomp.x()).abs() < LIMIT);
        assert!((q_orig.y() - q_decomp.y()).abs() < LIMIT);
        assert!((q_orig.z() - q_decomp.z()).abs() < LIMIT);
        assert!((q_orig.w() - q_decomp.w()).abs() < LIMIT);
    }

    #[test]
    fn matrix_to_quat_fourth_branch() {
        let axis = Float3::new([1.0, 0.0, 0.0]);
        let angle = Angle::Radians(std::f32::consts::FRAC_PI_2);
        let q_orig = Quat::from_axis_angle(&axis, angle);
        let m = Mat44::rotate(&q_orig);
        let q_decomp = m.to_quaternion();
        println!("{}", q_orig);
        println!("{}", q_decomp);

        assert!((q_orig.x() - q_decomp.x()).abs() < LIMIT);
        assert!((q_orig.y() - q_decomp.y()).abs() < LIMIT);
        assert!((q_orig.z() - q_decomp.z()).abs() < LIMIT);
        assert!((q_orig.w() - q_decomp.w()).abs() < LIMIT);
    }

    #[test]
    fn matrix_trs_to_and_back() {
        let t = Float3::new([-1.0, 2.0, 666.0]);
        let axis = Float3::new([0.0, 1.0, 1.0]);
        let angle = Angle::Radians(std::f32::consts::FRAC_PI_4);
        let r = Quat::from_axis_angle(&axis, angle);
        let s = Float3::new([15.0, 2.0, 1.0]);
        let m = Mat44::model(&t, &r, &s);

        let (t2, r2, s2) = m.extract_trs();
        println!("{}, {}", t, t2);
        println!("{}, {}", r, r2);
        println!("{}, {}", s, s2);

        assert!((t - t2).x().abs() < LIMIT);
        assert!((t - t2).y().abs() < LIMIT);
        assert!((t - t2).z().abs() < LIMIT);
        assert!((s - s2).x().abs() < LIMIT);
        assert!((s - s2).y().abs() < LIMIT);
        assert!((s - s2).z().abs() < LIMIT);
        assert!((r - r2).x().abs() < LIMIT);
        assert!((r - r2).y().abs() < LIMIT);
        assert!((r - r2).z().abs() < LIMIT);
        assert!((r - r2).w().abs() < LIMIT);
    }

    #[test]
    fn matrix_trs_to_and_back_negative_scale_fail() {
        // The function does not work with negative scale
        let t = Float3::new([-1.0, 2.0, 666.0]);
        let axis = Float3::new([0.0, 1.0, 1.0]);
        let angle = Angle::Radians(std::f32::consts::FRAC_PI_4);
        let r = Quat::from_axis_angle(&axis, angle);
        let s = Float3::new([15.0, 2.0, -1.0]);
        let m = Mat44::model(&t, &r, &s);

        let (t2, r2, s2) = m.extract_trs();
        println!("{}, {}", t, t2);
        println!("{}, {}", r, r2);
        println!("{}, {}", s, s2);

        assert!((t - t2).x().abs() < LIMIT);
        assert!((t - t2).y().abs() < LIMIT);
        assert!((t - t2).z().abs() < LIMIT);

        assert!((s - s2).x().abs() < LIMIT);
        assert!((s - s2).y().abs() < LIMIT);
        assert!((s - s2).z() == -2.0);

        // Quaternion is not a unit quaternion, so not a rotator
        assert!(length!(&r2) < 0.8);
    }

    #[test]
    fn inverse_2x2_1() {
        let m = Mat22 {
            data: [[1.0, 2.0], [2.0, 1.0]],
        };
        let inv_m = inverse!(&m);
        let m = m * inv_m;
        println!("{}", m);
        println!("{}", inv_m);
        assert!(m == Mat22::identity());
    }

    #[test]
    fn inverse_2x2_2() {
        let m = Mat22 {
            data: [[-1.0, 2.0], [0.0, 1.0]],
        };
        let inv_m = inverse!(&m);
        let m = m * inv_m;
        println!("{}", m);
        println!("{}", inv_m);
        assert!(m == Mat22::identity());
    }

    #[test]
    fn inverse_2x2_3() {
        let m = Mat22::identity();
        let inv_m = inverse!(&m);
        let m = m * inv_m;
        println!("{}", m);
        println!("{}", inv_m);
        assert!(m == Mat22::identity());
    }

    #[test]
    fn view1() {
        let position = Float3::new([0.0, 0.0, 0.0]);
        let look_at = Float3::new([0.0, 0.0, -1.0]);
        let up = Float3::new([0.0, 1.0, 0.0]);
        let view = Mat44::view(&position, &look_at, &up);

        let point = Float3::new([1.23, 2.34, 3.45]);
        let viewed_point = Float3::from_vec(view * point.to_point4());
        assert!(point == viewed_point);
    }

    #[test]
    fn view2() {
        let position = Float3::new([0.0, 0.0, 0.0]);
        let look_at = Float3::new([1.0, 0.0, 0.0]);
        let up = Float3::new([0.0, 1.0, 0.0]);
        let view = Mat44::view(&position, &look_at, &up);

        let point = Float3::z_axis();
        let viewed_point = Float3::from_vec(view * point.to_point4());
        assert!(viewed_point == Float3::x_axis());
    }

    #[test]
    fn gauss_jordan_inverse1() {
        let m = Mat44 {
            data: [
                [1.0, 2.11, 3.0, 4.4],
                [1.112321, -2.0, 3.7898, 4.321],
                [0.99955, 0.0, 3.123724, 4.0],
                [1.8888, 2.666, 0.0, -51.321],
            ],
        };

        let cl = m;
        let m = m.gj_inverse();
        let i = Mat44::identity();
        let i2 = cl * m;

        println!("{}", m);
        println!("{}", i2);

        for r in 0..4 {
            for c in 0..4 {
                assert!((i[r][c] - i2[r][c]).abs() < 1e-4);
            }
        }
    }

    #[test]
    fn adjugate_inverse_3x3() {
        let m = Mat33 {
            data: [
                [1.0, 2.11, 3.0],
                [1.112321, -2.0, 3.7898],
                [0.99955, 0.0, 3.123724],
            ],
        };
        let ati = inverse!(&m);
        let i2 = ati * m;

        println!("{}", m);
        println!("{}", i2);

        for r in 0..3 {
            for c in 0..3 {
                let limit = 1e-5;
                if r == c {
                    assert!((i2[r][c] - 1.0).abs() < limit);
                } else {
                    assert!(i2[r][c].abs() < limit);
                }
            }
        }
    }

    #[test]
    fn adjugate_inverse_4x4() {
        let m = Mat44 {
            data: [
                [1.0, 2.11, 3.0, 4.4],
                [1.112321, -2.0, 3.7898, 4.321],
                [0.99955, 0.0, 3.123724, 4.0],
                [1.8888, 2.666, 0.0, -51.321],
            ],
        };

        let i = Mat44::identity();
        let i2 = inverse!(&m) * m;

        println!("{}", m);
        println!("{}", i2);

        for r in 0..4 {
            for c in 0..4 {
                assert!((i[r][c] - i2[r][c]).abs() < 1e-4);
            }
        }
    }

    #[test]
    fn failing_inverse1() {
        let m = Mat44::zero();
        for it in inverse!(&m).iter() {
            assert!(f32::is_nan(*it));
        }
    }

    #[test]
    fn perspective1() {
        let view_pos = Float3::new([0.0, 0.0, -10.0]);
        let near = 0.1;
        let far = 12.0;
        let fov = Angle::Degrees(90.0);
        let width = 1920.0;
        let heigth = 1080.0;
        let projection = Mat44::perspective(width / heigth, fov, near, far);
        let clip_pos = projection * view_pos.to_point4();
        let ndc_pos = Float3::from_vec(clip_pos / clip_pos.w());
        println!("{}, {}", clip_pos, ndc_pos);

        assert!(ndc_pos.x() == 0.0);
        assert!(ndc_pos.y() == 0.0);
        assert!(ndc_pos.z() >= 0.0);
        assert!(ndc_pos.z() <= 1.0);
    }

    #[test]
    fn perspective2() {
        let view_pos = Float3::new([0.0, 0.0, 10.0]);
        let near = 0.1;
        let far = 12.0;
        let fov = Angle::Degrees(90.0);
        let width = 1920.0;
        let heigth = 1080.0;
        let projection = Mat44::perspective(width / heigth, fov, near, far);
        let clip_pos = projection * view_pos.to_point4();
        let ndc_pos = Float3::from_vec(clip_pos / clip_pos.w());
        println!("{}, {}", clip_pos, ndc_pos);

        assert!(ndc_pos.x() == 0.0);
        assert!(ndc_pos.y() == 0.0);
        assert!(ndc_pos.z() > 1.0);
    }

    #[test]
    fn perspective3() {
        let view_pos = Float3::new([0.0, 0.1, -0.1]);
        let near = 0.1;
        let far = 12.0;
        let fov = Angle::Degrees(90.0);
        let width = 1920.0;
        let heigth = 1080.0;
        let projection = Mat44::perspective(width / heigth, fov, near, far);
        let clip_pos = projection * view_pos.to_point4();
        let ndc_pos = Float3::from_vec(clip_pos / clip_pos.w());
        println!("{}, {}", clip_pos, ndc_pos);

        assert!(ndc_pos.x() == 0.0);
        assert!(ndc_pos.y() == -1.0);
        assert!(ndc_pos.z() == 0.0);
    }

    #[test]
    fn perspective4() {
        let view_pos = Float3::new([0.1 * 16.0 / 9.0, 0.0, -0.1]);
        let near = 0.1;
        let far = 12.0;
        let fov = Angle::Degrees(90.0);
        let width = 1920.0;
        let heigth = 1080.0;
        let projection = Mat44::perspective(width / heigth, fov, near, far);
        let clip_pos = projection * view_pos.to_point4();
        let ndc_pos = Float3::from_vec(clip_pos / clip_pos.w());
        println!("{}, {}", clip_pos, ndc_pos);

        assert!(ndc_pos.x() == 1.0);
        assert!(ndc_pos.y() == 0.0);
        assert!(ndc_pos.z() == 0.0);
    }

    #[test]
    fn orthographic1() {
        let view_pos = Float3::new([0.0, 0.0, -10.0]);
        let near = 0.1;
        let far = 12.0;
        let fov = Angle::Degrees(90.0);
        let width = 1920.0;
        let heigth = 1080.0;
        let projection = Mat44::orthographic(width / heigth, fov, near, far);
        let clip_pos = projection * view_pos.to_point4();
        let ndc_pos = Float3::from_vec(clip_pos / clip_pos.w());
        println!("{}, {}", clip_pos, ndc_pos);

        assert!(ndc_pos.x() == 0.0);
        assert!(ndc_pos.y() == 0.0);
        assert!(ndc_pos.z() >= 0.0);
        assert!(ndc_pos.z() <= 1.0);
    }

    #[test]
    fn orthographic2() {
        let view_pos = Float3::new([0.0, 0.0, 10.0]);
        let near = 0.1;
        let far = 12.0;
        let fov = Angle::Degrees(90.0);
        let width = 1920.0;
        let heigth = 1080.0;
        let projection = Mat44::orthographic(width / heigth, fov, near, far);
        let clip_pos = projection * view_pos.to_point4();
        let ndc_pos = Float3::from_vec(clip_pos / clip_pos.w());
        println!("{}, {}", clip_pos, ndc_pos);

        assert!(ndc_pos.x() == 0.0);
        assert!(ndc_pos.y() == 0.0);
        assert!(ndc_pos.z() < 1.0);
    }

    #[test]
    fn orthographic3() {
        let view_pos = Float3::new([0.0, 0.1, -0.1]);
        let near = 0.1;
        let far = 12.0;
        let fov = Angle::Degrees(90.0);
        let width = 1920.0;
        let heigth = 1080.0;
        let projection = Mat44::orthographic(width / heigth, fov, near, far);
        let clip_pos = projection * view_pos.to_point4();
        let ndc_pos = Float3::from_vec(clip_pos / clip_pos.w());
        println!("{}, {}", clip_pos, ndc_pos);

        assert!(ndc_pos.x() == 0.0);
        assert!(ndc_pos.y() == -1.0);
        assert!(ndc_pos.z() == 0.0);
    }

    #[test]
    fn orthographic4() {
        let view_pos = Float3::new([0.1 * 16.0 / 9.0, 0.0, -0.1]);
        let near = 0.1;
        let far = 12.0;
        let fov = Angle::Degrees(90.0);
        let width = 1920.0;
        let heigth = 1080.0;
        let projection = Mat44::orthographic(width / heigth, fov, near, far);
        let clip_pos = projection * view_pos.to_point4();
        let ndc_pos = Float3::from_vec(clip_pos / clip_pos.w());
        println!("{}, {}", clip_pos, ndc_pos);

        assert!(ndc_pos.x() == 1.0);
        assert!(ndc_pos.y() == 0.0);
        assert!(ndc_pos.z() == 0.0);
    }

    #[test]
    fn mat_ab_mul_mat_bc() {
        const A: usize = 3;
        const B: usize = 2;
        const C: usize = 4;
        let ab = Matrix::<f32, A, B> {
            data: [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        };
        let bc = Matrix::<f32, B, C> {
            data: [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
        };
        let ac: Matrix<f32, A, C> = ab * bc;
        let result_arr = [
            [11.0, 14.0, 17.0, 20.0],
            [23.0, 30.0, 37.0, 44.0],
            [35.0, 46.0, 57.0, 68.0],
        ];
        assert!(ac.data == result_arr);
    }

    #[test]
    fn mat_ab_mul_transpose_bc() {
        const A: usize = 3;
        const B: usize = 2;
        const C: usize = 4;
        let ab = Matrix::<f32, A, B> {
            data: [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        };
        let bc = Matrix::<f32, C, B> {
            data: [[1.0, 5.0], [2.0, 6.0], [3.0, 7.0], [4.0, 8.0]],
        }
        .transpose();
        let ac: Matrix<f32, A, C> = ab * bc;
        let result_arr = [
            [11.0, 14.0, 17.0, 20.0],
            [23.0, 30.0, 37.0, 44.0],
            [35.0, 46.0, 57.0, 68.0],
        ];
        assert!(ac.data == result_arr);
    }

    #[test]
    fn mat44_to_mat33_1() {
        let mut mat = Mat44::identity();
        mat.srv(2, &Float4::x_axis());
        mat[0][2] = 5.5;
        let mat = Mat33::from(mat);

        assert!(mat.rv(0) == Float3::new([1.0, 0.0, 5.5]));
        assert!(mat.rv(1) == Float3::new([0.0, 1.0, 0.0]));
        assert!(mat.rv(2) == Float3::new([1.0, 0.0, 0.0]));
    }

    #[test]
    fn mat44_to_mat33_2() {
        let mut mat = Mat44::identity();
        mat.srv(2, &Float4::x_axis());
        mat[0][2] = 5.5;
        let mat = Mat33::from(&mat);

        assert!(mat.rv(0) == Float3::new([1.0, 0.0, 5.5]));
        assert!(mat.rv(1) == Float3::new([0.0, 1.0, 0.0]));
        assert!(mat.rv(2) == Float3::new([1.0, 0.0, 0.0]));
    }

    #[test]
    fn mat44_to_mat33_3() {
        let mut mat = Mat44::identity();
        mat.srv(2, &Float4::x_axis());
        mat[0][2] = 5.5;
        let mat = Mat33::from(&mut mat);

        assert!(mat.rv(0) == Float3::new([1.0, 0.0, 5.5]));
        assert!(mat.rv(1) == Float3::new([0.0, 1.0, 0.0]));
        assert!(mat.rv(2) == Float3::new([1.0, 0.0, 0.0]));
    }

    #[test]
    fn mat33_to_mat44_1() {
        let mut mat = Mat33::identity();
        mat.srv(1, &Float3::x_axis());
        mat[0][2] = 5.5;
        let mat = Mat44::from(mat);

        assert!(mat.rv(0) == Float4::new([1.0, 0.0, 5.5, 0.0]));
        assert!(mat.rv(1) == Float4::new([1.0, 0.0, 0.0, 0.0]));
        assert!(mat.rv(2) == Float4::new([0.0, 0.0, 1.0, 0.0]));
        assert!(mat.rv(3) == Float4::new([0.0, 0.0, 0.0, 1.0]));
    }

    #[test]
    fn mat33_to_mat44_2() {
        let mut mat = Mat33::identity();
        mat.srv(1, &Float3::x_axis());
        mat[0][2] = 5.5;
        let mat = Mat44::from(&mat);

        assert!(mat.rv(0) == Float4::new([1.0, 0.0, 5.5, 0.0]));
        assert!(mat.rv(1) == Float4::new([1.0, 0.0, 0.0, 0.0]));
        assert!(mat.rv(2) == Float4::new([0.0, 0.0, 1.0, 0.0]));
        assert!(mat.rv(3) == Float4::new([0.0, 0.0, 0.0, 1.0]));
    }

    #[test]
    fn mat33_to_mat44_3() {
        let mut mat = Mat33::identity();
        mat.srv(1, &Float3::x_axis());
        mat[0][2] = 5.5;
        let mat = Mat44::from(&mut mat);

        assert!(mat.rv(0) == Float4::new([1.0, 0.0, 5.5, 0.0]));
        assert!(mat.rv(1) == Float4::new([1.0, 0.0, 0.0, 0.0]));
        assert!(mat.rv(2) == Float4::new([0.0, 0.0, 1.0, 0.0]));
        assert!(mat.rv(3) == Float4::new([0.0, 0.0, 0.0, 1.0]));
    }
}
