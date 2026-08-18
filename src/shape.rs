//! Fixed-capacity shape. LLM tensors rarely exceed rank 4
//! (batch, seq, heads, dim); we cap at 6 for headroom.

use crate::error::{Error, Result};

/// Maximum tensor rank. Bump if a model architecture ever needs more.
pub const MAX_RANK: usize = 6;

/// Tensor shape — fixed-cap array of dim sizes plus an explicit rank.
///
/// Stored inline (no heap allocation) so `Shape` is `Copy` and the
/// `Tensor` struct stays small.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shape {
    dims: [usize; MAX_RANK],
    rank: u8,
}

impl Shape {
    /// Construct a shape from a slice of dim sizes.
    /// Errors if `dims.len() > MAX_RANK`.
    pub const fn new(dims: &[usize]) -> Result<Self> {
        if dims.len() > MAX_RANK {
            return Err(Error::TooManyDims {
                got: dims.len(),
                max: MAX_RANK,
            });
        }
        let mut buf = [0usize; MAX_RANK];
        let mut i = 0;
        while i < dims.len() {
            buf[i] = dims[i];
            i += 1;
        }
        Ok(Self {
            dims: buf,
            rank: dims.len() as u8,
        })
    }

    /// Construct a scalar (rank-0) shape.
    pub const fn scalar() -> Self {
        Self {
            dims: [0; MAX_RANK],
            rank: 0,
        }
    }

    /// Number of dimensions.
    pub const fn rank(&self) -> usize {
        self.rank as usize
    }

    /// Slice view over the live dim sizes.
    pub fn dims(&self) -> &[usize] {
        &self.dims[..self.rank as usize]
    }

    /// Total element count (product of dims). Rank-0 = 1.
    pub fn numel(&self) -> usize {
        let mut total = 1usize;
        for &d in self.dims() {
            total = total.saturating_mul(d);
        }
        total
    }

    /// Compute row-major (C-order) strides for this shape, in
    /// elements (not bytes).
    pub fn contiguous_strides(&self) -> Strides {
        let mut strides = [0usize; MAX_RANK];
        let r = self.rank as usize;
        if r > 0 {
            strides[r - 1] = 1;
            let mut i = r - 1;
            while i > 0 {
                strides[i - 1] = strides[i] * self.dims[i];
                i -= 1;
            }
        }
        Strides {
            strides,
            rank: self.rank,
        }
    }
}

/// Strides matching a `Shape` — element offsets per dimension.
///
/// Separate type so transposes/views can change strides without
/// touching the shape array.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Strides {
    strides: [usize; MAX_RANK],
    rank: u8,
}

impl Strides {
    /// Slice view over the live strides.
    pub fn as_slice(&self) -> &[usize] {
        &self.strides[..self.rank as usize]
    }

    /// `true` if these strides describe a contiguous (row-major) layout
    /// for the given shape.
    pub fn is_contiguous(&self, shape: &Shape) -> bool {
        let expected = shape.contiguous_strides();
        self.as_slice() == expected.as_slice()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_numel() {
        let s = Shape::new(&[2, 3, 4]).unwrap();
        assert_eq!(s.rank(), 3);
        assert_eq!(s.numel(), 24);
    }

    #[test]
    fn scalar_numel_is_one() {
        let s = Shape::scalar();
        assert_eq!(s.rank(), 0);
        assert_eq!(s.numel(), 1);
    }

    #[test]
    fn rank_cap_enforced() {
        let too_many = [1usize; MAX_RANK + 1];
        assert!(Shape::new(&too_many).is_err());
    }

    #[test]
    fn contiguous_strides_are_row_major() {
        let s = Shape::new(&[2, 3, 4]).unwrap();
        let st = s.contiguous_strides();
        assert_eq!(st.as_slice(), &[12, 4, 1]);
        assert!(st.is_contiguous(&s));
    }
}
