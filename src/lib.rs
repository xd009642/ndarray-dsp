/// Discrete Cosine Transform
pub mod dct;
/// Discrete Fourier Transform
pub mod dft;

/// Trait for a forward transform to move a matrix into the frequency domain
pub trait ForwardTransformExt {
    type Output;

    fn transform(self) -> Self::Output;
}

/// Trait for an inverse transform to move from the frequency domain back
/// into the spatial domain
pub trait InverseTransformExt {
    type Output;

    fn inverse(self) -> Self::Output;
}
