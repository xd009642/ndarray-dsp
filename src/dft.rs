use crate::{ForwardTransformExt, InverseTransformExt};
use ndarray::{prelude::*, s};
use rustfft::num_complex::Complex;
use rustfft::num_traits::{cast::NumCast, Num, NumAssignOps};
use rustfft::{FFTplanner, FFT};
use std::sync::Arc;

#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub struct Dft<T>(Array2<T>);

fn fft_rows(mat: ArrayView<Complex<f64>, Ix2>, fft: Arc<dyn FFT<f64>>) -> Array2<Complex<f64>> {
    let mut row_freqs = Array2::<Complex<f64>>::zeros(mat.dim());
    let mut out = vec![Complex::<f64>::new(0.0, 0.0); mat.shape()[0]];

    for (i, row) in mat.outer_iter().enumerate() {
        let mut r = row.to_owned();
        if let Some(t) = r.as_slice_mut() {
            fft.process(t, &mut out);
            row_freqs.slice_mut(s![i, ..]).assign(&arr1(&out));
        }
    }
    row_freqs
}

impl<T> ForwardTransformExt for Dft<T>
where
    T: Clone + Copy + Num + NumAssignOps + NumCast,
{
    type Output = Array2<Complex<f64>>;

    fn transform(self) -> Self::Output {
        let shape = self.0.shape();

        let converted = self
            .0
            .mapv(|x| x.to_f64().unwrap_or_else(|| 0.0))
            .mapv(|x| Complex::<f64>::new(x, 0.0));

        let mut planner = FFTplanner::<f64>::new(false);
        let row_fft = planner.plan_fft(shape[0]);
        let col_fft = planner.plan_fft(shape[1]);

        let row_freqs = fft_rows(converted.view(), row_fft).reversed_axes();
        fft_rows(row_freqs.view(), col_fft).reversed_axes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ForwardTransformExt;
    use assert_approx_eq::assert_approx_eq;

    fn c(real: f64, imag: f64) -> Complex<f64> {
        Complex::new(real, imag)
    }

    #[test]
    fn simple_test() {
        let expected = arr2(&[
            [c(1.0, 0.0), c(-0.5, -0.8660254), c(-0.5, 0.8660254)],
            [c(-0.5, -0.8660254), c(-0.5, 0.8660254), c(1.0, 0.0)],
            [c(-0.5, 0.8660254), c(1.0, 0.0), c(-0.5, -0.8660254)],
        ]);

        let spatial: Array2<f64> = arr2(&[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]);
        let spatial = Dft(spatial);

        let actual = spatial.transform();

        for (a, e) in actual.iter().zip(expected.iter()) {
            assert_approx_eq!(a.re, e.re);
            assert_approx_eq!(a.im, e.im);
        }
    }
}
