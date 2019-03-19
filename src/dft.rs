use crate::{ForwardTransformExt, InverseTransformExt};
use ndarray::{prelude::*, s};
use num_traits::{cast::NumCast, Num, NumAssignOps};
use rustfft::num_complex::Complex;
use rustfft::{FFTplanner, FFT};
use std::sync::Arc;

pub trait FftShiftExt {
    
    fn fftshift_inplace(&mut self);

    fn fftshift(&self) -> Self;

}


impl<T> FftShiftExt for Array2<T> where T:Clone + Copy {
    
    fn fftshift_inplace(&mut self) {
        let rows = self.shape()[0];
        let cols = self.shape()[1];
        let rows_2 = ((rows as f32)/2.0).floor() as usize;
        let cols_2 = ((cols as f32)/2.0).floor() as usize;
        
        let temp = self.clone();
        for i in 0..rows {
            let index = (i + rows_2 ) % rows;
            self.slice_mut(s![index, ..]).assign(&temp.slice(s![i, ..]));
        }
        
        let temp = self.clone();
        for i in 0..cols {
            let index = (i + cols_2 ) % cols;
            self.slice_mut(s![.., index]).assign(&temp.slice(s![.., i]));
        }
    }

    fn fftshift(&self) -> Self {
        let mut result = self.clone();
        result.fftshift_inplace();
        result.to_owned()
    }
}


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

        let row_freqs = fft_rows(converted.view(), row_fft);
        if shape[1] == 1 {
            row_freqs
        } else {
            let row_freqs = row_freqs.reversed_axes();
            fft_rows(row_freqs.view(), col_fft).reversed_axes()
        }
    }
}

impl InverseTransformExt for Dft<Complex<f64>> {
    type Output = Array2<Complex<f64>>;

    fn inverse(self) -> Self::Output {
        let inv = self.0.reversed_axes();
        let shape = inv.shape();
        let scale = 1.0 / ((shape[0] * shape[1]) as f64);

        let mut planner = FFTplanner::<f64>::new(true);
        let row_fft = planner.plan_fft(shape[0]);
        let col_fft = planner.plan_fft(shape[1]);

        let row_freqs = fft_rows(inv.view(), row_fft);
        let ans = if shape[1] == 1 {
            row_freqs.reversed_axes()
        } else {
            let row_freqs = row_freqs.reversed_axes();
            fft_rows(row_freqs.view(), col_fft)
        };

        ans.mapv(|x| x * scale)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_approx_eq::assert_approx_eq;

    fn c(real: f64, imag: f64) -> Complex<f64> {
        Complex::new(real, imag)
    }

    #[test]
    fn forward_fourier() {
        let expected = arr2(&[
            [c(1.0, 0.0), c(-0.5, -0.8660254), c(-0.5, 0.8660254)],
            [c(-0.5, -0.8660254), c(-0.5, 0.8660254), c(1.0, 0.0)],
            [c(-0.5, 0.8660254), c(1.0, 0.0), c(-0.5, -0.8660254)],
        ]);

        let spatial: Array2<f64> = arr2(&[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]);
        let frequency_domain = Dft(spatial);

        let actual = frequency_domain.transform();

        for (a, e) in actual.iter().zip(expected.iter()) {
            assert_approx_eq!(a.re, e.re);
            assert_approx_eq!(a.im, e.im);
        }
    }

    #[test]
    fn inverse_fourier() {
        let zero = c(0.0, 0.0);
        let one = c(1.0, 0.0);
        let expected = arr2(&[
            [one, zero, zero, zero],
            [zero, zero, zero, one],
            [zero, zero, one, zero],
            [zero, one, zero, zero],
        ]);

        let freq = arr2(&[
            [c(4.0, 0.0), zero, zero, zero],
            [zero, c(4.0, 0.0), zero, zero],
            [zero, zero, c(4.0, 0.0), zero],
            [zero, zero, zero, c(4.0, 0.0)],
        ]);
        let freq = Dft(freq);
        let actual = freq.inverse();

        for (a, e) in actual.iter().zip(expected.iter()) {
            assert_approx_eq!(a.re, e.re);
            assert_approx_eq!(a.im, e.im);
        }
    }

    #[test]
    fn fftshift_odd_dimension_test() {
        let mut unshifty = arr2(&[[1,2,3],[4,5,6],[7,8,9]]);
        let shifty = arr2(&[[9,7,8],[3,1,2],[6,4,5]]);

        assert_eq!(shifty, unshifty.fftshift());

        unshifty.fftshift_inplace();

        assert_eq!(shifty, unshifty);
    }

    #[test]
    fn fftshift_even_dimension_test() {
        let mut unshifty = arr2(&[[1,2,3],[4,5,6],[7,8,9],[10,11,12]]);
        let shifty = arr2(&[[9,7,8],[12,10,11],[3,1,2],[6,4,5]]);

        assert_eq!(shifty, unshifty.fftshift());

        unshifty.fftshift_inplace();

        assert_eq!(shifty, unshifty);
    }
}
