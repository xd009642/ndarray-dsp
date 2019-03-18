use crate::{ForwardTransformExt, InverseTransformExt};
use ndarray::{prelude::*, s};
use num_traits::{Num, NumAssignOps};
use rustdct::*;

#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub struct Dct<T>(Array2<T>);

impl<T> ForwardTransformExt for Dct<T>
where
    T: Clone + Copy + Num + NumAssignOps + DCTnum,
{
    type Output = Array2<T>;

    fn transform(self) -> Self::Output {
        self.perform_dct2()
    }
}

impl<T> InverseTransformExt for Dct<T>
where
    T: Clone + Copy + Num + NumAssignOps + DCTnum,
{
    type Output = Array2<T>;

    fn inverse(self) -> Self::Output {
        self.perform_dct3()
    }
}

/// Provide access to ND DCT transformations for all supported types.
pub trait DctExt {
    type Output;
    
    fn perform_dct(self, ty: DctType) -> Self::Output;

    fn perform_dct1(self) -> Self::Output;
    fn perform_dct2(self) -> Self::Output;
    fn perform_dct3(self) -> Self::Output;
    fn perform_dct4(self) -> Self::Output;
}

#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub enum DctType {
    Type1, Type2, Type3, Type4
}

impl<T> DctExt for Dct<T>
where
    T: Copy + Num + NumAssignOps + DCTnum,
{
    type Output = Array2<T>;

    fn perform_dct(self, ty: DctType) -> Self::Output {
        use DctType::*;
        match ty {
            Type1 => self.perform_dct1(),
            Type2 => self.perform_dct2(),
            Type3 => self.perform_dct3(),
            Type4 => self.perform_dct4(),
        }
    }

    fn perform_dct1(self) -> Self::Output {
        let mut rows = Array2::<T>::zeros(self.0.dim());
        let mut result = Array2::<T>::zeros(self.0.dim());

        let shape = self.0.shape();

        let scale =
            T::from_f64(((shape[0] * shape[1] - 1) as f64) / 2.0).unwrap_or_else(|| T::one());

        let mut planner = DCTplanner::<T>::new();
        let row_dct = planner.plan_dct1(shape[0]);
        let col_dct = planner.plan_dct1(shape[1]);

        let mut buffer = vec![T::zero(); shape[0].max(shape[1])];
        for (i, row) in self.0.outer_iter().enumerate() {
            let mut r = row.to_owned();
            if let Some(t) = r.as_slice_mut() {
                row_dct.process_dct1(t, &mut buffer);
                rows.slice_mut(s![i, ..]).assign(&arr1(&buffer[..shape[0]]));
            }
        }
        for (i, col) in rows.t().outer_iter().enumerate() {
            let mut c = col.to_owned();
            if let Some(t) = c.as_slice_mut() {
                col_dct.process_dct1(t, &mut buffer);
                result
                    .slice_mut(s![.., i])
                    .assign(&arr1(&buffer[..shape[1]]));
            }
        }
        result.mapv(|x| x * scale)
    }

    fn perform_dct2(self) -> Self::Output {
        let mut rows = Array2::<T>::zeros(self.0.dim());
        let mut result = Array2::<T>::zeros(self.0.dim());

        let shape = self.0.shape();

        let scale = T::from_f64(((shape[0]*shape[1] - 1) as f64) / 2.0);
        let scale = scale.unwrap_or_else(|| T::one());

        let mut planner = DCTplanner::<T>::new();
        let row_dct = planner.plan_dct2(shape[0]);
        let col_dct = planner.plan_dct2(shape[1]);

        let mut buffer = vec![T::zero(); shape[0].max(shape[1])];
        for (i, row) in self.0.outer_iter().enumerate() {
            let mut r = row.to_owned();
            if let Some(t) = r.as_slice_mut() {
                row_dct.process_dct2(t, &mut buffer);
                rows.slice_mut(s![i, ..]).assign(&arr1(&buffer[..shape[0]]));
            }
        }
        result.mapv_inplace(|x| x*scale);
        for (i, col) in rows.t().outer_iter().enumerate() {
            let mut c = col.to_owned();
            if let Some(t) = c.as_slice_mut() {
                col_dct.process_dct2(t, &mut buffer);
                result
                    .slice_mut(s![.., i])
                    .assign(&arr1(&buffer[..shape[1]]));
            }
        }
        result.mapv(|x| x * scale)
    }

    fn perform_dct3(self) -> Self::Output {
        let mut rows = Array2::<T>::zeros(self.0.dim());
        let mut result = Array2::<T>::zeros(self.0.dim());

        let shape = self.0.shape();

        let scale =
            T::from_f64(((shape[0] * shape[1] - 1) as f64) / 2.0).unwrap_or_else(|| T::one());

        let mut planner = DCTplanner::<T>::new();
        let row_dct = planner.plan_dct3(shape[0]);
        let col_dct = planner.plan_dct3(shape[1]);

        let mut buffer = vec![T::zero(); shape[0].max(shape[1])];
        for (i, row) in self.0.outer_iter().enumerate() {
            let mut r = row.to_owned();
            if let Some(t) = r.as_slice_mut() {
                row_dct.process_dct3(t, &mut buffer);
                rows.slice_mut(s![i, ..]).assign(&arr1(&buffer[..shape[0]]));
            }
        }
        result.mapv_inplace(|x| x * scale);
        for (i, col) in rows.t().outer_iter().enumerate() {
            let mut c = col.to_owned();
            if let Some(t) = c.as_slice_mut() {
                col_dct.process_dct3(t, &mut buffer);
                result
                    .slice_mut(s![.., i])
                    .assign(&arr1(&buffer[..shape[1]]));
            }
        }
        result.mapv(|x| x * scale)
    }

    fn perform_dct4(self) -> Self::Output {
        let mut rows = Array2::<T>::zeros(self.0.dim());
        let mut result = Array2::<T>::zeros(self.0.dim());

        let shape = self.0.shape();

        let scale =
            T::from_f64(((shape[0] * shape[1] - 1) as f64) / 2.0).unwrap_or_else(|| T::one());

        let mut planner = DCTplanner::<T>::new();
        let row_dct = planner.plan_dct4(shape[0]);
        let col_dct = planner.plan_dct4(shape[1]);

        let mut buffer = vec![T::zero(); shape[0].max(shape[1])];
        for (i, row) in self.0.outer_iter().enumerate() {
            let mut r = row.to_owned();
            if let Some(t) = r.as_slice_mut() {
                row_dct.process_dct4(t, &mut buffer);
                rows.slice_mut(s![i, ..]).assign(&arr1(&buffer[..shape[0]]));
            }
        }
        for (i, col) in rows.t().outer_iter().enumerate() {
            let mut c = col.to_owned();
            if let Some(t) = c.as_slice_mut() {
                col_dct.process_dct4(t, &mut buffer);
                result
                    .slice_mut(s![.., i])
                    .assign(&arr1(&buffer[..shape[1]]));
            }
        }
        result.mapv(|x| x * scale)
    }
}

/// Test data for this module is generated using scipy
#[cfg(test)]
mod tests {
    use super::*;
    use assert_approx_eq::assert_approx_eq;

    #[test]
    fn test_dct_1() {
        let expected = arr2(&[[4.0, 0.0, -4.0], [0.0, 0.0, 0.0], [-4.0, 0.0, 4.0]]);

        let input: Array2<f64> = arr2(&[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]);

        let dct = Dct(input);

        let actual = dct.perform_dct1();
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert_approx_eq!(a, e);
        }
    }

    #[test]
    fn test_dct_2() {
        let expected = arr2(&[[16.0, 3.46410162, 2.0], 
                            [-3.46410162, 3.0, -1.73205081], 
                            [2.0, 1.73205081, 7.0]]);

        let input: Array2<f64> = arr2(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 1.0]]);

        let dct = Dct(input);

        let actual = dct.perform_dct2();
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert_approx_eq!(a, e);
        }
    }

    #[test]
    fn test_dct_3() {
        let expected = arr2(&[[6.0, -4.4408921e-16, -4.4408921e-16], 
                            [-3.0, 3.0, -3.0], 
                            [-8.8817842e-16, 0.0, 6.0]]);
        
        let input: Array2<f64> = arr2(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 1.0]]);

        let dct = Dct(input);

        let actual = dct.perform_dct3();

        for (a, e) in actual.iter().zip(expected.iter()) {
            assert_approx_eq!(a, e);
        }
    }

    #[test]
    fn test_dct_4() {
        let expected = arr2(&[[2.0, -2.0, -2.0], [-2.0, 2.0, 2.0], [-2.0, 2.0, 2.0]]);

        let input: Array2<f64> = arr2(&[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]);

        let dct = Dct(input);

        let actual = dct.perform_dct4();
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert_approx_eq!(a, e);
        }
    }
}
