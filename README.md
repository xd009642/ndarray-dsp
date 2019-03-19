# ndarray-dsp

[![Build Status](https://travis-ci.org/xd009642/ndarray-dsp.svg?branch=master)](https://travis-ci.org/xd009642/ndarray-dsp)
[![License:MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Coverage Status](https://coveralls.io/repos/github/xd009642/ndarray-dsp/badge.svg?branch=master)](https://coveralls.io/github/xd009642/ndarray-dsp?branch=master)

This crate aims to provide digital signal processing operations acting on
ndarrays. Currently, it implements the forward and inverse fourier and DCT 
transforms on 2D arrays as well as `fftshift` to move the fundamental frequency
to the matrix centre. 

Example usage:

```Rust
// With matrix as some 2D ndarray 
let mut frequency_domain = Dft(matrix).transform();
frequency_domain.fftshift_inplace();
```
