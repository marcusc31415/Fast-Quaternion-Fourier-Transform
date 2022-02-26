# Fast-Quaternion-Fourier-Transform

This is an implementation of the Fast Quaternion Fourier Transform. It follows the transform described by Wang, X.-y., Wang, C.-p., Yang, H.-y., & Niu, P.-p. (2013). A robust blind color image watermarking in quaternion fourier transform domain. _Journal of System and Software_, 86(2), 255–277. https://doi.org/10.1016/j.jss.2012.08.015

Also included is an example use of it: recreating a bandlimited signal which has missing data. The signal input is an MxN RGB image that is first encoded as an MxN array of quaternions with 0 real component and then bandlimited by setting some of the components in the fourier transform array to zero. The image (signal) is then reconstructed by using the inverse operator formula for operators with norm less than 1. 

This requires both Numpy and the Numpy-quaternion library to run. 
