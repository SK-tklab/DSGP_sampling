# DSGP_sampling
Implementation of Decoupled Sampling GP approximation (DSGP) and compare existing methods using only numpy.  
paper : [Efficiently Sampling Functions from Gaussian Process Posteriors](https://arxiv.org/abs/2002.09309) (ICML2020)

## Overview
Sampling function values from GPs requires O(m^3). (m: number of sampling points)
In addition, it is difficult to sample a function that can be evaluated at any point.  
By approximating the GP with Bayesian linear regression using random Fourier features (RFF-BLR), functions can be sampled, but the behavior of the extrapolation is poor.

In this study, they define a *pathwise update* derived from Matheron's Rule.
We obtain the sample path of the prior from the RFF-BLR and update it with kernel basis to sample function from the GP posterior.
This method allows us to evaluate the function value in O(m) and to sample the function that can be evaluated at any point.

## Plot
- Number of RFF : 2000
- Number of train data n : 3, 10, 50, 100 

DSGP allows efficient sampling with the same accuracy as exact GP.  
In RFF-BLR, the predictions in extrapolation is ill-behaved as the number of train data increases.
Predictive variance becomes small relative to the predictive mean.
This behavior is known as *variance starvation*. (see https://arxiv.org/pdf/1706.01445.pdf)
| |DSGP |Exact GP | RFF-BLR|
|--:|:-:|:-:| :-:|
|n=3| <img src="https://github.com/SK-tklab/DSGP_sampling/blob/main/image/DSGPsample_3.png" width="400px">  |<img src="https://github.com/SK-tklab/DSGP_sampling/blob/main/image/GPsample_3.png" width="400px"> | <img src="https://github.com/SK-tklab/DSGP_sampling/blob/main/image/RFFsample_3.png" width="400px"> |
|n=10| <img src="https://github.com/SK-tklab/DSGP_sampling/blob/main/image/DSGPsample_10.png" width="400px">  |<img src="https://github.com/SK-tklab/DSGP_sampling/blob/main/image/GPsample_10.png" width="400px"> | <img src="https://github.com/SK-tklab/DSGP_sampling/blob/main/image/RFFsample_10.png" width="400px"> |
|n=50| <img src="https://github.com/SK-tklab/DSGP_sampling/blob/main/image/DSGPsample_50.png" width="400px">  |<img src="https://github.com/SK-tklab/DSGP_sampling/blob/main/image/GPsample_50.png" width="400px"> | <img src="https://github.com/SK-tklab/DSGP_sampling/blob/main/image/RFFsample_50.png" width="400px"> |
|n=100| <img src="https://github.com/SK-tklab/DSGP_sampling/blob/main/image/DSGPsample_100.png" width="400px">  |<img src="https://github.com/SK-tklab/DSGP_sampling/blob/main/image/GPsample_100.png" width="400px"> | <img src="https://github.com/SK-tklab/DSGP_sampling/blob/main/image/RFFsample_100.png" width="400px"> |
