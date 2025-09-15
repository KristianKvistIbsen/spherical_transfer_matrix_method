# Spherical Transfer Matrix Method (STM)

![Conceptual idea behind STM method](Figure_0_graphical_abstract_STM.pdf)

## Introduction

The Spherical Transfer Matrix (STM) method provides a computationally efficient way to predict acoustic radiation from structures subjected to internal pressure loads. 

## Method Overview

The STM method works by:

1. **Identifying Key Surfaces**: The interior surface ($\Gamma_I$) where pressure is applied, and the exterior surface ($\Gamma_E$) which radiates sound.
2. **Spherical Parameterization**: Both surfaces are mapped to spheres using a quasi-isometric (area-preserving) transformation, making it possible to represent distributed excitations and responses using spherical harmonics.
3. **Transfer Matrix Construction**: For each spherical harmonic excitation on the mapped interior surface, a structural FE model is solved, and the resulting surface normal velocities on the exterior surface are mapped back to the spherical domain and decomposed into spherical harmonics. The relations between excitation and response coefficients form the STM.
4. **Fast Prediction**: Once the STM is built, new excitation conditions can be evaluated almost instantly through matrix-vector multiplication, predicting surface velocities and far-field sound power.

![STM computation steps](Figure_5_STM_method.pdf)


pySDEM module based on 
https://doi.org/10.1137/24M1633911
