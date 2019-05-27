
<!-- README.md is generated from README.Rmd. Please edit that file -->

# kernelPSI

This package implements an exhaustive framework to perform
post-selection inference with kernels. It rests upon quadratic kernel
association scores to measure the association between a given kernel and
an outcome of interest. These scores are used for the selection of the
kernels in a forward fashion. The selection procedure allows the
modeling of the selection event as a succession of quadratic constraints
of the outcome. Finally, under the selection event, we derive empirical
p-values to measure the significance of the effect of the selected
kernels on the outcome.

## Installation

You can install the released version of epiGWAS from
[CRAN](https://CRAN.R-project.org) with:

``` r
install.packages("kernelPSI")
```

The latest development version is directly available from
[GitHub](https://github.com):

``` r
install.packages("devtools")
devtools::install_github("EpiSlim/kernelPSI")
```

## Usage examples

We illustrate the use of `kernelPSI` on a toy example. For the sake of
simplicity, we use linear kernels. Other commonly-used kernels are also
available from the R package `kernlab`.

``` r
require("kernelPSI")
require("kernlab")
require("bindata")

set.seed(64)

# Generation of the covariates, the similarity matrices and the outcome
n_kernels <- 10
m_kernels <- 3
size_kernels <- 5

theta <- .01 

n <- 100
rho <- 0.6
corr <- outer(seq_len(size_kernels), seq_len(size_kernels),
              function(i, j) return(rho^(abs(i-j))))
marg <- 0.4

X <- replicate(n_kernels,
               rmvbin(n, margprob = rep(marg, size_kernels), bincorr = corr),
               simplify = FALSE)

K <- replicate(n_kernels, vanilladot())
Kmat <- sapply(seq_len(n_kernels),
               function(i) {kMatrix <- kernelMatrix(K[[i]], X[[i]]); return(as.kernelMatrix(kMatrix, center = TRUE))},
               simplify = FALSE)
Ksum <- Reduce(`+`, Kmat[seq_len(m_kernels)])
decompK <- eigen(Ksum) 

Y <- as.matrix(theta * decompK$values[1] * decompK$vectors[, 1] + rnorm(n), ncol = 1)
Lmat <- kernelMatrix(new("vanillakernel"), Y)
```

We can now proceed to the selection of the kernels, using either the
fixed or adaptive variants.

``` r
candidate_kernels <- 4

selectFOHSIC <- FOHSIC(Kmat, Lmat, mKernels = candidate_kernels) # Fixed variant
selectAHSIC <- adaFOHSIC(Kmat, Lmat) # adaptive variant
```

Before drawing replicates under the selection event, we first need to
model the corresponding constraints.

``` r
constraintFO <- forwardQ(Kmat, selectFOHSIC)

adaFO <- adaQ(Kmat, selectAHSIC[["selection"]], selectAHSIC[["n"]])
adaS <- selectAHSIC$selection[seq_len(selectAHSIC$n)] # indices of selected kernels
```

The wrapper function `kernelPSI` computes p-values for three different
statistics (see documentation).

``` r
n_replicates <- 1000 # To be increased for real use cases
burn_in <- 100

# Fixed variant ------------------
kernelPSI(Y, K_select = Kmat[selectFOHSIC], constraintFO, method = c("ridge", "hsic"),  
          n_replicates = n_replicates, burn_in = burn_in)
#> $ridge
#> [1] 0.953
#> 
#> $hsic
#> [1] 0.504

# Adaptive variant ------------------
kernelPSI(Y, K_select = Kmat[adaS], constraintFO, method = c("pca"), 
          n_replicates = n_replicates, burn_in = burn_in)
#> $pca
#> [1] 0.89
```
