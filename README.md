
<!-- README.md is generated from README.Rmd. Please edit that file -->

# kernelPSI

This package implements an exhaustive framework to perform
post-selection inference with kernels.

It uses quadratic kernel association scores to measure the association
between a given kernel and an outcome of interest. These scores are used
for the selection of the kernels in a forward fashion. If kernels are
defined on sets of features, this allows for non-linear feature
selection; if the kernels used all belong to the same family, but with
different hyperparameters, this allows for hyperparameter selection.

The selection procedure allows the modeling of the selection event as a
succession of quadratic constraints of the outcome. Finally, under the
selection event, we derive empirical p-values to measure the
significance of the effect of the selected kernels on the outcome.

## Installation

You can install the released version of kernelPSI from
[CRAN](https://CRAN.R-project.org) with:

``` r
install.packages("kernelPSI")
```

The latest development version is directly available from
[GitHub](https://github.com):

<!--
The latest version is directly available from [GitHub](https://github.com):
-->

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
n_kernels <- 10 # total number of kernels
m_kernels <- 3  # number of kernels used to generate the outcome
size_kernels <- 5 # dimensionality of the data associated to each kernel 

theta <- .05 # effect size

n <- 100 # sample size
rho <- 0.6 # correlation parameter (comprised between -1 and +1)

# Correlation matrix
corr <- outer(seq_len(size_kernels), seq_len(size_kernels),
              function(i, j) return(rho^(abs(i-j))))
marg <- 0.4 # parameter for for the Bernoulli margin distributions

# Design matrix
X <- replicate(n_kernels,
               rmvbin(n, margprob = rep(marg, size_kernels), bincorr = corr),
               simplify = FALSE)


K <- replicate(n_kernels, vanilladot()) # full set of kernels

# Gram matrices
Kmat <- sapply(seq_len(n_kernels),
               function(i) {kMatrix <- kernelMatrix(K[[i]], X[[i]]); return(as.kernelMatrix(kMatrix, center = TRUE))},
               simplify = FALSE)
Ksum <- Reduce(`+`, Kmat[seq_len(m_kernels)]) # Gram matrix for the sum kernel of the first m_kernels kernels
decompK <- eigen(Ksum) # eigenvalue decomposition of the Ksum matrix

Y <- as.matrix(theta * decompK$values[1] * decompK$vectors[, 1] + rnorm(n), ncol = 1) # response vector
Lmat <- kernelMatrix(new("vanillakernel"), Y) # linear kernel of the response
```

We can now proceed to the selection of the kernels, using either the
fixed or adaptive variants.

``` r
candidate_kernels <- 4 # number of selected kernels for the fixed variant

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
n_replicates <- 1000 # number of replicates (to be increased for real use cases)
burn_in <- 100 # number of burn-in iterations

# Fixed variant ------------------
# selected methods: 'ridge' for the kernel ridge regression prototype 
# and 'hsic' for the HSIC unbiased estimator
kernelPSI(Y, K_select = Kmat[selectFOHSIC], constraintFO, method = c("ridge", "hsic"),  
          n_replicates = n_replicates, burn_in = burn_in)
#> $ridge
#> [1] 0.022
#> 
#> $hsic
#> [1] 0.06

# Adaptive variant ------------------
# selected method: 'pca' for the kernel principal component regression prototype
kernelPSI(Y, K_select = Kmat[adaS], constraintFO, method = c("pca"),
          n_replicates = n_replicates, burn_in = burn_in)
#> $pca
#> [1] 0.192
```

## References

Lotfi Slim, Clément Chatelain, Chloé-Agathe Azencott, and Jean-Philippe
Vert. kernelPSI: a post-selection inference framework for nonlinear
variable selection, Proceedings of the Thirty-Sixth International
Conference on Machine Learning (ICML), 2019.
