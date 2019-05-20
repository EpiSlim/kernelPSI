#' @export
SKAT <- function(Y, K, sigma = 1){
 if (is.list(K)) {
    Ksum <- Reduce(`+`, K)
  } else {
    Ksum <- K
 }

 Q <- drop(t(Y - mean(Y)) %*% Ksum %*% (Y - mean(Y)))
 Pmat <- sigma * (eye(dim(Ksum)[1]) - ones(dim(Ksum)[1]) / (dim(Ksum)[1]))
 eigD <- eigen(Pmat %*% Ksum %*% Pmat)[["values"]]
 pvalue <- davies(Q, eigD, sigma = 1, lim = 10000, acc = 0.0001)[["Qq"]]

 return(pvalue)
}
