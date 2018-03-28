#' Loads previous model fit, or fits and saves
#'
#' \code{CachedFit} takes an expression to evaluate (presumably a model fitting call, but it could be anything), and the filename to which to save the fitted model. If the RDS object exists, it loads it instead of fitting the object. It does not check to make sure that the expression corresponds to that which resulted in the original model fit.
#'
#' @param expr An expression (e.g., \code{\{lm(y ~ x, data = aDataFrame)\}}) to be evaluated.
#' @param rds_filename A filename (full path, or filename in current working directory) to save or load using \code{saveRDS} or \code{loadRDS}.
#'
#' @return the return value of \code{expr} or the object saved in the RDS object located at \code{rds_filename}
#' @export
CachedFit <- function(expr, rds_filename){
  if(file.exists(rds_filename)){
    message('Loading...')
    theFit <- readRDS(rds_filename)
  } else {
    message('Evaluating...')
    theFit <- try(eval(expr))
    saveRDS(theFit, rds_filename)
  }
  return(theFit)
}

#' Get par summaries
#'
#' @param afit an rstan fit
#' @param par_regex regular expression for \code{grep} to use to get paramater names
#'
#' @return a table of summary statistics from all chains in \code{afit}
#' @importMethodsFrom rstan summary
#' @importClassesFrom rstan stanfit
#' @export
get_par_summaries <- function(afit, par_regex, probs = c(.025, .5, .975)){
    if(!class(afit)=='stanfit'){
        stop('afit must be of class "stanfit"')
    }
    par_names <- grep(par_regex, names(afit), value = T)
    return(
        rstan::summary(afit, pars = par_names, probs = probs)$summary
    )
}

#' Extract correlation/covariance matrix samples
#'
#' Because of the decomposition of the beta covariance matrix into scale and
#' correlation, it can be hard to interpret the posteriors. This function will
#' return an array which is a T-by-T-by-Samples array (where T is the number of
#' coefficients). It is then easy to use apply to get posterior means for each
#' cell in the matrix (e.g., \code{apply(Sigma_Array, c(1, 2), mean)}).
#'
#' @param splt_fit a fit of class stanfit
#' @param par_subscript the subscript for the L_Omega_subscript and tau_subscript parameters
#'
#' @return a list with both Omega (correlation matrix) and Sigma (covariance matrix) sample arrays.
#' @export
extract_cor_cov_samps <- function(splt_fit, par_subscript = 'ep'){
    # splt_fit <- readRDS('/data/jflournoy/split/probly/splt_rl_fit_sim.RDS')$fit
    L_Omega_regex <- paste0('L_Omega_', par_subscript)
    tau_regex <- paste0('tau_', par_subscript)

    L_Omega_samp <- as.matrix(splt_fit, pars = grep(L_Omega_regex, names(splt_fit), value = T))
    tau_samp <- as.matrix(splt_fit, pars = grep(tau_regex, names(splt_fit), value = T))

    Omega_samps <- sapply(1:dim(L_Omega_samp)[1], function(i){
        l_om_mat <-  matrix(L_Omega_samp[i, ], nrow = length(L_Omega_samp[i, ])^.5)
        omega_i <- l_om_mat %*% t(l_om_mat)
        return(omega_i)
    }, simplify = 'array')
    Sigma_samps <- sapply(1:dim(L_Omega_samp)[1], function(i){
        l_om_mat <-  matrix(L_Omega_samp[i, ], nrow = length(L_Omega_samp[i, ])^.5)
        tau_i <- tau_samp[i, ]
        Sigma_i <- (diag(tau_i) %*% l_om_mat) %*% t(diag(tau_i) %*% l_om_mat)
        return(Sigma_i)
    }, simplify = 'array')
    return(list(Omega = Omega_samps, Sigma = Sigma_samps))
}
