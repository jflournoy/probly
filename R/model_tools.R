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

