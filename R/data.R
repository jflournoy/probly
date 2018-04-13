#' Social Probabilistic Learning Task Data
#'
#' Description
#'
#' @format A data frame with M rows and N variables:
#' \describe{
#'   \item{blah}{blah}
#' }
"splt"

#' Social Probabilistic Learning Confidence Data
#'
#' Description
#'
#' @format A data frame with M rows and N variables:
#' \describe{
#'   \item{blah}{blah}
#' }
"splt_confidence"

#' Social Probabilistic Learning Development and Demographic Data
#'
#' Description
#'
#' @format A data frame with M rows and N variables:
#' \describe{
#'   \item{blah}{blah}
#' }
"splt_dev_and_demog"

#' Fundamental Social Motives Inventory Data
#'
#' College sample only. Questionnaire source is:
#'
#' Neel, R., Kenrick, D. T., White, A. E., & Neuberg, S. L. (2015). Individual
#' Differences in Fundamental Social Motives. Journal of Personality and Social
#' Psychology, No Pagination Specified. https://doi.org/10.1037/pspp0000068
#'
#' @format A data frame with M rows and N variables: \describe{
#'   \item{blah}{blah} }
"splt_fsmi"

#' Get sample index
#'
#' Creates a data frame that links each individual to a particular sample. The
#' column \code{m} can be easily passed to the data simulation functions, and to
#' the Stan reinforcement learning model.
#'
#' @param splt_df a data frame containing data from the SPLT
#' @param id_col the name of the id column
#' @param sample_col the name of the sample column
#' @param levels the levels to be used in coercing the sample column to a factor
#'   (to control ordering)
#'
#' @return a data frame with columns id_col, sample_col, m_fac (with sample_col
#'   coerced to a factor) and m (with m_fac coerced to a number)
#' @export
get_sample_index <- function(splt_df, id_col = 'id', sample_col = 'sample', levels = sort(unlist(unique(splt_df[, sample_col])))) {
    mm <- unique(splt_df[, c(id_col, sample_col)])
    rownames(mm) <- 1:dim(mm)[1]
    mm$m_fac <- factor(mm$sample,
                       levels = levels)
    mm$m <- as.numeric(mm$m_fac)
    return(mm)
}

#' Get max trials per individual
#'
#' Easy creation of a vector with the number of trials for each individual
#'
#' @param i_by_t_mat a matrix of N-individuals by T-trials, with NA in any cells
#'   after that individuals max number of trials
#'
#' @return a vector of the number of trials for each individual
#' @export
get_max_trials_per_individual <- function(i_by_t_mat){
    apply(i_by_t_mat, 1, function(row) sum(!is.na(row)))
}

#' Get column as a trial matrix
#'
#' Create an N-individual by T-trial matrix from a column in a SPLT data frame
#'
#' @param splt_df a data frame containing data from the SPLT
#' @param col the column to use for cell values in the returned matrix
#' @param id_col the column if individual IDs
#' @param sample_col the column denoting the sample for each row
#' @param trial_col the column indexing trials, to ensure correct order
#'
#' @return an N-individual by T-trial matrix with values taken from the column
#'   in \code{splt_df} named by \code{col}
#' @export
get_col_as_trial_matrix <- function(splt_df, col, id_col = 'id', sample_col = 'sample', trial_col = 'trial_index'){
    #expects all rows with pressed_r == NA to have been removed
    if(!(is.factor(splt_df[, col][[1]]) | is.numeric(splt_df[, col][[1]]))){
        stop("col must be numeric or a factor that will be coerced to numeric.")
    }
    id_sample_index <- paste0(unlist(splt_df[,id_col]), unlist(splt_df[,sample_col]))
    ids <- unique(id_sample_index)
    max_trials <- max(unlist(lapply(split(splt_df, id_sample_index),
                                    function(x) dim(x)[1])))
    col_mat <- matrix(nrow = length(ids), ncol = max_trials)
    for(id in ids){
        trials_index <- unlist(splt_df[id_sample_index == id, trial_col])
        col_vec <- unlist(splt_df[id_sample_index == id, col])[order(trials_index)]
        col_mat[which(ids == id), 1:length(col_vec)] <- col_vec
    }# end id
    rownames(col_mat) <- ids
    return(col_mat)
}

#' Get date from ms since epoch
#'
#' @param miliseconds_since_epoch how many miliseconds have passed since 1970-01-01
#'
#' @return a date
#' @export
#'
get_date_from_epoch_ms <- function(miliseconds_since_epoch){
    as.POSIXct(miliseconds_since_epoch/1000, origin="1970-01-01")
}
