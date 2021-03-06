% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/sim_tools.R
\name{sample_betas}
\alias{sample_betas}
\title{Sample Betas}
\usage{
sample_betas(deltas, group_index, Sigma)
}
\arguments{
\item{deltas}{an M by K matrix of coefficients for each M samples and K
conditions}

\item{group_index}{a vector of group labels (1:M) for each individual from 1:N}

\item{Sigma}{a K by K correlation matrix definining the covariance of
individually varying coefficients}
}
\value{
an N by K matrix of coefficients
}
\description{
Using the per-sample means, generate individually varrying coefficients for
each condition.
}
