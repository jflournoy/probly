% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/sim_tools.R
\name{plot_splt_sim_behavior}
\alias{plot_splt_sim_behavior}
\title{Plot simulated SPLT behavior}
\usage{
plot_splt_sim_behavior(splt_df, press_right, se = T)
}
\arguments{
\item{splt_df}{a data frame from the SPLT, with non-response
(\code{is.na(pressed_r)}) rows removed}

\item{press_right}{an individual x trial matrix of trial responses}

\item{se}{plot the standard error around smooth lines}
}
\value{
a plot split by sample, and overall summary plot

If press_right is a single matrix, returns a list of plots: one
  across all samples (\code{$sample}), and one overall (\code{$overall}). If
  press_right is a list of matrices, the plot returned does not group by
  sample, returning an overall plot that contains the additional variable
  \code{sim} that can be used in a facet expression. For use with gganimate,
  note that the frame aesthetic is set to \code{sim}.
}
\description{
Plot simulated SPLT behavior
}
