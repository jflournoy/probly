% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/sim_tools.R
\name{plot_mu_estimates_v_sims}
\alias{plot_mu_estimates_v_sims}
\title{Plot estimates versus simulation parameters}
\usage{
plot_mu_estimates_v_sims(estimated_samples, sim_params, transform = NULL,
  contrasted = TRUE)
}
\arguments{
\item{estimated_samples}{A matrix or data.frame with iterations in rows and (3) parameters in columns.}

\item{sim_params}{Vector with parameters that generated simulated data.}

\item{transform}{String of name of function to use to transform values.}

\item{contrasted}{If true, parameters are combined like: \code{params[1] + c(0, params[2:3])}}
}
\value{
a plot
}
\description{
Plot estimates versus simulation parameters
}
