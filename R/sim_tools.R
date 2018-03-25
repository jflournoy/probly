#' Sample Deltas
#'
#' This will provide mean coefficients for each sample that you can use to
#' construct individual-level coefficients.
#'
#' @param mu_vec vector of means of length K (one mean per condition)
#' @param sigma positive real number for standard deviation of means across
#'   samples
#' @param nsamples number of samples over which deltas should vary
#'
#' @return deltas is a M = nsamples by K condition matrix
#' @export
#'
sample_deltas <- function(mu_vec, sigma, nsamples){
    deltas_vec <- replicate(nsamples,
                            rnorm(n = length(mu_vec),
                                  mean = mu_vec, sd = sigma))
    deltas <- matrix(deltas_vec,
                     nrow = nsamples,
                     byrow = TRUE)
    return(deltas)
}

#' Sample Betas
#'
#' Using the per-sample means, generate individually varrying coefficients for
#' each condition.
#'
#' @param deltas an M by K matrix of coefficients for each M samples and K
#'   conditions
#' @param group_index a vector of group labels (1:M) for each individual from 1:N
#' @param Sigma a K by K correlation matrix definining the covariance of
#'   individually varying coefficients
#'
#' @return an N by K matrix of coefficients
#' @import MASS
#' @export
#'
sample_betas <- function(deltas, group_index, Sigma){
    deltas_mm <- deltas[group_index, ]
    betas_l <- lapply(1:length(group_index), function(j){
        MASS::mvrnorm(1, mu = deltas[group_index[j], ], Sigma = Sigma)
    })
    betas <- do.call(rbind, betas_l)
    return(betas)
}

#' Generate task responses
#'
#' Creates an N-individuals by T-trials matrix of press-right decisions and
#' resulting outcomes. It takes task definitions, as well as individually
#' varying parameters for each parameter governing the reinforcement learning
#' algorithm.
#'
#' @param N number of individuals
#' @param M number of samples
#' @param K number of conditions
#' @param mm a vector of group labels (1:M) for each individual from 1:N
#' @param Tsubj a vector of trial counts for each individual
#' @param cue an N by max(Tsubj) matrix of integers in 1:n_cues identifying the
#'   cue displayed on each trial
#' @param n_cues total number of unique cues
#' @param condition an N by max(Tsubj) matrix of integers in 1:K identifying the
#'   condition for each trial
#' @param outcome an N by max(Tsubj) by 2 array of real numbers identifying the
#'   outcome reward if the individual chooses left (outcome[,,1]) or right
#'   (outcome[,,2])
#' @param beta_xi an N by K matrix of coefficients governing the amount of
#'   irreducible noise
#' @param beta_b an N by K matrix of coefficients governing the amount of
#'   press-right bias
#' @param beta_eps an N by K matrix of coefficients governing the learning rate
#' @param beta_rho an N by K matrix of coefficients adjusting the relative
#'   amount of reward
#'
#' @return a list with two N by max(Tsubj) matrices (with NA in unused trial
#'   cells). The first matrix, \code{press_right} contains 0 if the decision was
#'   to press left, and 1 if the decision was to press right. The second matrix,
#'   \code{outcome_realized} contains the amount of the feedback received after
#'   the press.
#' @export
#'
generate_responses <- function(N, M, K, mm, Tsubj, cue, n_cues, condition, outcome, beta_xi, beta_b, beta_eps, beta_rho){
    beta_xi_prime <- pnorm(beta_xi)
    beta_eps_prime <- pnorm(beta_eps)
    beta_rho_prime <- exp(beta_rho)

    press_right <- matrix(nrow = N, ncol = max(Tsubj)) #the matrices to return
    outcome_realized <- matrix(nrow = N, ncol = max(Tsubj))

    for(i in 1:N){
        wv_r <- numeric(n_cues) #action weight for press-right
        wv_l <- numeric(n_cues) #action weight for press-left
        qv_r <- numeric(n_cues) #Q value for right
        qv_l <- numeric(n_cues) #Q value for left
        p_right <- numeric(n_cues) #probability of pressing right

        for(t in 1:Tsubj[i]){
            wv_r[ cue[i, t] ]    <- qv_r[ cue[i, t] ] + beta_b[ i, condition[i, t] ] #add bias
            wv_l[ cue[i, t] ]    <- qv_l[ cue[i, t] ]

            p_right[ cue[i, t] ] <- arm::invlogit( wv_r[ cue[i, t] ] - wv_l[ cue[i, t] ] )
            p_right[ cue[i, t] ] <-
                p_right[ cue[i, t] ] * (1 - beta_xi_prime[ i, condition[i, t] ]) +
                beta_xi_prime[ i, condition[i, t] ] / 2 #incorporate noise

            press_right[i, t]    <- rbinom(n = 1, size = 1, prob = p_right[ cue[i, t] ])
            # message('i = ', i, ', t = ', t, ', press_right = ', press_right[i, t], '.')
            if(press_right[i, t]){ # press_right[i, t] == 1
                outcome_realized[i, t] <- outcome[i, t, 2]
                qv_r[ cue[i, t] ] <-
                    qv_r[ cue[i, t] ] + beta_eps_prime[ i, condition[i, t] ] *
                    (beta_rho_prime[ i, condition[i, t] ] * outcome[i, t, 2] - qv_r[ cue[i, t] ])
            } else { # press_right[i, t] == 0
                outcome_realized[i, t] <- outcome[i, t, 1]
                qv_l[ cue[i, t] ] <-
                    qv_l[ cue[i, t] ] + beta_eps_prime[ i, condition[i, t] ] *
                    (beta_rho_prime[ i, condition[i, t] ] * outcome[i, t, 1] - qv_l[ cue[i, t] ])
            }
        } # t loop
    }# i loop
    return(list(press_right = press_right, outcome_realized = outcome_realized))
}
