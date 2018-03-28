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

#' Make task structure from data
#'
#' @param splt_df a data frame from the SPLT, with non-response (\code{is.na(pressed_r)}) rows removed
#'
#' @return a list containing the elements defining behavior from a SPLT sample
#'   from which to simulate data.
#' @export
make_task_structure_from_data <- function(splt_df){
    if(any(is.na(splt_df$pressed_r))){
        stop('Please remove rows without left or right response before running this function.')
    }
    splt_df$cue <- as.numeric(as.factor(paste0(splt_df$condition, '_', splt_df$sex)))
    splt_df$condition <- factor(splt_df$condition)

    group_index_mm <- probly::get_sample_index(splt_df)
    N <- dim(group_index_mm)[1]
    M <- length(levels(group_index_mm$m_fac))
    K <- length(unique(splt_df$condition))
    cue_mat <- get_col_as_trial_matrix(splt_df, 'cue', id_col = 'id', sample_col = 'sample', trial_col = 'trial_index')
    Tsubj <- probly::get_max_trials_per_individual(cue_mat)
    condition_mat <- get_col_as_trial_matrix(splt_df, 'condition', id_col = 'id', sample_col = 'sample', trial_col = 'trial_index')
    correct_r_mat <- get_col_as_trial_matrix(splt_df, 'correct_r', id_col = 'id', sample_col = 'sample', trial_col = 'trial_index')
    reward_possible_mat <- get_col_as_trial_matrix(splt_df, 'reward_possible', id_col = 'id', sample_col = 'sample', trial_col = 'trial_index')
    outcome_arr <- array(reward_possible_mat, dim = c(dim(reward_possible_mat), 2))
    outcome_arr[,,1][correct_r_mat == 1] <- 0
    outcome_arr[,,2][correct_r_mat == 0] <- 0

    task_structure <- list(
        N = N, M = M, K = K,
        mm = group_index_mm$m,
        Tsubj = Tsubj,
        cue = cue_mat,
        n_cues = max(cue_mat, na.rm = T),
        condition = condition_mat,
        outcome = outcome_arr)

    return(task_structure)
}

#' Simulate SPLT data
#'
#' @param splt_structure a list, as returned from
#'   \code{\link{make_task_structure_from_data}}
#' @param mu_xi a vector of length splt_structure$K of population means for \eqn{\xi}
#' @param mu_b a vector of length splt_structure$K of population means for \eqn{b}
#' @param mu_eps a vector of length splt_structure$K of population means for
#'   \eqn{\epsilon}
#' @param mu_rho a vector of length splt_structure$K of population means for
#'   \eqn{\rho}
#'
#' @return a list with simulated behavior where element \code{press_right} is 0
#'   if the simulation chose 'left' and 1 if it chose 'right'; and where element
#'   \code{outcome_realized} contains the value for each trial that resulted
#'   from the choice.
#' @export
simulate_splt_data <- function(splt_structure, mu_xi, mu_b, mu_eps, mu_rho){
    M <- splt_structure$M
    K <- splt_structure$K

    delta_xi <- probly::sample_deltas(mu_xi, .1, M)
    delta_b <- probly::sample_deltas(mu_b, .1, M)
    delta_eps <- probly::sample_deltas(mu_eps, .1, M)
    delta_rho <- probly::sample_deltas(mu_rho, .1, M)

    Sigma <- matrix(rep(.1, K*K), nrow = K)
    diag(Sigma) <- 1

    Sigma_eps <- Sigma_rho <- Sigma
    Sigma_b <- Sigma_xi <- diag(K)

    beta_xi <- probly::sample_betas(deltas = delta_xi,
                                    group_index = splt_structure$mm,
                                    Sigma = Sigma_xi)
    beta_b <- probly::sample_betas(deltas = delta_b,
                                   group_index = splt_structure$mm,
                                   Sigma = Sigma_b)
    beta_eps <- probly::sample_betas(deltas = delta_eps,
                                     group_index = splt_structure$mm,
                                     Sigma = Sigma_eps)
    beta_rho <- probly::sample_betas(deltas = delta_rho,
                                     group_index = splt_structure$mm,
                                     Sigma = Sigma_rho)

    splt_sim_trials <- generate_responses(
        N = splt_structure$N, M = splt_structure$M, K = splt_structure$K,
        mm = splt_structure$mm, Tsubj = splt_structure$Tsubj,
        cue = splt_structure$cue, n_cues = splt_structure$n_cues,
        condition = splt_structure$condition, outcome = splt_structure$outcome,
        beta_xi = beta_xi, beta_b = beta_b, beta_eps = beta_eps, beta_rho = beta_rho)

    return(list(task_behavior = splt_sim_trials,
                delta_xi = delta_xi,
                delta_b = delta_b,
                delta_eps = delta_eps,
                delta_rho = delta_rho,
                beta_xi = beta_xi,
                beta_b = beta_b,
                beta_eps = beta_eps,
                beta_rho = beta_rho))
}


#' Plot simulated SPLT behavior
#'
#' @param splt_df a data frame from the SPLT, with non-response (\code{is.na(pressed_r)}) rows removed
#' @param press_right_mat an individual x trial matrix of trial responses
#'
#' @return a plot split by sample, and overall summary plot
#' @import dplyr
#' @import tidyr
#' @import ggplot2
#' @export
plot_splt_sim_behavior <- function(splt_df, press_right_mat){
    if(any(is.na(splt_df$pressed_r))){
        stop('Please remove rows without left or right response before running this function.')
    }
    sample_index <- probly::get_sample_index(splt_df)
    p_r_df <- cbind(sample_index, press_right_mat)
    p_r_df <- tidyr::gather(p_r_df, trial, press_r, -id, -sample, -m_fac, -m)

    splt_sim <- dplyr::mutate(
        dplyr::arrange(
            dplyr::group_by(splt_df, id, sample),
            trial_index),
        trial = as.character(1:n()))
    splt_sim <- dplyr::mutate(
        dplyr::arrange(
            dplyr::group_by(splt_sim, id, sample, condition),
            trial_index),
        condition_trial = 1:n())
    splt_sim <- dplyr::left_join(
        splt_sim,
        p_r_df,
        by = c('id', 'sample', 'trial'))
    splt_sim <- dplyr::mutate(
        splt_sim,
        press_r_opt = proportion == '20_80',
        press_opt = press_r == press_r_opt)

    splt_sim_summary <- dplyr::summarise(
        dplyr::group_by(splt_sim, condition, condition_trial),
        trial_mean = mean(press_opt)
    )
    splt_sim_samp_summary <- dplyr::summarise(
        dplyr::group_by(splt_sim, sample, condition, condition_trial),
        trial_mean = mean(press_opt)
    )
    sample_plot <- ggplot2::ggplot(
        splt_sim_samp_summary,
        ggplot2::aes(x = condition_trial,
                     y = trial_mean,
                     group = condition,
                     linetype = condition,
                     shape = condition)) +
        ggplot2::geom_point(alpha = .1) +
        ggplot2::geom_smooth(
            color = 'black', size = .5,
            method = 'gam', formula = y ~ s(x, bs = "cs", k = 8), se = T) +
        ggplot2::facet_wrap(~sample, nrow = 2) +
        ggplot2::labs(x = 'Within-condition trial number',
                      y = 'Proportion of optimal responses') +
        ggplot2::theme_minimal()
    overall_plot <- ggplot2::ggplot(
        splt_sim_summary,
        ggplot2::aes(x = condition_trial,
                     y = trial_mean,
                     group = condition,
                     linetype = condition,
                     shape = condition)) +
        ggplot2::geom_point(alpha = .1) +
        ggplot2::geom_smooth(
            color = 'black', size = .5,
            method = 'gam', formula = y ~ s(x, bs = "cs", k = 8), se = T) +
        ggplot2::labs(x = 'Within-condition trial number',
                      y = 'Proportion of optimal responses') +
        ggplot2::theme_minimal()
    return(list(sample = sample_plot, overall = overall_plot))
}
