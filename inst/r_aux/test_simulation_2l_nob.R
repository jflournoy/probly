## ---- estimate_rl_from_sim

#assumes `load_data_for_sim.R` has been run

library(future)
library(rstan)
data_dir <- '/data/jflournoy/split/probly'
sim_test_fn <- file.path(data_dir, 'splt_sim_test_sims.RDS')
sim_test_pr_fn <- file.path(data_dir, 'splt_sim_test_sims_pr.RDS')
sim_test_fit_fn <- file.path(data_dir, 'splt_sim_test_fit.RDS')

condition_mat[is.na(condition_mat)] <- -1
outcome_arr[is.na(outcome_arr)] <- -1
outcome_dummy <- matrix(rep(-1, N*max(Tsubj)), nrow = N)
outcome_l <- outcome_arr[,,1]
outcome_r <- outcome_arr[,,2]
press_right_dummy <- matrix(rep(-1, N*max(Tsubj)), nrow = N)
cue_mat[is.na(cue_mat)] <- -1

stan_sim_data <- list(
    N = N,
    `T` = max(Tsubj),
    K = K,
    ncue = max(cue_mat, na.rm = T),
    Tsubj = Tsubj,
    condition = condition_mat,
    outcome = outcome_dummy,
    outcome_r = outcome_r,
    outcome_l = outcome_l,
    press_right = press_right_dummy,
    cue = cue_mat,
    run_estimation = 0
)

if(!file.exists(sim_test_fn)){
    rl_2l_nob_cpl <- rstan::stan_model(
        system.file('stan', 'splt_rl_2_level_no_b.stan', package = 'probly'))

    rl_2l_nob_sim <- rstan::sampling(
        rl_2l_nob_cpl,
        data = stan_sim_data,
        chains = 4, cores = 4,
        iter = 1100, warmup = 1000,
        control = list(max_treedepth = 15, adapt_delta = 0.99))
    gc()
    pright_pred_samps <- as.matrix(rl_2l_nob_sim)#, pars = grep('pright_pred', names(rl_2l_nob_sim), value = T))
    gc()
    pright_pred_samps <- pright_pred_samps[, grep('pright_pred', dimnames(pright_pred_samps)[[2]], value = T)]
    saveRDS(rl_2l_nob_sim, sim_test_fn)
    saveRDS(pright_pred_samps, sim_test_pr_fn)
    gc()
} else {
    rl_2l_nob_sim <- readRDS(sim_test_fn)
    pright_pred_samps <- readRDS(sim_test_pr_fn)
}

if(!file.exists(sim_test_fit_fn)){
    press_right_sim_mat <- matrix(pright_pred_samps[200,], nrow = N)
    this_sim_outcome <- outcome_dummy
    this_sim_outcome[press_right_sim_mat == 1] <- outcome_r[press_right_sim_mat == 1]
    this_sim_outcome[press_right_sim_mat == 0] <- outcome_l[press_right_sim_mat == 0]

    stan_sim_data_to_fit <- stan_sim_data
    stan_sim_data_to_fit$outcome <- this_sim_outcome
    stan_sim_data_to_fit$press_right <- press_right_sim_mat
    stan_sim_data_to_fit$run_estimation <- 1

    rl_2l_nob_cpl <- rstan::stan_model(
        system.file('stan', 'splt_rl_2_level_no_b.stan', package = 'probly'))

    rl_2l_nob_sim <- rstan::sampling(
        rl_2l_nob_cpl,
        data = stan_sim_data,
        chains = 4, cores = 4,
        iter = 1100, warmup = 1000,
        control = list(max_treedepth = 15, adapt_delta = 0.99))
    gc()
    pright_pred_samps <- as.matrix(rl_2l_nob_sim)#, pars = grep('pright_pred', names(rl_2l_nob_sim), value = T))
    gc()
    pright_pred_samps <- pright_pred_samps[, grep('pright_pred', dimnames(pright_pred_samps)[[2]], value = T)]
    saveRDS(rl_2l_nob_sim, sim_test_fn)
    saveRDS(pright_pred_samps, sim_test_pr_fn)
    gc()
} else {
    rl_2l_nob_sim <- readRDS(sim_test_fn)
    pright_pred_samps <- readRDS(sim_test_pr_fn)
}

