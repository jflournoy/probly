## ---- test_simulation_2l_nob

#assumes `load_data_for_sim.R` has been run

library(future)
library(rstan)
data_dir <- '/data/jflournoy/split/probly'
sim_test_fn <- file.path(data_dir, 'splt_sim_test_sims.RDS')
sim_test_pr_fn <- file.path(data_dir, 'splt_sim_test_sims_pr.RDS')
sim_test_fit_fn <- file.path(data_dir, 'splt_sim_test_fit.RDS')

nsims <- 100
nchains <- 4

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

dont_save_diff_pars_in_sim <- c('beta_xi_diffs',
                                'beta_ep_diffs',
                                'beta_rho_diffs',
                                'mu_delta_xi_diff',
                                'mu_delta_ep_diff',
                                'mu_delta_rho_diff',
                                'Sigma_xi',
                                'Sigma_ep',
                                'Sigma_rho')

if(!file.exists(sim_test_fn)){
    message('Generating simulated data')
    rl_2l_nob_cpl <- rstan::stan_model(
        system.file('stan', 'splt_rl_2_level_no_b.stan', package = 'probly'))

    future::plan(future::multiprocess)
    rl_2l_nob_sim_f <- future::future({
        afit <- rstan::sampling(
            rl_2l_nob_cpl,
            data = stan_sim_data,
            include = F,
            pars = dont_save_diff_pars_in_sim,
            chains = nchains, cores = nchains,
            iter = 1000+nsims, warmup = 1000,
            control = list(max_treedepth = 15, adapt_delta = 0.99))
        gc()
        pright_pred_samps <- rstan::extract(rl_2l_nob_sim, pars = 'pright_pred')[[1]]
        gc()
        saveRDS(afit, sim_test_fn)
        saveRDS(pright_pred_samps, sim_test_pr_fn)
        gc()
    })
    resolved(rl_2l_nob_sim_f)
    rl_2l_nob_sim <- future::value(rl_2l_nob_sim_f)
} else {
    message('Loading simulated data')
    rl_2l_nob_sim <- readRDS(sim_test_fn)
    pright_pred_samps <- readRDS(sim_test_pr_fn)
}

pright_pred_samps <- rstan::extract(rl_2l_nob_sim, pars = 'pright_pred')[[1]]
list_of_pright_pred_mats <- lapply(1:dim(pright_pred_samps)[1], function(i) pright_pred_samps[i,,])

if(!file.exists(sim_test_fit_fn)){
    message('Fitting simulated data')
    press_right_sim_mat <- list_of_pright_pred_mats[[50]]
    this_sim_outcome <- outcome_dummy
    this_sim_outcome[press_right_sim_mat == 1] <- outcome_r[press_right_sim_mat == 1]
    this_sim_outcome[press_right_sim_mat == 0] <- outcome_l[press_right_sim_mat == 0]

    stan_sim_data_to_fit <- stan_sim_data
    stan_sim_data_to_fit$outcome <- this_sim_outcome
    stan_sim_data_to_fit$press_right <- press_right_sim_mat
    stan_sim_data_to_fit$run_estimation <- 1

    rl_2l_nob_cpl <- rstan::stan_model(
        system.file('stan', 'splt_rl_2_level_no_b.stan', package = 'probly'))

    future::plan(future::multiprocess)
    rl_2l_nob_simfit_f <- future::future(
        {
            afit <- rstan::sampling(
                rl_2l_nob_cpl,
                data = stan_sim_data_to_fit,
                chains = nchains, cores = nchains,
                iter = 1500, warmup = 1000,
                include = FALSE, pars = 'pright_pred', #no need to save predicted task behavior
                control = list(max_treedepth = 15, adapt_delta = 0.99))
            saveRDS(afit, sim_test_fit_fn)
            gc()
            afit
        })
    future::resolved(rl_2l_nob_simfit_f)
    rl_2l_nob_simfit <- future::value(rl_2l_nob_simfit_f)
} else {
    message('Loading fit of simulated data')
    rl_2l_nob_simfit <- readRDS(sim_test_fit_fn)
}

rstan::summary(rl_2l_nob_simfit, pars = 'mu_delta_ep')$summary
true_mu_ep <- rstan::extract(rl_2l_nob_sim, pars = 'mu_delta_ep')[[1]]
behav_test <- rstan::extract(rl_2l_nob_sim, pars = 'pright_pred')[[1]]
all(stan_sim_data_to_fit$press_right == behav_test[50,,])
true_mu_ep[50,,]

bayesplot::mcmc_areas(rl_2l_nob_simfit_mat, pars = paste0('mu_delta_ep[1,', 1:3, ']')) +
    ggplot2::geom_vline(xintercept = true_ep[50,])
