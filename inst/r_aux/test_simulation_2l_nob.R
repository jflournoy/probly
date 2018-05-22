## ---- test_simulation_2l_nob

#assumes `load_data_for_sim.R` has been run

library(future)
library(future.batchtools)
library(listenv)
library(rstan)
if(grepl('(^n\\d|talapas-ln1)', system('hostname', intern = T))){
    nsims <- 100
    nchains <- 6
    nsimsperchain <- ceiling(nsims/nchains)
    data_dir <- '/gpfs/projects/dsnlab/flournoy/data/splt/probly'
    message('Data dir: ', data_dir)
    plan(batchtools_slurm,
         template = system.file('batchtools', 'batchtools.slurm.tmpl', package = 'probly'),
         resources = list(ncpus = 1, walltime = 60*24*5, memory = '5G',
                          partitions = 'long,longfat'))
    sim_test_fit_fn_pre <- file.path(data_dir, 'splt_sim2_test_fit_wide_prior')
    sim_test_fit_fn <- file.path(data_dir, 'splt_sim2_test_fit_wide_prior.RDS')
} else {
    data_dir <- '/data/jflournoy/split/probly'
    nsims <- 100
    nchains <- 4
    nsimsperchain <- ceiling(nsims/nchains)
    message('Data dir: ', data_dir)
    plan(tweak(multiprocess, gc = T, workers = nchains))
    sim_test_fit_fn <- file.path(data_dir, 'splt_sim2_test_fit.RDS')
}
test_sim_num <- 50
sim_test_fn <- file.path(data_dir, 'splt_sim2_test_sims_wider.RDS')
sim_test_pr_fn <- file.path(data_dir, 'splt_sim2_test_sims_wider_pr.RDS')
stan_model_fn <- system.file('stan', 'splt_rl_2_level_no_b.stan', package = 'probly')
stan_model_est_fn <- system.file('stan', 'splt_rl_2_level_no_b.stan', package = 'probly')
####TEST stan_model_fn <- './inst/stan/splt_rl_2_level_no_b_2.stan'

condition_mat[is.na(condition_mat)] <- -1
outcome_arr[is.na(outcome_arr)] <- -1
outcome_dummy <- matrix(rep(-1, N*max(Tsubj)), nrow = N)
outcome_l <- outcome_arr[,,1]
outcome_r <- outcome_arr[,,2]
press_right_dummy <- matrix(rep(-1, N*max(Tsubj)), nrow = N)
cue_mat[is.na(cue_mat)] <- -1

J_pred <- 0
Xj <- matrix(0,nrow=N,ncol=0)

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
    # J_pred = J_pred,
    # Xj = Xj,
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
    rl_2l_nob_cpl <- rstan::stan_model(stan_model_fn)

    future::plan(future::multiprocess)
    rl_2l_nob_sim_f <- future::future({
        afit <- rstan::sampling(
            rl_2l_nob_cpl,
            data = stan_sim_data,
            include = F,
            pars = dont_save_diff_pars_in_sim,
            chains = nchains, cores = nchains,
            iter = 1000+nsimsperchain, warmup = 1000,
            control = list(max_treedepth = 15, adapt_delta = 0.99))
        pright_pred_samps <- rstan::extract(afit, pars = 'pright_pred')[[1]]
        gc()
        message('Saving sim to: ', sim_test_fn)
        saveRDS(afit, sim_test_fn)
        message('Saving prior predicted samples to: ', sim_test_pr_fn)
        saveRDS(pright_pred_samps, sim_test_pr_fn)
        gc()
        message('Returning fit')
        afit
    })
    message('resolved(rl_2l_nob_sim_f): ', resolved(rl_2l_nob_sim_f))
    message('Getting value of future...')
    future::resolve(rl_2l_nob_sim_f, sleep = 10)
    rl_2l_nob_sim <- future::value(rl_2l_nob_sim_f)
    message('resolved(rl_2l_nob_sim_f): ', resolved(rl_2l_nob_sim_f))
    message('Value of future, rl_2l_nob_sim_f, obtained? ', ifelse(any(grepl('rl_2l_nob_sim$', ls())), T, F))
} else {
    message('Loading simulated data')
    rl_2l_nob_sim <- readRDS(sim_test_fn)
}

message('Value of future, rl_2l_nob_sim_f, available? ', ifelse(any(grepl('rl_2l_nob_sim$', ls())), T, F))

pright_pred_samps <- rstan::extract(rl_2l_nob_sim, pars = 'pright_pred')[[1]]
list_of_pright_pred_mats <- lapply(1:dim(pright_pred_samps)[1], function(i) pright_pred_samps[i,,])
# rm(rl_2l_nob_sim)
nada <- gc(verbose = F)

pop_parlist <- c('mu_delta_ep', 'mu_delta_rho', 'mu_delta_xi',
                 'tau_ep', 'tau_rho', 'tau_xi',
                 'L_Omega_xi', 'L_Omega_ep', 'L_Omega_rho')
indiv_parlist <- c('beta_ep_prm', 'beta_rho_prm', 'beta_xi_prm', 'pR_final', 'log_lik')

if(!file.exists(sim_test_fit_fn)){
    message('Fitting simulated data')
    press_right_sim_mat <- list_of_pright_pred_mats[[test_sim_num]]
    this_sim_outcome <- outcome_dummy
    this_sim_outcome[press_right_sim_mat == 1] <- outcome_r[press_right_sim_mat == 1]
    this_sim_outcome[press_right_sim_mat == 0] <- outcome_l[press_right_sim_mat == 0]

    stan_sim_data_to_fit <- stan_sim_data
    stan_sim_data_to_fit$outcome <- this_sim_outcome
    stan_sim_data_to_fit$press_right <- press_right_sim_mat
    stan_sim_data_to_fit$run_estimation <- 1

    rl_2l_nob_simfit_f <- listenv()
    for(chain in 1:nchains){
        sim_test_fit_fn <- paste0(sim_test_fit_fn_pre, '-chain_', chain, '.RDS')
        rl_2l_nob_simfit_f[[chain]] <- future::future(
            {
                message('This is chain ', chain)
                message('Will save sim fit to: ', sim_test_fit_fn)
                afit <- rstan::stan(
                    stan_model_est_fn,
                    data = stan_sim_data_to_fit,
                    chains = 1, cores = 1,
                    iter = 1534, warmup = 1200,
                    include = TRUE, pars = c(pop_parlist, indiv_parlist),
                    control = list(max_treedepth = 15, adapt_delta = 0.99))
                message('Saving sim fit to: ', sim_test_fit_fn)
                saveRDS(afit, sim_test_fit_fn)
                list(completed = TRUE)
            })
    }
} else {
    message('Loading fit of simulated data')
    rl_2l_nob_simfit <- readRDS(sim_test_fit_fn)
}
