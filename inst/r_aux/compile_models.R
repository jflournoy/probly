library(cmdstanr)
library(posterior)
library(bayesplot)
source('inst/r_aux/prep_model_data.R')
source(system.file('inst', 'r_aux', 'prep_model_data.R', package = 'probly'))
model_filename_list <- list(
    intercept_only = system.file('stan', 'splt_intercept_only.stan', package = 'probly'),
    rl_2_level = system.file('stan', 'splt_rl_2_level.stan', package = 'probly'),
    rl_2_level_rho5 = system.file('stan', 'splt_rl_2_level_2rho.stan', package = 'probly'),
    rl_2_level_rho5 = '~/code_new/probly/inst/stan/splt_rl_2_level_2rho.stan',
    rl_2_level_no_b = system.file('stan', 'splt_rl_2_level_no_b.stan', package = 'probly'),
    rl_2_level_no_b_no_rho = system.file('stan', 'splt_rl_2_level_no_b_no_rho.stan', package = 'probly'),
    rl_repar_exp = system.file('stan', 'splt_rl_reparam_exp.stan', package = 'probly'),
    rl_repar_exp_no_b = system.file('stan', 'splt_rl_reparam_exp_no_b.stan', package = 'probly'),
    rl_repar_exp_no_b_no_rho = system.file('stan', 'splt_rl_reparam_exp_no_b_no_rho.stan', package = 'probly')
)

pop_parlist <- c('mu_delta_ep', 'mu_delta_xi',
                 'tau_ep', 'tau_xi',
                 'L_Omega_xi', 'L_Omega_ep')
indiv_parlist <- c('beta_ep_prm', 'beta_xi_prm', 'pR_final', 'log_lik')

pop_parlist_rho <- c('mu_delta_rho', 'tau_rho', 'L_Omega_rho')
indiv_parlist_rho <- c('beta_rho_prm')

pop_parlist_rho5 <- c('mu_delta_rho5', 'tau_rho5', 'L_Omega_rho5')
indiv_parlist_rho5 <- c('beta_rho_prm5')

pop_parlist_b <- c('mu_delta_b', 'tau_b', 'L_Omega_b')
indiv_parlist_b <- c('beta_b')

pop_parlist_3l <- c('delta_xi', 'delta_ep',
                    'sigma_delta_xi', 'sigma_delta_ep')
pop_parlist_3l_rho <- c('delta_rho', 'sigma_delta_rho')
pop_parlist_3l_b <- c('delta_b', 'sigma_delta_b')


save_pars_list <- list(
    rl_2_level = c(pop_parlist, indiv_parlist, pop_parlist_rho, indiv_parlist_rho, pop_parlist_b, indiv_parlist_b),
    rl_2_level_rho5 = c(pop_parlist, indiv_parlist,
                        pop_parlist_rho, indiv_parlist_rho,
                        pop_parlist_rho5, indiv_parlist_rho5,
                        pop_parlist_b, indiv_parlist_b),
    rl_2_level_no_b = c(pop_parlist, indiv_parlist, pop_parlist_rho, indiv_parlist_rho),
    rl_2_level_no_b_no_rho = c(pop_parlist, indiv_parlist),
    rl_repar_exp = c(pop_parlist, indiv_parlist,
                     pop_parlist_rho, indiv_parlist_rho,
                     pop_parlist_b, indiv_parlist_b,
                     pop_parlist_3l, pop_parlist_3l_rho, pop_parlist_3l_b),
    rl_repar_exp_no_b = c(pop_parlist, indiv_parlist,
                          pop_parlist_rho, indiv_parlist_rho,
                          pop_parlist_3l, pop_parlist_3l_rho),
    rl_repar_exp_no_b_no_rho = c(pop_parlist, indiv_parlist, pop_parlist_3l)
)

model_file <- 'inst/stan/splt_rl_2_level.stan'
model_file <- system.file('stan', 'splt_rl_2_level.stan', package = 'probly')
bin_dir <- file.path(system.file(package = 'probly'), 'bin')
if(! dir.exists(bin_dir)){
    dir.create()
}

#model_bin <- cmdstan_model(stan_file = model_file, dir = bin_dir, pedantic = TRUE, cpp_options = list(stan_threads = TRUE))
model_bin <- cmdstan_model(stan_file = model_file, dir = bin_dir, pedantic = TRUE)
stan_data <- make_stan_data(subset = 30)
fit <- model_bin$sample(data = stan_data, chains = 4, parallel_chains = 4, iter_warmup = 100, iter_sampling = 100,
                        max_treedepth = 15, adapt_delta = .99)
fit$summary(variables = save_pars_list$rl_2_level)

