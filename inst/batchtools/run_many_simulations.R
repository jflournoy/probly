## ---- run_many_simulations

library(future)
library(future.batchtools)
library(probly)
library(listenv)

plan(batchtools_slurm,
     template = system.file('batchtools', 'batchtools.slurm.tmpl', package = 'probly'),
     resources = list(ncpus = 6, walltime = 60*24*4, memory = '1G',
                      partitions = 'long,longfat'))

data_dir <- '/home/flournoy/otherhome/data/splt/probly/'
if(!file.exists(data_dir)){
    stop('Data directory "', data_dir, '" does not exist')
} else {
    message("Data goes here: ", data_dir)
}

model_filename_list <- list(
    # intercept_only = system.file('stan', 'splt_intercept_only.stan', package = 'probly'),
    rl_2_level = system.file('stan', 'splt_rl_2_level.stan', package = 'probly'),
    rl_2_level_no_b = system.file('stan', 'splt_rl_2_level_no_b.stan', package = 'probly'),
    rl_repar_exp = system.file('stan', 'splt_rl_reparam_exp.stan', package = 'probly'),
    rl_repar_exp_no_b = system.file('stan', 'splt_rl_reparam_exp_no_b.stan', package = 'probly')
)

iter <- 100
results_f <- listenv()

for(mod in seq_along(model_filename_list)){
    print(paste0('Generating simulated behavior for: ', names(model_filename_list)[mod]))
    for(i in 1:iter){
        results_f[[i]] %<-% {
            library(rstan)
            stan_sim_length <- length(stan_sim_data)
            task_data <- probly::simulate_splt_data(task_structure,
                                                    mu_xi = mu_xi,
                                                    mu_b = mu_b,
                                                    mu_eps = mu_eps,
                                                    mu_rho = mu_rho)
            task_data_stan <- task_data$task_behavior
            task_data_stan$outcome_realized[is.na(task_data_stan$outcome_realized)] <- -1
            task_data_stan$press_right[is.na(task_data_stan$press_right)] <- -1

            stan_sim_data$press_right = task_data_stan$press_right
            stan_sim_data$outcome = task_data_stan$outcome

            trial_num <- -(dim(optimal_side_mat)[2]-1):0
            prop_optimal_resp <- 100*apply(
                task_data$task_behavior$press_right == (optimal_side_mat - 1),
                2, mean, na.rm = T)
            learning_traj <- coef(summary(lm(prop_optimal_resp ~ trial_num)))[,'Estimate']

            # ##--TESTING
            # devtools::install_local('~/code_new/probly')
            # stan_rl_fn <- '~/code_new/probly/inst/stan/splt_rl_reparam.stan'
            # stan_model_thing <- rstan::stan_model(file = stan_rl_fn)
            # stan_optim <- rstan::optimizing(stan_model_thing, data = stan_sim_data)
            # stan_optim_parnames <- grep('mu', names(stan_optim$par), value = T)
            # stan_optim$par[stan_optim_parnames] - c(mu_xi, mu_eps, mu_b, mu_rho)
            # stan_vb <- rstan::vb(stan_model_thing, data = stan_sim_data)
            # summary(stan_vb, pars = stan_optim_parnames)
            # ##--TESTING

            stanFit <- rstan::stan(file = stan_rl_fn,
                                   data = stan_sim_data,
                                   chains = 4, cores = 4,
                                   iter = 1500, warmup = 1000,
                                   control = list(max_treedepth = 15, adapt_delta = 0.99))

            model_summary <- rstan::summary(stanFit)
            sampler_params <- rstan::get_sampler_params(stanFit)

            return_list <- list(fit_summary = model_summary,
                                sampler_params = sampler_params,
                                coefs = learning_traj,
                                data = task_data)

            saveRDS(stanFit,
                    file.path(data_dir,
                              paste0('splt-tght-', names(model_filename_list)[mod],
                                     '-', round(as.numeric(Sys.time())/1000,0),'.RDS')))
        }
    }
}

results <- as.list(results_f)

saveRDS(results, file.path(data_dir, 'test_rez.RDS'))

