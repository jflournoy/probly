## ---- run_many_simulations

#####TEST
mod <- 1
test_sim_num <- 1
#####

library(future)
library(future.batchtools)
library(probly)
library(listenv)

if(grepl('(^n\\d|talapas-ln1)', system('hostname', intern = T))){
    simiter <- 100 #number of sims we have already generated.
    niter <- 2000
    nchains <- 6
    niterperchain <- ceiling(niter/nchains)
    warmup <- 1000
    data_dir <- '/gpfs/projects/dsnlab/flournoy/data/splt/probly'
    plan(batchtools_slurm,
         template = system.file('batchtools', 'batchtools.slurm.tmpl', package = 'probly'),
         resources = list(ncpus = nchains, walltime = 60*24*4, memory = '1G',
                          partitions = 'long,longfat'))
} else {
    simiter <- 2 #number of sims we have already generated.
    data_dir <- '/data/jflournoy/split/probly'
    niter <- 40
    nchains <- 4
    warmup <- 10
    niterperchain <- ceiling(niter/nchains)
    plan(tweak(multiprocess, gc = T, workers = nchains))
}

if(!file.exists(data_dir)){
    stop('Data directory "', data_dir, '" does not exist')
} else {
    message("Data goes here: ", data_dir)
}

model_filename_list <- list(
    # rl_2_level = system.file('stan', 'splt_rl_2_level.stan', package = 'probly'),
    rl_2_level_no_b_2 = list(
        sim_test_fn = file.path(data_dir, 'splt_sim2_test_sims.RDS'),
        stan_model_fn = system.file('stan', 'splt_rl_2_level_no_b_2.stan', package = 'probly'))
    # rl_repar_exp = system.file('stan', 'splt_rl_reparam_exp.stan', package = 'probly'),
    # rl_repar_exp_no_b = system.file('stan', 'splt_rl_reparam_exp_no_b.stan', package = 'probly')
)


results_f <- listenv()

for(mod in seq_along(model_filename_list)){
    print(paste0('Fitting to simulated data from: ', names(model_filename_list)[mod]))
    for(test_sim_num in 1:simiter){
        results_f[[test_sim_num]] %<-% {
            library(rstan)

            source(system.file('r_aux', 'load_data_for_sim.R', package = 'probly'))
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

            message('Loading simulated data from: ', model_filename_list[[mod]]$sim_test_fn)
            mu_tau_parlist <- c('mu_delta_ep', 'mu_delta_rho', 'mu_delta_xi', 'tau_ep', 'tau_rho', 'tau_xi')
            beta_parlist <- c('beta_ep_prm', 'beta_rho_prm', 'beta_xi_prm')
            sim_data <- readRDS(model_filename_list[[1]]$sim_test_fn) #load data

            #for later comparison
            true_pars <- rstan::extract(sim_data, pars = c(mu_tau_parlist, beta_parlist))

            #get the predicted right presses from simulation
            #seems like we shouldn't do this for each sim, but it's a very small
            #amount of the total time (the sampling takes much longer)
            pright_pred_samps <- rstan::extract(sim_data, pars = 'pright_pred')[[1]]
            list_of_pright_pred_mats <- lapply(1:dim(pright_pred_samps)[1], function(i) pright_pred_samps[i,,])

            #done extracting from simulations. rm and gc
            rm(sim_data)
            nada <- gc(verbose = F)

            message('Fitting simulated data for iter: ', test_sim_num)
            press_right_sim_mat <- list_of_pright_pred_mats[[test_sim_num]]
            this_sim_outcome <- outcome_dummy
            this_sim_outcome[press_right_sim_mat == 1] <- outcome_r[press_right_sim_mat == 1]
            this_sim_outcome[press_right_sim_mat == 0] <- outcome_l[press_right_sim_mat == 0]

            stan_sim_data_to_fit <- stan_sim_data
            stan_sim_data_to_fit$outcome <- this_sim_outcome
            stan_sim_data_to_fit$press_right <- press_right_sim_mat
            stan_sim_data_to_fit$run_estimation <- 1

            model_cpl <- rstan::stan_model(model_filename_list[[mod]]$stan_model_fn)



            afit <- rstan::sampling(
                model_cpl,
                data = stan_sim_data_to_fit,
                chains = nchains, cores = nchains,
                iter = warmup + niterperchain, warmup = warmup,
                include = TRUE,
                pars = c(mu_tau_parlist, beta_parlist),
                control = list(max_treedepth = 15, adapt_delta = 0.99))

            mu_tau_ecdf_quantiles <- lapply(mu_tau_parlist, function(par){
                true_par <- true_pars[[par]]
                estimated_par <- rstan::extract(afit, pars = par)[[1]]
                if(length(dim(true_par)) == 3) {
                    true_par <- true_par[test_sim_num,1,]
                    estimated_par <- estimated_par[,1,]
                } else {
                    true_par <- true_par[test_sim_num,]
                }
                true_par_ecdf_quantile <- lapply(1:length(true_par), function(i){
                    ecdf(estimated_par[,i])(true_par[i])
                })
            })
            names(mu_tau_ecdf_quantiles) <- mu_tau_parlist


            beta_ecdf_quantiles <- lapply(beta_parlist, function(par){
                true_par <- true_pars[[par]][test_sim_num,,]
                estimated_par <- rstan::extract(afit, pars = par)[[1]]
                param_grid <- expand.grid(i = 1:(dim(true_par)[1]), k = 1:(dim(true_par)[2]))
                true_par_ecdf_quantile <- lapply(1:dim(param_grid)[1], function(i){
                    ecdf(estimated_par[,param_grid[i,1], param_grid[i,2]])(true_par[param_grid[i,1], param_grid[i,2]])
                })
                param_grid$ecdf_quantile <- unlist(true_par_ecdf_quantile)
                param_grid
            })
            names(beta_ecdf_quantiles) <- beta_parlist

            true_par_quantiles <- list(mu_tau_ecdf_quantiles = mu_tau_ecdf_quantiles,
                                       beta_ecdf_quantiles = beta_ecdf_quantiles)

            saveRDS(true_par_quantiles,
                    file.path(data_dir,
                              paste0('splt-sim_ecdf_check-', names(model_filename_list)[mod],
                                     '-', round(as.numeric(Sys.time())/1000,0),'.RDS')))
            true_par_quantiles
        }
    }
}

results <- lapply(as.list(results_f), future::value)

saveRDS(results, file.path(data_dir, 'splt_ecdf_test_rez.RDS'))

