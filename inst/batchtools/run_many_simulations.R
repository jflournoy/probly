## ---- run_many_simulations

library(future)
library(future.batchtools)
library(probly)
library(listenv)

plan(batchtools_slurm,
     template = system.file('batchtools', 'batchtools.slurm.tmpl', package = 'probly'),
     resources = list(ncpus = 4, walltime = 60*24-1, memory = '1G',
                      partitions = 'short,fat,long,longfat'))

data_dir <- '/home/flournoy/otherhome/data/splt/probly/'
if(!file.exists(data_dir)){
    stop('Data directory "', data_dir, '" does not exist')
} else {
    message("Data goes here: ", data_dir)
}

data(splt)
splt_no_na <- splt[!is.na(splt$pressed_r), ]
splt_no_na$proportion_fac <- factor(splt_no_na$proportion, levels = c('80_20', '20_80'))
optimal_side_mat <- get_col_as_trial_matrix(splt_no_na, col = 'proportion_fac')

mu_xi <- c(-2.7, -2.6, -2.4)
mu_b <- c(-.07, .28, .39)
mu_eps <- c(-1.65, -1.65 + .3, -1.65 + .2)
mu_rho <- c(-.3, -.3 + .35, -.3 + .45)

task_structure <- probly::make_task_structure_from_data(splt_no_na)
task_structure_stan <- task_structure #just to sub NA with -1
task_structure_stan$condition[is.na(task_structure_stan$condition)] <- -1
task_structure_stan$cue[is.na(task_structure_stan$cue)] <- -1

stan_sim_data <- list(
    N = task_structure_stan$N,
    `T` = max(task_structure_stan$Tsubj),
    K = task_structure_stan$K,
    M = task_structure_stan$M,
    ncue = task_structure_stan$n_cues,
    mm = task_structure_stan$mm,
    Tsubj = task_structure_stan$Tsubj,
    condition = task_structure_stan$condition,
    cue = task_structure_stan$cue
)

stan_rl_fn <- system.file('stan', 'splt_rl.stan', package = 'probly')

iter <- 100
results_f <- listenv()

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

        stanFit <- rstan::stan(file = stan_rl_fn,
                               data = stan_sim_data,
                               chains = 4, cores = 4,
                               iter = 1500, warmup = 1000)

        model_summary <- rstan::summary(stanFit)

        list(fit_summary = model_summary, coefs = learning_traj, data = task_data)
    }
}

results <- as.list(results_f)

saveRDS(results, file.path(data_dir, 'test_rez.RDS'))

