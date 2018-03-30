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

run_stan_f %<-% {
    library(rstan)
    data(splt)
    splt_no_na <- splt[!is.na(splt$pressed_r), ]

    task_structure <- probly::make_task_structure_from_data(splt_no_na)
    task_structure$condition[is.na(task_structure$condition)] <- -1
    task_structure$cue[is.na(task_structure$cue)] <- -1

    outcome <- probly::get_col_as_trial_matrix(
        splt_no_na,
        col = 'outcome', id_col = 'id',
        sample_col = 'sample', trial_col = 'trial_index')

    press_right <- probly::get_col_as_trial_matrix(
        splt_no_na,
        col = 'pressed_r', id_col = 'id',
        sample_col = 'sample', trial_col = 'trial_index')

    outcome[is.na(outcome)] <- -1
    press_right[is.na(press_right)] <- -1

    stan_data <- list(
        N = task_structure$N,
        `T` = max(task_structure$Tsubj),
        K = task_structure$K,
        M = task_structure$M,
        ncue = task_structure$n_cues,
        mm = task_structure$mm,
        Tsubj = task_structure$Tsubj,
        condition = task_structure$condition,
        cue = task_structure$cue,
        press_right = press_right,
        outcome = outcome
    )

    stan_rl_fn <- system.file('stan', 'splt_rl_reparam.stan', package = 'probly')

    stanFit <- rstan::stan(file = stan_rl_fn,
                           data = stan_data,
                           chains = 6, cores = 6,
                           iter = 1750, warmup = 1000,
                           control = list(max_treedepth = 15, adapt_delta = 0.99))

    saveRDS(stanFit, file.path(data_dir, 'stan_fit_baseline_model.RDS'))
}


