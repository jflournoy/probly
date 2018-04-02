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
    intercept_only = system.file('stan', 'splt_intercept_only.stan', package = 'probly'),
    rl_2_level = system.file('stan', 'splt_rl_2_level.stan', package = 'probly'),
    rl_2_level_no_b = system.file('stan', 'splt_rl_2_level_no_b.stan', package = 'probly'),
    rl_repar_exp = system.file('stan', 'splt_rl_reparam_exp.stan', package = 'probly'),
    rl_repar_exp_no_b = system.file('stan', 'splt_rl_reparam_exp_no_b.stan', package = 'probly')
)

optim_many_mods_f <- listenv()

for(mod in 1:5){
    print(paste0('Optimizing model: ', names(model_filename_list)[mod]))
    optim_many_mods_f[[mod]] %<-% {
        library(rstan)
        library(probly)
        data(splt)
        print(dim(splt))
        splt_no_na <- splt[!is.na(splt$pressed_r), ]

        splt_no_na$opt_is_right <- as.numeric(factor(splt_no_na$proportion,
                                                     levels = c('80_20', '20_80'))) - 1
        splt_no_na$press_opt <- as.numeric(splt_no_na$opt_is_right == splt_no_na$pressed_r)

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

        press_opt <- probly::get_col_as_trial_matrix(
            splt_no_na,
            col = 'press_opt', id_col = 'id',
            sample_col = 'sample', trial_col = 'trial_index')

        outcome[is.na(outcome)] <- -1
        press_right[is.na(press_right)] <- -1
        press_opt[is.na(press_opt)] <- -1

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

        if(names(model_filename_list)[mod] == 'intercept_only'){
            stan_data$press_right <- NULL
            stan_data$press_opt <- press_opt
        }

        stanFit <- rstan::stan(file = model_filename_list[[mod]],
                               data = stan_data,
                               chains = 6, cores = 6,
                               iter = 1750, warmup = 1000,
                               control = list(max_treedepth = 15, adapt_delta = 0.99))

        saveRDS(stanFit,
                file.path(data_dir,
                          paste0('splt-', names(model_filename_list)[mod],
                                 '-', round(as.numeric(Sys.time())/1000,0),'.RDS')))
    }
}

