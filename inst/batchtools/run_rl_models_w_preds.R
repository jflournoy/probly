## ---- run_baseline_models

library(future)
library(future.batchtools)
library(listenv)
library(rstan)

if(grepl('(^n\\d|talapas-ln1)', system('hostname', intern = T))){
    niter <- 2000
    nchains <- 6
    niterperchain <- ceiling(niter/nchains)
    warmup <- 1200
    data_dir <- '/gpfs/projects/dsnlab/flournoy/data/splt/probly'
    plan(batchtools_slurm,
         template = system.file('batchtools', 'batchtools.slurm.tmpl', package = 'probly'),
         resources = list(ncpus = 6, walltime = 60*24*8, memory = '4G',
                          partitions = 'long,longfat'))
    AWS = F
} else if(grepl('^ip-', system('hostname', intern = T))) { #AWS
    niter <- 2000
    nchains <- 4
    niterperchain <- ceiling(niter/nchains)
    warmup <- 1250
    data_dir <- '/home/ubuntu/data'
    if(!any(grepl('WHICH_MOD', ls()))) stop("var WHICH_MOD must be set on AWS")
    AWS = T
    devtools::install_github('jflournoy/probly')
} else {
    data_dir <- '/data/jflournoy/split/probly'
    niter <- 1
    nchains <- 2
    warmup <- 1
    niterperchain <- ceiling(niter/nchains)
    plan(tweak(multiprocess, gc = T, workers = 8))
    AWS = F
}

library(probly)

if(!file.exists(data_dir)){
    stop('Data directory "', data_dir, '" does not exist')
} else {
    message("Data goes here: ", data_dir)
}

model_filename_list <- list(
    rl_2_level_preds = system.file('stan', 'splt_rl_2_level_w2lpreds.stan', package = 'probly')
)

pop_parlist <- c('mu_delta_ep', 'mu_delta_xi',
                 'tau_ep', 'tau_xi',
                 'L_Omega_xi', 'L_Omega_ep')
indiv_parlist <- c('beta_ep_prm', 'beta_xi_prm', 'pR_final', 'log_lik')

pop_parlist_rho <- c('mu_delta_rho', 'tau_rho', 'L_Omega_rho')
indiv_parlist_rho <- c('beta_rho_prm')

pop_parlist_b <- c('mu_delta_b', 'tau_b', 'L_Omega_b')
indiv_parlist_b <- c('beta_b')

save_pars_list <- list(
    rl_2_level_preds = c(pop_parlist, indiv_parlist, pop_parlist_rho, indiv_parlist_rho, pop_parlist_b, indiv_parlist_b)
)

if(AWS){
    append_to_data_fn <- names(model_filename_list)[[WHICH_MOD]]
    model_filename_list <- model_filename_list[WHICH_MOD] #one mod per instance here.
} else {
    append_to_data_fn <- ''
}

load(file.path(data_dir, 'data/splt.rda'))
load(file.path(data_dir, 'data/splt_dev_and_demog.rda'))
load(file.path(data_dir, 'data/splt_fsmi.rda'))

college_rubric_dir <- system.file('scoring_rubrics', 'college', package = 'probly')
ksrq_rubric <- scorequaltrics::get_rubrics(
    rubric_filenames = dplyr::data_frame(
        file = file.path(college_rubric_dir, 'K-SRQ_scoring_rubric.csv')),
    type = 'scoring')
ksrq_key <- scorequaltrics::create_key_from_rubric(ksrq_rubric)

splt_ksrq_mate_stat <- as.data.frame(psych::scoreItems(
    ksrq_key[c('k_srq_sexual_relationships', 'k_srq_admiration')],
    splt_fsmi, missing = TRUE, impute = 'none')$scores)
splt_ksrq_mate_stat$SID <- splt_fsmi$SID

print(dim(splt))
splt_no_na <- splt[!is.na(splt$pressed_r), ]
print(dim(splt_no_na))
splt_no_na_dev <- dplyr::left_join(
    splt_no_na,
    dplyr::distinct(splt_dev_and_demog,
                    SID, age, PDS_mean_score, gender),
    by = c('id' = 'SID'))
print(dim(splt_no_na_dev))
splt_no_na_dev_matestat <- dplyr::left_join(
    splt_no_na_dev,
    splt_ksrq_mate_stat,
    by = c('id' = 'SID'))
print(dim(splt_no_na_dev_matestat))
print(dim(dplyr::distinct(splt_no_na_dev_matestat, id)))
# print(dim(dplyr::distinct(dplyr::filter(
#     splt_no_na_dev_matestat,
#     !is.na(age), !is.na(PDS_mean_score),
#     !is.na(k_srq_sexual_relationships),
#     !is.na(k_srq_admiration)), id)))

splt <- splt_no_na_dev_matestat[!is.na(splt_no_na_dev_matestat$age) &
                                    !is.na(splt_no_na_dev_matestat$gender) &
                                    splt_no_na_dev_matestat$age < 30, ]

na_age <- is.na(unique(splt[,c('id', 'age')])$age)
na_gender <- is.na(unique(splt[,c('id', 'gender')])$gender)
na_pds <- is.na(unique(splt[,c('id', 'PDS_mean_score')])$PDS_mean_score)
age_sd <- sd(unique(splt[,c('id', 'age')])$age[unique(splt[,c('id', 'age')])$age < 30], na.rm = T)
age_mean <- mean(unique(splt[,c('id', 'age')])$age[unique(splt[,c('id', 'age')])$age < 30], na.rm = T)
pds_sd <- sd(as.numeric(unique(splt[,c('id', 'PDS_mean_score')])$PDS_mean_score), na.rm = T)
pds_mean <- mean(as.numeric(unique(splt[,c('id', 'PDS_mean_score')])$PDS_mean_score), na.rm = T)
splt$age_std <- (splt$age - age_mean)/age_sd
splt$pds_std <- (as.numeric(splt$PDS_mean_score) - pds_mean)/pds_sd
splt$gender_c <- as.numeric(splt$gender) - .5 #0 or -0.5 = male, 1 or .5 = female

gender_c <- get_col_as_trial_matrix(splt, 'gender_c', id_col = 'id', sample_col = 'sample', trial_col = 'trial_index')[,1]
age_std <- get_col_as_trial_matrix(splt, 'age_std', id_col = 'id', sample_col = 'sample', trial_col = 'trial_index')[,1]
age_x_gender <- age_std * gender_c

Xj <- model.matrix(~1+age_std+gender_c+age_x_gender)
J_pred <- dim(Xj)[2]

fit_many_mods_f <- listenv()

for(mod in 1:length(model_filename_list)){
    print(paste0('Sampling from model: ', names(model_filename_list)[mod]))
    fit_many_mods_f[[mod]]  %<-% {
        library(rstan)
        library(probly)

        dim(splt)
        splt_no_na <- splt

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
        outcome_r <- outcome_l <- outcome
        outcome_r[] <- outcome_l[] <- -1

        press_right[is.na(press_right)] <- -1
        press_opt[is.na(press_opt)] <- -1

        stan_data <- list(
            N = task_structure$N,
            `T` = max(task_structure$Tsubj),
            K = task_structure$K,
            M = task_structure$M,
            J = J_pred,
            ncue = task_structure$n_cues,
            mm = task_structure$mm,
            Tsubj = task_structure$Tsubj,
            condition = task_structure$condition,
            cue = task_structure$cue,
            press_right = press_right,
            outcome = outcome,
            outcome_r = outcome_r,
            outcome_l = outcome_l,
            u = Xj,
            run_estimation = 1
        )

        if(names(model_filename_list)[mod] == 'intercept_only'){
            stan_data$press_right <- NULL
            stan_data$press_opt <- press_opt
        }

        save_pars <- save_pars_list[[mod]]

        message('Model: ', names(model_filename_list)[[mod]])
        message('Save pars: ', save_pars)

        stanFit <- rstan::stan(file = model_filename_list[[mod]],
                               data = stan_data,
                               chains = nchains, cores = nchains,
                               iter = warmup + niterperchain, warmup = warmup,
                               pars = save_pars, include = TRUE,
                               control = list(max_treedepth = 15, adapt_delta = 0.99))

        savepath <- file.path(data_dir,
                              paste0('splt-preds-', names(model_filename_list)[mod],
                                     '-', round(as.numeric(Sys.time())/1000,0),'.RDS'))

        message('Saving fit to ', savepath)
        message('Size: ', format(object.size(stanFit), units = 'GB'))

        saveRDS(stanFit, savepath)
        list(complete = TRUE)
    }
}

saveRDS(splt,
        file.path(data_dir, paste0('splt-preds-data-', append_to_data_fn, '-',
                                   round(as.numeric(Sys.time())/1000,0),'.RDS')))
