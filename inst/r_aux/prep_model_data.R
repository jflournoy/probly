make_stan_data <- function(subset = FALSE){
    library(tidyverse)
    raw_data <- '~/data/probly/data/'
    load(file.path(raw_data, 'splt.rda'))
    load(file.path(raw_data, 'splt_dev_and_demog.rda'))
    load(file.path(raw_data, 'splt_fsmi.rda'))

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
        dplyr::distinct(ungroup(splt_dev_and_demog),
                        SID, age, PDS_mean_score, gender),
        by = c('id' = 'SID'))
    #should be the same length, bigger width
    print(dim(splt_no_na_dev))

    splt_no_na_dev_matestat <- dplyr::left_join(
        splt_no_na_dev,
        splt_ksrq_mate_stat,
        by = c('id' = 'SID'))
    print(dim(splt_no_na_dev_matestat))
    print(dim(dplyr::distinct(splt_no_na_dev_matestat, id)))
    print(dim(dplyr::distinct(dplyr::filter(
        splt_no_na_dev_matestat,
        !is.na(age), !is.na(PDS_mean_score),
        !is.na(k_srq_sexual_relationships),
        !is.na(k_srq_admiration)), id)))

    dim(splt_no_na_dev_matestat)

    if(subset){
        #SUBSET
        id_names <- unique(splt_no_na_dev_matestat[, c('id', 'sample')])
        id_names <- id_names[sample(dim(id_names)[[1]], subset), ]
        splt_mod_data <- left_join(id_names, splt_no_na_dev_matestat)
    } else {
        splt_mod_data <- splt_no_na_dev_matestat
    }
    #Vector of as.numeric(logical) about whether the optimal press is on the right (->) side
    splt_mod_data$opt_is_right <- as.numeric(factor(splt_mod_data$proportion,
                                                    levels = c('80_20', '20_80'))) - 1
    #Vector of whether the optimal answer was chosen
    splt_mod_data$press_opt <- as.numeric(splt_mod_data$opt_is_right == splt_mod_data$pressed_r)

    #creates the variables we need for Stan
    task_structure <- probly::make_task_structure_from_data(splt_mod_data)

    #some folks don't have every trial, in which case the condition and cue is NA. Stan
    #doesn't like NA so we set to -1 which will throw an error if it is included in
    #the sampling (it shouldn't be included)
    task_structure$condition[is.na(task_structure$condition)] <- -1
    task_structure$cue[is.na(task_structure$cue)] <- -1

    outcome <- probly::get_col_as_trial_matrix(
        splt_mod_data,
        col = 'outcome', id_col = 'id',
        sample_col = 'sample', trial_col = 'trial_index')

    press_right <- probly::get_col_as_trial_matrix(
        splt_mod_data,
        col = 'pressed_r', id_col = 'id',
        sample_col = 'sample', trial_col = 'trial_index')

    press_opt <- probly::get_col_as_trial_matrix(
        splt_mod_data,
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
        ncue = task_structure$n_cues,
        mm = task_structure$mm,
        Tsubj = task_structure$Tsubj,
        condition = task_structure$condition,
        cue = task_structure$cue,
        press_right = press_right,
        outcome = outcome,
        outcome_r = outcome_r,
        outcome_l = outcome_l,
        run_estimation = 1
    )
    return(stan_data)
}
