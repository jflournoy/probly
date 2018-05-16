## ---- load_data_for_sim

library(probly)
library(dplyr)
library(tidyr)

data(splt)
data(splt_dev_and_demog)
data(splt_confidence)
data(splt_fsmi)

college_rubric_dir <- system.file('scoring_rubrics', 'college', package = 'probly')
adolescent_rubric_dir <- system.file('scoring_rubrics', 'adolescent', package = 'probly')
relationship_status <- c('long_term' = 1, 'actively_dating' = 2, 'single_nodate' = 3, 'single_neverdate' = 4)

fsmi_rubric <- scorequaltrics::get_rubrics(
    rubric_filenames = data_frame(
        file = file.path(college_rubric_dir, 'fsmi_scoring_rubric.csv')),
    type = 'scoring')
dnp_rubric <- scorequaltrics::get_rubrics(
    rubric_filenames = data_frame(
        file = file.path(college_rubric_dir, 'DominanceAndPrestige_scoring_rubric.csv')),
    type = 'scoring')
ksqr_rubric <- scorequaltrics::get_rubrics(
    rubric_filenames = data_frame(
        file = file.path(college_rubric_dir, 'K-SRQ_scoring_rubric.csv')),
    type = 'scoring')
upps_rubric <- scorequaltrics::get_rubrics(
    rubric_filenames = data_frame(
        file = file.path(adolescent_rubric_dir, 'UPPSP_scoring_rubric.csv')),
    type = 'scoring')

fsmi_key <- scorequaltrics::create_key_from_rubric(fsmi_rubric)
dnp_key <- scorequaltrics::create_key_from_rubric(dnp_rubric)
ksqr_key <- scorequaltrics::create_key_from_rubric(ksqr_rubric)
ksrq_EFA_key <- list(m1 = c('K_SRQ_11', 'K_SRQ_18', 'K_SRQ_1', 'K_SRQ_7'),
                     m2 = c('K_SRQ_4', 'K_SRQ_20', 'K_SRQ_13', 'K_SRQ_9'))
upps_key <- scorequaltrics::create_key_from_rubric(upps_rubric)

self_report_motive_keys <- c(fsmi_key, dnp_key,
                             ksqr_key, ksrq_EFA_key,
                             upps_key)

fsmi_scored <- psych::scoreItems(keys = self_report_motive_keys,
                                 items = splt_fsmi,
                                 missing = TRUE, impute = 'none')

fsmi_scores_df <- as.data.frame(fsmi_scored$scores)
fsmi_scores_df$SID <- splt_fsmi$SID

splt <- dplyr::left_join(splt,
                  unique(splt_dev_and_demog[, c('SID', 'gender', 'PDS_mean_score', 'age')]),
                  by = c('id' = 'SID'))
splt <- dplyr::left_join(splt,
                         fsmi_scores_df,
                         by = c('id' = 'SID'))

sample_labels <- c(
    'TDS1' = 'Foster-care involved adolescents',
    'TDS2' = 'Community adolescents',
    'yads' = 'College students',
    'yads_online' = 'College students - online'
)

condition_labels <- c(
    'HngT' = 'Hungry/Thirsty',
    'DtnL' = 'Dating/Looking',
    'PplU' = 'Popular/Unpopular'
)

splt$condition <- factor(splt$condition, levels = names(condition_labels), labels = condition_labels)
splt$sample <- factor(splt$sample, levels = names(sample_labels), labels = sample_labels)

splt$gender <- factor(splt$gender, levels = c(0, 1), labels = c('Male', 'Female'))

splt_confidence$sample <- factor(splt_confidence$sample, levels = names(sample_labels), labels = sample_labels)

splt <- splt[!is.na(splt$pressed_r), ]
splt_orig <- splt
splt <- splt[!is.na(splt$age) & !is.na(splt$gender) & splt$age < 30, ]

splt$opt_is_right <- as.numeric(factor(splt$proportion,
                                             levels = c('80_20', '20_80'))) - 1
splt$press_opt <- as.numeric(splt$opt_is_right == splt$pressed_r)

splt$cue <- as.numeric(as.factor(paste0(splt$condition, '_', splt$sex)))

na_age <- is.na(unique(splt[,c('id', 'age')])$age)
na_gender <- is.na(unique(splt[,c('id', 'gender')])$gender)
na_pds <- is.na(unique(splt[,c('id', 'PDS_mean_score')])$PDS_mean_score)
age_sd <- sd(unique(splt[,c('id', 'age')])$age[unique(splt[,c('id', 'age')])$age < 30], na.rm = T)
age_mean <- mean(unique(splt[,c('id', 'age')])$age[unique(splt[,c('id', 'age')])$age < 30], na.rm = T)
pds_sd <- sd(as.numeric(unique(splt[,c('id', 'PDS_mean_score')])$PDS_mean_score), na.rm = T)
pds_mean <- mean(as.numeric(unique(splt[,c('id', 'PDS_mean_score')])$PDS_mean_score), na.rm = T)
splt$age_std <- (splt$age - age_mean)/age_sd
splt$pds_std <- (as.numeric(splt$PDS_mean_score) - pds_mean)/pds_sd
splt$gender_c <- as.numeric(splt$gender) - 1.5 #0 or -0.5 = male, 1 or .5 = female

splt$trial_index_c0_s <-
    (splt$trial_index - max(splt$trial_index)) /
    sd(splt$trial_index)

splt$condition_trial_index_c0_s <-
    (splt$condition_trial_index - max(splt$condition_trial_index)) /
    sd(splt$condition_trial_index)

# - N number of individuals
# - M number of samples
# - K number of conditions
# - mm sample ID for all individuals
# - Tsubj number of trials for each individual
# - cue an N x max(Tsubj) matrix of cue IDs for
#   each trial
# - n_cues total number of cues
# - condtion an N x max(Tsubj) matrix of condition
#   IDs for each trial
# - outcome is an array with dimensions N x T x 2
#   (response options) with the feedback for each
#   possible response. outcome[,,1] is for
#   correct left-presses, and outcome[,,2] is for
#   correct right-presses.
# - beta_xi, beta_b, beta_eps, beta_rho are N x K
#   matrices of the individually varying parameter
#   coefficients

group_index_mm <- get_sample_index(splt, levels = sample_labels)
N <- dim(group_index_mm)[1]
M <- length(levels(group_index_mm$m_fac))
K <- length(unique(splt$condition))
cue_mat <- get_col_as_trial_matrix(splt, 'cue', id_col = 'id', sample_col = 'sample', trial_col = 'trial_index')
Tsubj <- get_max_trials_per_individual(cue_mat)
condition_mat <- get_col_as_trial_matrix(splt, 'condition', id_col = 'id', sample_col = 'sample', trial_col = 'trial_index')
correct_r_mat <- get_col_as_trial_matrix(splt, 'correct_r', id_col = 'id', sample_col = 'sample', trial_col = 'trial_index')
reward_possible_mat <- get_col_as_trial_matrix(splt, 'reward_possible', id_col = 'id', sample_col = 'sample', trial_col = 'trial_index')
outcome_arr <- array(reward_possible_mat, dim = c(dim(reward_possible_mat), 2))
outcome_arr[,,1][correct_r_mat == 1] <- 0 #reward if left press, but right is correct = 0
outcome_arr[,,2][correct_r_mat == 0] <- 0 #reward if right press, but correct is not right = 0

gender_c <- get_col_as_trial_matrix(splt, 'gender_c', id_col = 'id', sample_col = 'sample', trial_col = 'trial_index')[,1]
age_std <- get_col_as_trial_matrix(splt, 'age_std', id_col = 'id', sample_col = 'sample', trial_col = 'trial_index')[,1]
age_x_gender <- age_std * gender_c

Xj <- cbind(age_std, gender_c, age_x_gender)
J_pred <- dim(Xj)[2]
