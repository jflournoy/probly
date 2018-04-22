## ---- load_data_for_sim

library(probly)
data(splt)
data(splt_dev_and_demog)
dim(splt)
splt <- dplyr::left_join(splt,
                  unique(splt_dev_and_demog[, c('SID', 'gender', 'PDS_mean_score', 'age')]),
                  by = c('id' = 'SID'))
splt <- splt[!is.na(splt$pressed_r), ]
splt <- splt[!is.na(splt$age) & !is.na(splt$gender) & splt$age < 30, ]

splt$cue <- as.numeric(as.factor(paste0(splt$condition, '_', splt$sex)))
splt$condition <- factor(splt$condition, levels = c('HngT', 'DtnL', 'PplU'))

sum(na_age <- is.na(unique(splt[,c('id', 'age')])$age))
sum(na_gender <- is.na(unique(splt[,c('id', 'gender')])$gender))
sum(na_pds <- is.na(unique(splt[,c('id', 'PDS_mean_score')])$PDS_mean_score))
age_sd <- sd(unique(splt[,c('id', 'age')])$age[unique(splt[,c('id', 'age')])$age < 30], na.rm = T)
age_mean <- mean(unique(splt[,c('id', 'age')])$age[unique(splt[,c('id', 'age')])$age < 30], na.rm = T)
pds_sd <- sd(as.numeric(unique(splt[,c('id', 'PDS_mean_score')])$PDS_mean_score), na.rm = T)
pds_mean <- mean(as.numeric(unique(splt[,c('id', 'PDS_mean_score')])$PDS_mean_score), na.rm = T)
splt$age_std <- (splt$age - age_mean)/age_sd
splt$pds_std <- (as.numeric(splt$PDS_mean_score) - pds_mean)/pds_sd
unique(splt$gender_c <- splt$gender - .5) #0 or -0.5 = male, 1 or .5 = female

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

group_index_mm <- get_sample_index(splt, levels = c("TDS1", "TDS2", "yads", "yads_online"))
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
