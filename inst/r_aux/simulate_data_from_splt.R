## ---- simulate_data_from_splt

library(probly)
data(splt)

splt <- splt[!is.na(splt$pressed_r), ]
splt$cue <- as.numeric(as.factor(paste0(splt$condition, '_', splt$sex)))
splt$condition <- factor(splt$condition, levels = c('HngT', 'DtnL', 'PplU'))

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
outcome_arr[,,1][correct_r_mat == 1] <- 0
outcome_arr[,,2][correct_r_mat == 0] <- 0

set.seed(99232486)

mu_xi <- rnorm(3, rep(-2.5, 3), .25)
mu_b <- rnorm(3, rep(0, 3), .25)
mu_eps <- c(-1.65, -1.65 + .3, -1.65 + .2)
mu_rho <- c(-.3, -.3 + .35, -.3 + .45)

delta_xi <- sample_deltas(mu_xi, .1, 4)
delta_b <- sample_deltas(mu_b, .1, 4)
delta_eps <- sample_deltas(mu_eps, .1, 4)
delta_rho <- sample_deltas(mu_rho, .1, 4)

Sigma <- matrix(rep(.1, 3*3), nrow = 3)
diag(Sigma) <- 1

Sigma_eps <- Sigma_rho <- Sigma
Sigma_b <- Sigma_xi <- diag(3)

beta_xi <- sample_betas(deltas = delta_xi,
                        group_index = group_index_mm$m,
                        Sigma = Sigma_xi)
beta_b <- sample_betas(deltas = delta_b,
                       group_index = group_index_mm$m,
                       Sigma = Sigma_b)
beta_eps <- sample_betas(deltas = delta_eps,
                         group_index = group_index_mm$m,
                         Sigma = Sigma_eps)
beta_rho <- sample_betas(deltas = delta_rho,
                         group_index = group_index_mm$m,
                         Sigma = Sigma_rho)

splt_sim_trials <- generate_responses(
    N = N, M = M, K = K, mm = group_index_mm$m, Tsubj = Tsubj,
    cue = cue_mat, n_cues = max(cue_mat, na.rm = T), condition = condition_mat, outcome = outcome_arr,
    beta_xi = beta_xi, beta_b = beta_b, beta_eps = beta_eps, beta_rho = beta_rho)
