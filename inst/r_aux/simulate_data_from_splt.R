## ---- simulate_data_from_splt

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
