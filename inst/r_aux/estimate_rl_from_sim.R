## ---- estimate_rl_from_sim

library(future)
library(rstan)

condition_mat_stan <- condition_mat
outcome_stan <- splt_sim_trials$outcome_realized
press_right_stan <- splt_sim_trials$press_right
cue_mat_stan <- cue_mat

condition_mat_stan[is.na(condition_mat_stan)] <- -1
outcome_stan[is.na(outcome_stan)] <- -1
press_right_stan[is.na(press_right_stan)] <- -1
cue_mat_stan[is.na(cue_mat_stan)] <- -1

stan_sim_data <- list(
    N = N,
    `T` = max(Tsubj),
    K = K,
    M = M,
    ncue = max(cue_mat, na.rm = T),
    mm = group_index_mm$m,
    Tsubj = Tsubj,
    condition = condition_mat_stan,
    outcome = outcome_stan,
    press_right = press_right_stan,
    cue = cue_mat_stan
)

stan_sim_fit_fn <- file.path(data_dir, 'splt_rl_fit_sim.RDS')
stan_rl_fn <- system.file('stan', 'splt_rl.stan', package = 'probly')
stan_sim_fit <- probly::CachedFit(
    {
        stanFit <- rstan::stan(file = stan_rl_fn,
                               data = stan_sim_data, chains = 4, iter = 1500,
                               warmup = 1000, cores = 4, open_progress = T)
        list(fit = stanFit, data = stan_sim_data)
    },
    rds_filename = stan_sim_fit_fn)

format(object.size(x = stan_sim_fit), units = 'MB')
