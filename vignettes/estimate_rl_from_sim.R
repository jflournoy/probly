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

stan_fit_sim <- '/data/jflournoy/split/probly/splt_rl_fit_sim.RDS'
if(!file.exists(stan_fit_sim)){
    plan(multiprocess)
    stan_sim_m <- rstan::stan_model(file = '../exec/splt_rl.stan')
    stan_optim_sim <- rstan::optimizing(stan_sim_m,
                                        data = stan_sim_data)
    round(stan_optim_sim$par[grep('delta', names(stan_optim_sim$par))],3)
    stan_sim_fit_f <- future({rstan::stan(file = '../exec/splt_rl.stan',
                                          data = stan_sim_data, chains = 4, iter = 1500,
                                          warmup = 1000, cores = 4, open_progress = T)})
    stan_sim_fit <- value(stan_sim_fit_f)
    saveRDS(stan_sim_fit, stan_fit_sim)
} else {
    stan_sim_fit <- readRDS(stan_fit_sim)
}
