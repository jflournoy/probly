data {
    int<lower=1> N; //number of subjects
    int<lower=1> T; //max number of trials
    int<lower=2> K; //number conditions
    int<lower=1> ncue; //number of cues
    int<lower=1, upper=T> Tsubj[N]; //trials per subject
    int<lower=-1,upper=3> condition[N,T]; //Conditions 1-3

    real outcome[N, T]; //reward for the trial -- either 0, 1, or 5 points
    real outcome_r[N, T]; //for simulated data - reward for the trial -- either 0, 1, or 5 points
    real outcome_l[N, T]; //for simulated data - reward for the trial -- either 0, 1, or 5 points
    int<lower=-1, upper=1> press_right[N, T]; //choices "0" = left, "1" = right
    int<lower=-1, upper=ncue> cue[N, T]; //which cue appeared

    int<lower=0, upper=1> run_estimation; // a switch to evaluate the likelihood
}

transformed data {
    vector[ncue] initV;
    matrix[N,1] u; //just the intercept for each parameter
    initV  = rep_vector(0.0, ncue);
    u[,1] = rep_vector(1.0, N);
}

parameters {
    matrix[K, N] z_xi; //deviations from separate intercept for each condition, k in 1:K
    matrix[K, N] z_ep;
    matrix[K, N] z_b;
    matrix[K, N] z_rho;

    cholesky_factor_corr[K] L_Omega_xi; //for the correlation among intercepts across conditions
    cholesky_factor_corr[K] L_Omega_ep;
    cholesky_factor_corr[K] L_Omega_b;
    cholesky_factor_corr[K] L_Omega_rho;

    vector<lower=0>[K] tau_xi; //scaling of the correlations
    vector<lower=0>[K] tau_ep;
    vector<lower=0>[K] tau_b;
    vector<lower=0>[K] tau_rho;

    matrix[1,K] mu_delta_xi; //overall mean for each condition
    matrix[1,K] mu_delta_ep;
    matrix[1,K] mu_delta_b;
    matrix[1,K] mu_delta_rho;
}

transformed parameters {
    matrix<lower=0, upper=1>[N, K] beta_xi_prm; //per-individual coefficients for xi, for each condition, transformed
    matrix<lower=0, upper=1>[N, K] beta_ep_prm; //per-individual coefficients for ep, for each condition, transformed
    matrix[N, K] beta_b; //per-individual coefficients for b, for each condition
    matrix<lower=0>[N, K] beta_rho_prm; //per-individual coefficients for rho, for each condition, transformed

    beta_xi_prm = Phi_approx(u * mu_delta_xi + (diag_pre_multiply(tau_xi, L_Omega_xi) * z_xi)');
    beta_ep_prm = Phi_approx(u * mu_delta_ep + (diag_pre_multiply(tau_ep, L_Omega_ep) * z_ep)');
    beta_b = u * mu_delta_b + (diag_pre_multiply(tau_b, L_Omega_b) * z_b)';
    beta_rho_prm = exp(u * mu_delta_rho + (diag_pre_multiply(tau_rho, L_Omega_rho) * z_rho)');
}

model {
    // gng_m2: RW + noise + bias model in Guitart-Masip et al 2012
    // hyper parameters
    to_vector(mu_delta_xi)  ~ normal(0, 1);
    to_vector(mu_delta_ep)  ~ normal(0, 1);
    to_vector(mu_delta_b)  ~ normal(0, 2);
    to_vector(mu_delta_rho)  ~ normal(0, 1);

    //Sigma_param for individual level coefficients
    to_vector(z_xi) ~ normal(0, 1);
    to_vector(z_ep) ~ normal(0, 1);
    to_vector(z_b) ~ normal(0, 1);
    to_vector(z_rho) ~ normal(0, 1);
    tau_xi ~ exponential(1);
    tau_ep ~ exponential(1);
    tau_rho ~ exponential(1);
    tau_b ~  exponential(1);
    L_Omega_xi ~ lkj_corr_cholesky(2);
    L_Omega_ep ~ lkj_corr_cholesky(2);
    L_Omega_b ~ lkj_corr_cholesky(2);
    L_Omega_rho ~ lkj_corr_cholesky(2);

    if(run_estimation == 1){
        for (i in 1:N) {
            vector[ncue] wv_r;  // action wegith for go
            vector[ncue] wv_l; // action wegith for nogo
            vector[ncue] qv_r;  // Q value for go
            vector[ncue] qv_l; // Q value for nogo
            vector[ncue] pR;   // prob of go (press)
            real beta_xi_it;
            real beta_ep_it;
            real beta_rho_it;
            real beta_b_it;

            wv_r  = initV;
            wv_l = initV;
            qv_r  = initV;
            qv_l = initV;

            for (t in 1:Tsubj[i]) {
                beta_xi_it = beta_xi_prm[i, condition[i, t]];
                beta_ep_it = beta_ep_prm[i, condition[i, t]];
                beta_rho_it = beta_rho_prm[i, condition[i, t]];
                beta_b_it = beta_b[i, condition[i, t]];

                wv_r[cue[i, t]]  = qv_r[cue[i, t]] + beta_b_it;
                wv_l[cue[i, t]] = qv_l[cue[i, t]];  // qv_l is always equal to wv_l (regardless of action)
                pR[cue[i, t]]   = inv_logit(wv_r[cue[i, t]] - wv_l[cue[i, t]]);
                pR[cue[i, t]]   = pR[cue[i, t]] * (1 - beta_xi_it) + beta_xi_it/2;  // noise

                press_right[i, t] ~ bernoulli(pR[cue[i, t]]);

                // update action values
                if (press_right[i, t]) { // update go value
                    qv_r[cue[i, t]]  = qv_r[cue[i, t]] + beta_ep_it * (beta_rho_it * outcome[i, t] - qv_r[cue[i, t]]);
                } else { // update no-go value
                    qv_l[cue[i, t]] = qv_l[cue[i, t]] + beta_ep_it * (beta_rho_it * outcome[i, t] - qv_l[cue[i, t]]);
                }
            } // end of t loop
        } // end of i loop
    } // end estimate check
}

generated quantities {
    vector[N] log_lik;
    int<lower=-1, upper=1> pright_pred[N, T]; //choices "0" = left, "1" = right

    //save final pR
    vector[ncue] pR_final[N];

    for (i in 1:N) {
        for (t in 1:T) {
            pright_pred[i, t] = -1;
        }
    }

    { //local to save space and time
    for (i in 1:N) {
        vector[ncue] wv_r;  // action wegith for go
        vector[ncue] wv_l; // action wegith for nogo
        vector[ncue] qv_r;  // Q value for go
        vector[ncue] qv_l; // Q value for nogo
        vector[ncue] pR;   // prob of go (press)
        real beta_xi_it;
        real beta_ep_it;
        real beta_rho_it;
        real beta_b_it;
        vector[T] log_lik_iters;

        wv_r  = initV;
        wv_l = initV;
        qv_r  = initV;
        qv_l = initV;

        log_lik[i] = 0;
        log_lik_iters = rep_vector(0, T);

        for (t in 1:Tsubj[i]) {
            //neater code
            beta_xi_it = beta_xi_prm[i, condition[i, t]];
            beta_ep_it = beta_ep_prm[i, condition[i, t]];
            beta_rho_it = beta_rho_prm[i, condition[i, t]];
            beta_b_it = beta_b[i, condition[i, t]];

            wv_r[cue[i, t]]  = qv_r[cue[i, t]] + beta_b_it;
            wv_l[cue[i, t]] = qv_l[cue[i, t]];  // qv_l is always equal to wv_l (regardless of action)
            pR[cue[i, t]]   = inv_logit(wv_r[cue[i, t]] - wv_l[cue[i, t]]);
            pR[cue[i, t]]   = pR[cue[i, t]] * (1 - beta_xi_it) + beta_xi_it/2;  // noise

            pright_pred[i, t] = bernoulli_rng(pR[cue[i, t]]);

            if(run_estimation == 1){
                log_lik_iters[t] = bernoulli_lpmf(press_right[i, t] | pR[cue[i, t]]);

                // update action values
                if (press_right[i, t]) { // update go value
                    qv_r[cue[i, t]]  = qv_r[cue[i, t]] + beta_ep_it * (beta_rho_it * outcome[i, t] - qv_r[cue[i, t]]);
                } else { // update no-go value
                    qv_l[cue[i, t]] = qv_l[cue[i, t]] + beta_ep_it * (beta_rho_it * outcome[i, t] - qv_l[cue[i, t]]);
                }
            } else {
                log_lik_iters[t] = bernoulli_lpmf(pright_pred[i, t] | pR[cue[i, t]]);

                if (pright_pred[i, t]) { // update go value
                    qv_r[cue[i, t]]  = qv_r[cue[i, t]] + beta_ep_it * (beta_rho_it * outcome_r[i, t] - qv_r[cue[i, t]]);
                } else { // update no-go value
                    qv_l[cue[i, t]] = qv_l[cue[i, t]] + beta_ep_it * (beta_rho_it * outcome_l[i, t] - qv_l[cue[i, t]]);
                }
            }
        } // end of t loop
        log_lik[i] = sum(log_lik_iters);
        pR_final[i] = pR;
    } // end of i loop
    }
}
