data {
    int<lower=1> N; //number of subjects
    int<lower=1> T; //max number of trials
    int<lower=2> K; //number of trial predictors (in this case, conditions)
    int<lower=1> M; //number of top-level groups
    int<lower=1> ncue; //number of cues
    int<lower=1, upper=M> mm[N]; //M-level ID for each subject
    int<lower=1, upper=T> Tsubj[N]; //trials per subject
    int<lower=-1,upper=3> condition[N,T]; //1 = ctrl, 2 = mateseeking, 3 = status

    real outcome[N, T]; //reward for the trial -- either 0, 1, or 5 points
    real outcome_r[N, T]; //for simulated data - reward for the trial -- either 0, 1, or 5 points
    real outcome_l[N, T]; //for simulated data - reward for the trial -- either 0, 1, or 5 points
    int<lower=-1, upper=1> press_right[N, T]; //choices "0" = left, "1" = right
    int<lower=-1, upper=ncue> cue[N, T]; //which cue appeared

    int<lower=0, upper=1> run_estimation; // a switch to evaluate the likelihood
}

transformed data {
    vector[ncue] initV;
    initV  = rep_vector(0.0, ncue);
}

parameters {
    matrix[K, N] z_xi;
    matrix[K, N] z_ep;
    matrix[K, N] z_b;
    matrix[K, N] z_rho;

    cholesky_factor_corr[K] L_Omega_xi;
    cholesky_factor_corr[K] L_Omega_ep;
    cholesky_factor_corr[K] L_Omega_b;
    cholesky_factor_corr[K] L_Omega_rho;

    vector<lower=0,upper=pi()/2>[K] tau_unif_xi;
    vector<lower=0,upper=pi()/2>[K] tau_unif_ep;
    vector<lower=0,upper=pi()/2>[K] tau_unif_b;
    vector<lower=0,upper=pi()/2>[K] tau_unif_rho;

    vector[K] mu_delta_xi;
    vector[K] mu_delta_ep;
    vector[K] mu_delta_b;
    vector[K] mu_delta_rho;

    real<lower=0> sigma_delta_xi;
    real<lower=0> sigma_delta_ep;
    real<lower=0> sigma_delta_b;
    real<lower=0> sigma_delta_rho;

    matrix[M, K] delta_xi_raw; //unit normals for matt trick on these params
    matrix[M, K] delta_ep_raw; //unit normals for matt trick on these params
    matrix[M, K] delta_b_raw; //unit normals for matt trick on these params
    matrix[M, K] delta_rho_raw; //unit normals for matt trick on these params
}

transformed parameters {
    matrix[M, K] delta_xi; //M-level intercepts for N-level (that is, sample-level for each subject)
    matrix[M, K] delta_ep; //M-level intercepts for N-level (that is, sample-level for each subject)
    matrix[M, K] delta_b; //M-level intercepts for N-level (that is, sample-level for each subject)
    matrix[M, K] delta_rho; //M-level intercepts for N-level (that is, sample-level for each subject)

    vector<lower=0>[K] tau_xi; // prior scale
    vector<lower=0>[K] tau_ep; // prior scale
    vector<lower=0>[K] tau_b; // prior scale
    vector<lower=0>[K] tau_rho; // prior scale

    matrix<lower=0, upper=1>[N, K] beta_xi_prm; //per-individual coefficients for xi, for each condition, transformed
    matrix<lower=0, upper=1>[N, K] beta_ep_prm; //per-individual coefficients for ep, for each condition, transformed
    matrix[N, K] beta_b; //per-individual coefficients for b, for each condition
    matrix<lower=0>[N, K] beta_rho_prm; //per-individual coefficients for rho, for each condition, transformed

    for(k in 1:K){
        for(m in 1:M){
            //implies delta_param[m,k] ~ normal(mu_delta_param[k], sigma_delta_param);
            delta_xi[m,k] = mu_delta_xi[k] + sigma_delta_xi * delta_xi_raw[m,k];
            delta_ep[m,k] = mu_delta_ep[k] + sigma_delta_ep * delta_ep_raw[m,k];
            delta_b[m,k] = mu_delta_b[k] + sigma_delta_b * delta_b_raw[m,k];
            delta_rho[m,k] = mu_delta_rho[k] + sigma_delta_rho * delta_rho_raw[m,k];
        }

        tau_xi[k] = 2.5 * tan(tau_unif_xi[k]);
        tau_ep[k] = 2.5 * tan(tau_unif_ep[k]);
        tau_b[k] = 2.5 * tan(tau_unif_b[k]);
        tau_rho[k] = 2.5 * tan(tau_unif_rho[k]);
    }

    beta_xi_prm  = Phi_approx(delta_xi[mm] + (diag_pre_multiply(tau_xi, L_Omega_xi) * z_xi)');
    beta_ep_prm  = Phi_approx(delta_ep[mm] + (diag_pre_multiply(tau_ep, L_Omega_ep) * z_ep)');
    beta_b   = delta_b[mm] + (diag_pre_multiply(tau_b, L_Omega_b) * z_b)';
    beta_rho_prm = exp(delta_rho[mm] + (diag_pre_multiply(tau_rho, L_Omega_rho) * z_rho)');
}

model {
    // gng_m2: RW + noise + bias model in Guitart-Masip et al 2012
    // hyper parameters
    mu_delta_xi  ~ normal(0, 1);
    mu_delta_ep  ~ normal(0, 1);
    mu_delta_b  ~ normal(0, 2);
    mu_delta_rho  ~ normal(0, 1);
    sigma_delta_xi ~ exponential(2.0);
    sigma_delta_ep ~ exponential(2.0);
    sigma_delta_b ~ exponential(2.0);
    sigma_delta_rho ~ exponential(2.0);

    // print("sigma_delta_xi = ", sigma_delta_xi);
    // print("sigma_delta_ep = ", sigma_delta_ep);
    // print("sigma_delta_b = ", sigma_delta_b);
    // print("sigma_delta_rho = ", sigma_delta_rho);

    //Sample-level parameters
    to_vector(delta_xi_raw) ~ normal(0, 1);
    to_vector(delta_ep_raw) ~ normal(0, 1);
    to_vector(delta_b_raw) ~ normal(0, 1);
    to_vector(delta_rho_raw) ~ normal(0, 1);

    //Sigma_param for individual level coefficients
    to_vector(z_xi) ~ normal(0, 1);
    to_vector(z_ep) ~ normal(0, 1);
    to_vector(z_b) ~ normal(0, 1);
    to_vector(z_rho) ~ normal(0, 1);
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
