data {
  int<lower=1> N; //number of subjects
  int<lower=1> T; //max number of trials
  int<lower=2> K; //number of trial predictors (in this case, conditions)
  int<lower=1> M; //number of top-level groups
  int<lower=1> ncue; //number of cues
  int<lower=1, upper=M> mm[N]; //M-level ID for each subject
  int<lower=1, upper=T> Tsubj[N]; //trials per subject
  int<lower=-1,upper=3> condition[N,T]; //1 = ctrl, 2 = mateseeking, 3 = status

  real outcome[N, T];
  int<lower=-1, upper=1> press_right[N, T]; //choices "0" = left, "1" = right
  int<lower=-1, upper=ncue> cue[N, T];
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

  matrix[M, K] delta_xi; //M-level intercepts for N-level (that is, sample-level for each subject)
  matrix[M, K] delta_ep; //M-level intercepts for N-level (that is, sample-level for each subject)
  matrix[M, K] delta_b; //M-level intercepts for N-level (that is, sample-level for each subject)
  matrix[M, K] delta_rho; //M-level intercepts for N-level (that is, sample-level for each subject)
}

transformed parameters {
  matrix<lower=0, upper=1>[N, K] beta_xi_prm; //per-individual coefficients for xi, for each condition, transformed
  matrix<lower=0, upper=1>[N, K] beta_ep_prm; //per-individual coefficients for ep, for each condition, transformed
  matrix[N, K] beta_b; //per-individual coefficients for b, for each condition
  matrix<lower=0>[N, K] beta_rho_prm; //per-individual coefficients for rho, for each condition, transformed
  vector<lower=0>[K] tau_xi; // prior scale
  vector<lower=0>[K] tau_ep; // prior scale
  vector<lower=0>[K] tau_b; // prior scale
  vector<lower=0>[K] tau_rho; // prior scale

  for (k in 1:K){
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
  mu_delta_xi  ~ normal(0, 5.0);
  mu_delta_ep  ~ normal(0, 5.0);
  mu_delta_b  ~ normal(0, 10.0);
  mu_delta_rho  ~ normal(0, 5.0);
  sigma_delta_xi ~ exponential(1);
  sigma_delta_ep ~ exponential(1);
  sigma_delta_b ~ exponential(1);
  sigma_delta_rho ~ exponential(1);

  for(k in 1:K){
    delta_xi[,k] ~ normal(mu_delta_xi[k], sigma_delta_xi);
    delta_ep[,k] ~ normal(mu_delta_ep[k], sigma_delta_ep);
    delta_b[,k] ~ normal(mu_delta_b[k], sigma_delta_b);
    delta_rho[,k] ~ normal(mu_delta_rho[k], sigma_delta_rho);
  }

  //Sigma_param
  to_vector(z_xi) ~ normal(0, 1);
  to_vector(z_ep) ~ normal(0, 1);
  to_vector(z_b) ~ normal(0, 1);
  to_vector(z_rho) ~ normal(0, 1);
  L_Omega_xi ~ lkj_corr_cholesky(2);
  L_Omega_ep ~ lkj_corr_cholesky(2);
  L_Omega_b ~ lkj_corr_cholesky(2);
  L_Omega_rho ~ lkj_corr_cholesky(2);


  for (i in 1:N) {
    vector[ncue] wv_r;  // action wegith for go
    vector[ncue] wv_l; // action wegith for nogo
    vector[ncue] qv_r;  // Q value for go
    vector[ncue] qv_l; // Q value for nogo
    vector[ncue] pR;   // prob of go (press)

    wv_r  = initV;
    wv_l = initV;
    qv_r  = initV;
    qv_l = initV;

    for (t in 1:Tsubj[i]) {
      wv_r[cue[i, t]]  = qv_r[cue[i, t]] + beta_b[i, condition[i, t]];
      wv_l[cue[i, t]] = qv_l[cue[i, t]];  // qv_l is always equal to wv_l (regardless of action)
      pR[cue[i, t]]   = inv_logit(wv_r[cue[i, t]] - wv_l[cue[i, t]]);
      pR[cue[i, t]]   = pR[cue[i, t]] * (1 - beta_xi_prm[i, condition[i, t]]) + beta_xi_prm[i, condition[i, t]]/2;  // noise
      press_right[i, t] ~ bernoulli(pR[cue[i, t]]);

      // update action values
      if (press_right[i, t]) { // update go value
        qv_r[cue[i, t]]  = qv_r[cue[i, t]] + beta_ep_prm[i, condition[i, t]] * (beta_rho_prm[i, condition[i, t]] * outcome[i, t] - qv_r[cue[i, t]]);
      } else { // update no-go value
        qv_l[cue[i, t]] = qv_l[cue[i, t]] + beta_ep_prm[i, condition[i, t]] * (beta_rho_prm[i, condition[i, t]] * outcome[i, t] - qv_l[cue[i, t]]);
      }
    } // end of t loop
  } // end of i loop
}
