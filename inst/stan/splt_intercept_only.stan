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
  int<lower=-1, upper=1> press_opt[N, T]; //choices "0" = non-opt, "1" = optimal
  int<lower=-1, upper=ncue> cue[N, T];
}

transformed data {
  vector[ncue] initV;
  initV  = rep_vector(0.0, ncue);
}

parameters {
  matrix[K, N] z_theta;

  cholesky_factor_corr[K] L_Omega_theta;

  vector<lower=0,upper=pi()/2>[K] tau_unif_theta;

  vector[K] mu_delta_theta;

  real<lower=0> sigma_delta_theta;

  matrix[M, K] delta_theta_raw; //unit normals for matt trick on these params
}

transformed parameters {
  matrix[M, K] delta_theta; //M-level intercepts for N-level (that is, sample-level for each subject)

  vector<lower=0>[K] tau_theta; // prior scale

  matrix[N, K] beta_theta; //per-individual coefficients for xi, for each condition, transformed

  for(k in 1:K){
    for(m in 1:M){
      //implies delta_param[m,k] ~ normal(mu_delta_param[k], sigma_delta_param);
      delta_theta[m,k] = mu_delta_theta[k] + sigma_delta_theta * delta_theta_raw[m,k];
    }

    tau_theta[k] = 2.5 * tan(tau_unif_theta[k]);
  }

  beta_theta  = delta_theta[mm] + (diag_pre_multiply(tau_theta, L_Omega_theta) * z_theta)';
}

model {
// gng_m2: RW + noise + bias model in Guitart-Masip et al 2012
  // hyper parameters
  mu_delta_theta  ~ normal(0, 5.0);
  sigma_delta_theta ~ cauchy(0, 1.0);

//Sample-level parameters
  to_vector(delta_theta_raw) ~ normal(0, 1);

//Sigma_param for individual level coefficients
  to_vector(z_theta) ~ normal(0, 1);
  L_Omega_theta ~ lkj_corr_cholesky(2);


  for (i in 1:N) {
    for (t in 1:Tsubj[i]) {
      press_opt[i, t] ~ bernoulli_logit(beta_theta[i, condition[i, t]]);
    } // end of t loop
  } // end of i loop
}
