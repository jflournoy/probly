// _reg: generates model-based regressors
data {
  int<lower=1> N; //number of subjects
  int<lower=1> T; //max number of trials
  int<lower=2> K; //number of trial predictors (in this case, conditions)
  // int<lower=1> L; //number if subject-level predictors.
  int<lower=1> M; //number of top-level groups
  int<lower=1, upper=M> mm[N]; //M-level ID for each subject
  int<lower=1, upper=T> Tsubj[N]; //trials per subject
  int<lower=-1,upper=1> press_right[N,T]; //choices "0" = left, "1" = right
  int<lower=-1,upper=3> condition[N,T]; //1 = ctrl, 2 = mateseeking, 3 = status
  // matrix[N, L] u; //N-level predictors. for adding, e.g., age and puberty
}

parameters {
  matrix[K, N] z;
  cholesky_factor_corr[K] L_Omega;
  vector<lower=0,upper=pi()/2>[K] tau_unif;
  // matrix[L, K] gamma; //group coefficients. gamma[1, ] are intercepts for each condition
  vector[K] mu_delta;
  real<lower=0> sigma_delta;
  matrix[M, K] delta; //M-level intercepts for N-level (that is, sample-level for each subject)
}

transformed parameters {
  matrix[N, K] beta; //per-subject coefficients for each K
  vector<lower=0>[K] tau; // prior scale
  for (k in 1:K) tau[k] = 2.5 * tan(tau_unif[k]);
  //{
  //  matrix[N, K] beta_raw; //per-subject coefficients for each K
  //  for (i in 1:N){
  //    beta_raw[i] = u[i] * gamma + delta[mm[i]];
  //  }
  // //Put calculations of delta and beta in here
  //}
  beta = delta[mm] + (diag_pre_multiply(tau,L_Omega) * z)';
}

model {
  to_vector(z) ~ normal(0, 1);
  L_Omega ~ lkj_corr_cholesky(2);
  // to_vector(gamma) ~ normal(0, 5);
  mu_delta ~ normal(0, 5);
  sigma_delta ~ exponential(1);
  for (k in 1:K) delta[,k] ~ normal(mu_delta[k], sigma_delta);

  for (i in 1:N) {
    for (t in 1:Tsubj[i]) {
      press_right[i, t] ~ bernoulli_logit(beta[i, condition[i, t]]);
    } // end of t loop
  } // end of i loop
}
