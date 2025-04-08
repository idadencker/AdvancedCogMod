// Weighted Beta-Binomial Stan model

// Bayesian integration model relying on a beta-binomial distribution
// to preserve all uncertainty

// The input (data) for the model 
data {
 int<lower=1> n; // n is trials
 array[n] int f; // f is the first rating
 array[n] int g; // g is the group rating
 array[n] int s; // s is the second rating
 // array[n] int fe; // fe is the feedback
 // array[n] int c; // c is the change 
}

// non-scaled
// trust = f // B1
// nontrust = 8-f //R1

// group_trust = g // B2
// group_nontrust = 8-g //R2

// alpha = trust + group_trust
// beta = non_trust + group_nontrust

// Scaling / transforming data, so the rating go from 0-7 instead of 1-8 to fit with the beta-binomial distribution
// We also add the total, which we will need later
// if it doesn't work, maybe add a for-loop
transformed data {
  array[n] int scale_f;
  array[n] int scale_g;
  array[n] int scale_s;
  int total = 7;

// We do this for-loop because STAN is not able to do scale_f = f - 1
for (i in 1:n) {
  scale_f[i] = f[i] - 1;
  scale_g[i] = g[i] - 1;
  scale_s[i] = s[i] - 1;
}
}

// Paramaters 
parameters {
  real<lower=0> total_weight;         // Total influence of all evidence
  real<lower=0, upper=1> weight_prop; // Proportion of weight for direct evidence
}


transformed parameters {
  real<lower=0> weight_direct = total_weight * weight_prop;
  real<lower=0> weight_social = total_weight * (1 - weight_prop);
}


model {
  // Each observation is a separate decision
  for (i in 1:n) {
    // Calculate Beta parameters for posterior belief distribution
    real alpha_post = 1 + weight_direct* scale_f[i] + weight_social*scale_g[i];
    real beta_post = 1 + weight_direct* (total - scale_f[i]) + weight_social*(total - scale_g[i]);
    
    // Use beta_binomial distribution which integrates over all possible values
    // of the rate parameter weighted by their posterior probability
    // the seven means that the outcome scale is 0-7
    target += beta_binomial_lpmf(scale_s[i] | total, alpha_post, beta_post);
  }
}


generated quantities {
  // Log likelihood for model comparison
  vector[n] log_lik;
  
  //save the priors here 
  
  // Prior and posterior predictive checks
  array[n] int prior_pred_choice;
  array[n] int posterior_pred_choice;
  
  for (i in 1:n) {
    // For prior predictions, use uniform prior (Beta(1,1))
    //prior_pred_choice[i] = beta_binomial_rng(1, 1, 1);
    prior_pred_choice[i] = beta_binomial_rng(1, 1, 1);
    
    // For posterior predictions, use integrated evidence
    real alpha_post = 1 + weight_direct* scale_f[i] + weight_social* scale_g[i];
    real beta_post = 1 + weight_direct*(total - scale_f[i]) + weight_social*(total - scale_g[i]);
    
    // Generate predictions using the complete beta-binomial model
    posterior_pred_choice[i] = beta_binomial_rng(total, alpha_post, beta_post);
    
    // Log likelihood calculation using beta-binomial
    log_lik[i] = beta_binomial_lpmf(scale_s[i] | total, alpha_post, beta_post);
  }
}
