// The input (data) for the model 
data {
 int<lower=1> n; // n is trials
 array[n] int h; // h is the choices of 'Self'
 array[n] int other; // other is the choices of 'Other'
}


// The parameters accepted by the model
parameters {
  real forgetting_raw; // how much of memory is determined by the last move of 'Other'
  
}


// Transforming the forgetting_raw parameter from log odds to probability space 
transformed parameters {
  real<lower=0, upper=1> forgetting = inv_logit(forgetting_raw);
  vector[n] memory;  // Declare memory so we can store it for later use
  
    // Set memory to be 0.5 for the first trial
  memory[1] = 0.5;

  // Update memory for all trials
  for (trial in 1:(n-1)) {
    memory[trial + 1] = forgetting * other[trial] + (1 - forgetting) * memory[trial];

    // Bound memory between 0.01 and 0.99
    if (memory[trial + 1] == 0) {
      memory[trial + 1] = 0.01;
    }
    if (memory[trial + 1] == 1) {
      memory[trial + 1] = 0.99;
    }
  }

}


// The model to be estimated
model {
  // Set Priors
  // We only need a prior for forgetting_raw as Stan will handle the transformation from log-odds to probability in the transformed parameters block
  // We choose a normal prior with a mean of 0 and sd of 1
  target += normal_lpdf(forgetting_raw | 0, 1);
  
  // Model, looping to keep track of memory
  for (trial in 1:n) {
      
      // The choices of 'Self' is given by the memeory of the trial 
      target += bernoulli_lpmf(h[trial] | memory[trial]);
    }
  } 



// This block is defining a generated quantity, meaning it does not affect inference but is useful for checking priors and posterior predictions
generated quantities {
  real forgetting_prior; // sampled prior value of forgetting
  
  array[n] int prior_predictions; // array for simulated choices based on priors
  array[n] int posterior_predictions;// array for simulated choices based on posteriors
  
  // The prior is a normal prior with a mean of 0 and sd of 1
  forgetting_prior = inv_logit(normal_rng(0, 1)); //Sample from log-odds scale and transform
 
  // Random choice for the first trial
  prior_predictions[1] = bernoulli_rng(0.5);   // Prior prediction for the first trial
  posterior_predictions[1] = bernoulli_rng(0.5); // Posterior prediction for the first trial
  
  // Goes through all trials to generate predictions
  for (trial in 2:n) {
   
   // Generate a prior based prediction for 'Self' choice using prior forgetting  
    prior_predictions[trial] = bernoulli_rng(
      forgetting_prior * other[trial-1] + (1-forgetting_prior) * memory[trial-1]);
   // Generate a posterior based prediction using estimated forgetting
    posterior_predictions[trial]  = bernoulli_rng(            
      forgetting * other[trial-1] + (1-forgetting) * memory[trial-1]);
  }
}


