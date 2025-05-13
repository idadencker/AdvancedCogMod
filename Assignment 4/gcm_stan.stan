// The Generalized Context Model (GCM) formalized as STAN
// obs: structured for parameter estimation

// The GCM is a exampler-based categorization model that assume that every encountered example is stored in memory with its category label.
// We then compute similarity between a new stimulus and all stored exemplars and use that similary to make our decision
// Similarity is caluclated as the difference between 2 vectors (where the vector represents the features) using a mathematical function
// Attention can be distributed differently across stimulus dimensions i.e. we can assign more importance to a specific feature (e.g. spots in the alien game experiment) than to other features 
// In summary: Categorization of new items depends on their similarity to specific remembered examples and hence learning is incremental, one example at a time


// The GCM has 2 parameters: w and c
// c is the sensitivity parameter that determines how quickly similarity decreases with distance
// A higher value of c means similarity decreases more rapidly with distance. This can represent human factors like increased discriminability, reduced generalization, or greater specificity in memory.
// With low c values, similarity remains high even for distant stimuli and with high c values, similarity drops rapidly even for small distances

// w is the attention weight for dimension m, and must sum to 1
// w parameter takes the format of a vector e.g. (0.7,0.1,0.1,0.06,0.04) (note how the 5 attention numbers sum to 1)


//the code as it is now deals with stimuli which can be of 2 categories (1 or 2), in our data we also have 2 categories (dangerous vs non-dangerous)


data {
    int<lower=1> ntrials;  // number of trials. 288 per part (96*3) (no test trial)
    int<lower=1> nfeatures;  // number of predefined relevant features (we have 5)
    array[ntrials] int<lower=0, upper=1> cat_one; // true responses on a trial by trial basis (this is the dangerous column which in our case can be 0 or 1)
    array[ntrials] int<lower=0, upper=1> y;  // decisions on a trial by trial basis (this is the response column which can be 0 or 1)
    array[ntrials, nfeatures] real obs; // stimuli as vectors of features. So we have 5 features that can be 0 or 1, so a vector could e.g. be [0,0,1,0,1]
    real<lower=0, upper=1> b;  // initial bias for category one over two

    // priors
    //The dirichlet distribution is using a vector α of K positive real numbers as parameter, which is called the concentration parameter. 
    vector[nfeatures] w_prior_values;  // concentration parameters for dirichlet distribution <lower=1>
    //the dirichlet distribution in our case splits into 5 (w1,w2,w3,w4,w5) where the 5 w's after using the dirichlet must sum to exatcly 1. If e.g. w1 is 0.9 there are only 0.1 to be distributed across the 4 remaining w's. i.e. we deem the w1 very important (0.9) and the 4 other w's less so. 
    // prior vector must be same lenght of nfeatures, so it takes the format [1,1,1,1,1]
    //w_prior_values of [1,1,1,1,1] will mean we put equal evidence to each feature but since 1 is a very low number it is not very certain of the equal evidence
    //w_prior_values of [100,100,100,100,100] will also mean we put equal evidence to each feature but here we are very certain of of the equal evidence since 100 is a very high number
    
    // c is a real number and hence is sampled with a mean (of 0) and a sd (of 1)
    array[2] real c_prior_values;  // mean and variance for logit-normal distribution
}


transformed data {
    array[ntrials] int<lower=0, upper=1> cat_two; // create dummy variable for cat 2 over cat 1
    
    array[sum(cat_one)] int<lower=1, upper=ntrials> cat_one_idx; // create an array of which stimuli are cat 1. 
    
    array[ntrials-sum(cat_one)] int<lower=1, upper=ntrials> cat_two_idx; // create an array of which stimuli are cat 2. Follow the logic that if it is not cat 1 it will be cat 2 hence doing: array[ntrials-sum(cat_one)] 
    
    
    int idx_one = 1; // Initializing. These will be used to track positions in two separate index arrays: cat_one_idx and cat_two_idx
    int idx_two = 1;
    for (i in 1:ntrials){
        cat_two[i] = abs(cat_one[i]-1); // abs computes the absolute value, e.g. -23 will be 23. 
              //logic: If cat_one[i] is 1, then cat_two[i] becomes abs(1 - 1) = 0. And vice versa if cat_one[i] is 0, then cat_two[i] becomes abs(0 - 1) = 1
          
        // This block classifies the trial index i based on the value of cat_one[i]:    
        if (cat_one[i]==1){ //if the category is 1
            cat_one_idx[idx_one] = i; //store i in the cat_one_idx array at position idx_one
            idx_one +=1; //increment idx_one beacuse criteria is met 
            
        } else { //if the category is not 1 (and hence 0)
            cat_two_idx[idx_two] = i; //store i in cat_two_idx at position idx_two
            idx_two += 1; //increment
        }
    }
}



parameters {
    simplex[nfeatures] w;  // defines a vector w of nfeatures lenght. simplex means sum(w)=1, i.e the 5 weights must sum to 1
    real logit_c;
}


transformed parameters {
    // parameter c 
    real<lower=0, upper=2> c = inv_logit(logit_c)*2;  // times 2 as c is bounded between 0 and 2

    // parameter r (probability of response = category 1)
    //r[i] is the probability of choosing category 1 on trial i (bounded to avoid exact 0 or 1 — which can break sampling
    array[ntrials] real<lower=0.0001, upper=0.9999> r; 
    //rr[i] is a raw version of the computed probability before bounding/clipping.These will be filled during the loop.
    array[ntrials] real rr;

    for (i in 1:ntrials) {

        // calculate distance from obs to all exemplars
        
        //Create a vector exemplar_sim of length i-1 to hold similarity values between the current trial i and each previous trial
        array[(i-1)] real exemplar_sim;
        
        //For each previous trial e:
        for (e in 1:(i-1)){
          
            //Create a vector of lenght nfeatures to hold the temporary distance
            array[nfeatures] real tmp_dist;
            
            //for each feature 
            for (j in 1:nfeatures) {
              
                //calcualte the weighted absolute difference on feature j.
                tmp_dist[j] = w[j]*abs(obs[e,j] - obs[i,j]);
            }
            
            // take the total dissimilarity between previous trial e and curent trial i, and use exponential similarity function: exemplar_sim[e] = exp(-c * distance)
            exemplar_sim[e] = exp(-c * sum(tmp_dist));
        }

        
        if (sum(cat_one[:(i-1)])==0 || sum(cat_two[:(i-1)])==0){  // if there are no examplars in one of the categories
            r[i] = 0.5; //set the probability to be 50% (random )

        } else {
            // calculate similarity
            array[2] real similarities; //define 2-element array to store summed similarities for category 1 and category 2 exemplars.
            
            // Extract previous trial indices for each category
            array[sum(cat_one[:(i-1)])] int tmp_idx_one = cat_one_idx[:sum(cat_one[:(i-1)])];
            array[sum(cat_two[:(i-1)])] int tmp_idx_two = cat_two_idx[:sum(cat_two[:(i-1)])];
            
            //Sum the similarity values for exemplars in each category
            similarities[1] = sum(exemplar_sim[tmp_idx_one]);
            similarities[2] = sum(exemplar_sim[tmp_idx_two]);

            // calculate r[i]
            
            //first, Compute raw response probability
            rr[i] = (b*similarities[1]) / (b*similarities[1] + (1-b)*similarities[2]);

            // to make the sampling work
            if (rr[i] > 0.9999){
                r[i] = 0.9999;
            } else if (rr[i] < 0.0001) {
                r[i] = 0.0001;
            } else if (rr[i] > 0.0001 && rr[i] < 0.9999) {
                r[i] = rr[i];
            } else {
                r[i] = 0.5;
            }
        }
    }
}

model {
    // Priors
    //sample the w using the dirichlet distrubtion using the w_prior_values (remember w_prior_values is a vector)
    // the dirichlet distrubtion make sure w sum to 1 
    target += dirichlet_lpdf(w | w_prior_values);
    
    //the c parameters are normally sampled using the mean (c_prior_values[1]) and sd c_prior_values[2]
    target += normal_lpdf(logit_c | c_prior_values[1], c_prior_values[2]);
    
    // Decision Data
    //the decision of the participant is determined by the calculated probability 
    target += bernoulli_lpmf(y | r);
}


//generated quantities {
    // Prior samples, posterior predictions, and log likelihood calculations...
    //real w_prior;
    //real c_prior;
    //w_prior = dirichlet_rng(w | w_prior_values);
    //c_prior = inv_logit(normal_rng(c_prior_values[1], c_prior_values[2]));
//}

generated quantities {
    vector[nfeatures] w_prior;
    real c_prior;
    
    w_prior = dirichlet_rng(w_prior_values);
    c_prior = inv_logit(normal_rng(c_prior_values[1], c_prior_values[2]));
}
