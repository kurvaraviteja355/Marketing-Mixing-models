import numpy as np
import pandas as pd
import subprocess
import warnings
import arviz as az
import pymc as pm
import seaborn as sns
import pymc as pm
import pytensor.tensor as pt
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
import mlflow


def geometric_adstock(x, alpha: float = 0.0, l_max: int = 12):
    """Geometric adstock transformation."""
    cycles = [
        pt.concatenate(
            [pt.zeros(i), x[: x.shape[0] - i]]
        )
        for i in range(l_max)
    ]
    x_cycle = pt.stack(cycles)
    w = pt.as_tensor_variable([pt.power(alpha, i) for i in range(l_max)])
    return pt.dot(w, x_cycle)

def logistic_saturation(x, lam: float = 0.5):
    """Logistic saturation transformation."""
    return (1 - pt.exp(-lam * x)) / (1 + pt.exp(-lam * x))


def saturate(x, a):
    return 1 - pt.exp(-a*x)

def carryover(x, strength, length=21):
    w = pt.as_tensor_variable(
        [pt.power(strength, i) for i in range(length)]
    )
    
    x_lags = pt.stack(
        [pt.concatenate([
            pt.zeros(i),
            x[:x.shape[0]-i]
        ]) for i in range(length)]
    )
    
    return pt.dot(w, x_lags)


def get_git_revision_hash() -> str:
    """
    Retrieve the current Git commit hash of the HEAD revision.

    This function runs the command `git rev-parse HEAD` to obtain the commit hash of the HEAD revision
    in the current Git repository. It requires that the command is run within a Git repository and that
    the Git executable is available in the system's PATH.
    Credit to: https://stackoverflow.com/questions/14989858/get-the-current-git-hash-in-a-python-script

    Returns:
        str: A string representing the current Git commit hash, if successfully retrieved.
             If the command fails (e.g., not run within a Git repository or Git is not installed),
             a CalledProcessError will be raised by subprocess.check_output.

    Raises:
        subprocess.CalledProcessError: An error occurred while trying to execute the git command.
    """
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

## read the data 
data_df = pd.read_csv('Input_data/Mediaspenddata.csv', parse_dates=["Date"])

# Group inputs
original_paid_features = sorted(['tv_s', 'ooh_s', 'print_s', 'facebook_s', 'search_s'])
original_organic_features = sorted(['newsletter'])
original_competitor_features = sorted(['competitor_sales_b'])
original_control_features = sorted(['event1', 'event2'])
EXCLUDED_FEATURES = sorted(['facebook_i', 'search_clicks_p'])

# Gather all inputs into 1 sorted list
all_original_features = sorted(original_paid_features +
                               original_organic_features + 
                               original_competitor_features +
                               original_control_features)

# Create a feature selection dict
# And to be able to access values with later on
FEATURES = {
    # Gather all inputs into 1 sorted list for logging
    "features_all_possible": all_original_features,
    
    # Remove any features that the user specified earlier
    "features_included": sorted(set(all_original_features) - set(EXCLUDED_FEATURES)),
    
    # Also log the excluded features
    "features_excluded": EXCLUDED_FEATURES,
    
    # Update relevant feature groups to exclude user-specified features
    "features_paid": sorted(set(original_paid_features) - set(EXCLUDED_FEATURES)),
    "features_organic": sorted(set(original_organic_features) - set(EXCLUDED_FEATURES)),
    "features_competitor": sorted(set(original_competitor_features) - set(EXCLUDED_FEATURES)),
    "features_control": sorted(set(original_control_features) - set(EXCLUDED_FEATURES))
}

X = data_df.drop('Sales', axis=1)
y = data_df['sales']



# Scale control variables - these are exogenous and require a MinMax scaling
control_scaler = MinMaxScaler()
X[FEATURES["features_control"]] = control_scaler.fit_transform(X[FEATURES["features_control"]])

# Scale all media variables - these are exogenous too and require a MinMax scaling
paid_scaler = MinMaxScaler()
X[FEATURES["features_paid"]] = paid_scaler.fit_transform(X[FEATURES["features_paid"]])

# Remaining media features to scale
other_media_features = FEATURES["features_organic"] + FEATURES["features_competitor"]
other_media_scaler = MinMaxScaler()

X[other_media_features] = other_media_scaler.fit_transform(X[other_media_features])
# Then make competitor negative (it's >0 after scaling) so that it's forced to subtract from Revenue
# in our regression equation later
X.competitor_sales_b = X.competitor_sales_b * -1

# Also scale the target - the endogenous variable
endog_scaler = MaxAbsScaler()
y = endog_scaler.fit_transform(y.to_numpy().reshape(-1, 1)).flatten()



coords = {"paid": FEATURES['features_paid'],
          "organic": FEATURES['features_organic'],
          "competitor": FEATURES['features_competitor'], 
          "control": FEATURES['features_control'], 
          "fourier_mode": np.arange(SEASONALITY_CONFIG["seasonality_features"].size),
         }
         


# Create model in context manager
# Specify model coordinates from before
with pm.Model(coords=coords) as mmm:

    ##### ---- Mutable Coords ------ ####
    # Add date as mutable coordinate - for out-of-sample prediction later on
    mmm.add_coord(name="date", values=X.index, mutable=True)

    
    ##### ---- Data Containers ------ ####
    # Create container for channel data
    paid_data = pm.MutableData(name = "paid_data", 
                                 value = X[FEATURES['features_paid']].to_numpy(),
                                 dims = ("date", "paid"))
    
    # Create container for organic data
    organic_data = pm.MutableData(name = "organic_data", 
                                  value = X[FEATURES['features_organic']].to_numpy(),
                                  dims = ("date", "organic"))    
    
    # Create container for control data
    control_data = pm.MutableData(name = "control_data", 
                                  value = X[FEATURES['features_control']].to_numpy(),
                                  dims = ("date", "control"))
    
    # Create container for control data
    control_data = pm.MutableData(name = "control_data", 
                                  value = X[FEATURES['features_control']].to_numpy(),
                                  dims = ("date", "control"))
    

    ##### ---- Define priors ------ ####
    ## prior for baseline intercept term
    intercept =  pm.HalfNormal("intercept", sigma=2)

    ## prior for beta coeffecients / regressors
    beta_paid_coeffs = pm.HalfNormal("beta_paid_coeffs", sigma=2, dims="paid")

    ## prior for beta organic coeffecients / regressors
    beta_organic_coeffs = pm.HalfNormal("beta_organic_coeffs", sigma=2, dims="organic") 

    ## prior for beta competitor coeffecient / regressor
    beta_competitor_coeffs = pm.HalfNormal("beta_competitor_coeffs", sigma=2, dims="competitor")

    ## prior for beta control coeffecients / regressors
    beta_control_coeffs = pm.Normal("beta_control_coeffs", mu=0, sigma=2, dims="control")    

    ## prior for trend
    beta_trend = pm.Normal(name="beta_trend", mu=0, sigma=trend_priors.sigma)

    ## prior for seasonality
    beta_fourier = pm.Laplace(name="beta_fourier", mu=0, b=1, dims="fourier_mode")

    ## Geometric adstock priors
    geoad_param_paid = pm.Beta(f"geoad_param_paid", alpha=1, beta=3, dims="paid")
    geoad_param_organic = pm.Beta(f"geoad_param_organic", alpha=1, beta=3, dims="organic")   

    ## Logistic saturation priors
    sat_lam_paid = pm.Gamma(f"sat_lam_paid", alpha=3, beta=1, dims="paid")
    sat_lam_organic = pm.Gamma(f"sat_lam_organic", alpha=3, beta=1, dims="organic")

    # prior for likelihood noise level (note : must be positive)
    sigma = pm.HalfNormal("sigma", sigma=2)

    ## Adstocking with Geometric
    paid_adstock = pm.Deterministic(name="paid_adstock", 
                                    var=geometric_adstock(x=paid_data, alpha=geoad_param_paid, l_max=12), # Geometric adstock transformation
                                    dims=("date", "paid"))   
    ## Saturation (Logistic Function)
    paid_adstock_saturated = pm.Deterministic(name="paid_adstock_saturated",
                                            var=logistic_saturation(x=paid_adstock, lam=sat_lam_paid), # Logistic saturation transformation                                      
                                            dims=("date", "paid"))
    

    # Expected value of outcome
    mu = pm.Deterministic(name = "mu",
                        var = paid_contributions.sum(axis=-1) + 
                                organic_contributions.sum(axis=-1) +
                                competitor_contributions.sum(axis=-1) + # this is negative
                                control_contributions.sum(axis=-1) +
                                trend + 
                                seasonality,
                        dims="date")


    # Likelihood (sampling distribution) of observations
    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_obs_data, dims="date")
    



# Set seed for this cell
rng = np.random.default_rng(42)

# Fit model 
with mmm:
    # Run MCMC algorithm
    trace = pm.sample(draws=1000, # number of samples to draw from posterior distribution
                      tune=500, # number of burn-in samples, samples that are discarded 
                      chains=4, # number Markov Chains (separate sequences of samples to pull)
                      cores=4, # how many cores to run the model with, defaults to same number of chains
                      random_seed=rng)

    # Sample from posteriors
    mmm_posterior_predictive = pm.sample_posterior_predictive(trace=trace,
                                                              # pass in earlier defined seed
                                                              random_seed = rng)








# Initiate the MLflow run context
with mlflow.start_run(run_name="your_run_name") as run:

    # Log git hash
    git_commit_hash = get_git_revision_hash()
    if git_commit_hash:
        # Log the git commit hash as a tag
        mlflow.set_tag("git_commit_id", git_commit_hash)


    # Log the pre-processing steps taken (Dictionaries of constants)
    mlflow.log_params(FEATURES)
    mlflow.log_params(DIM_REDUCTION_CONFIG)
    mlflow.log_params(SEASONALITY_CONFIG)
    mlflow.log_params(TRAIN_TEST_CONFIG)
    mlflow.log_params(MODEL_CONFIG)
    

    # Log artifacts
    mlflow.log_figure(prior_plot, "graphs/prior_plot.png")
    mlflow.log_figure(train_plot, "graphs/train_plot.png")
    mlflow.log_figure(test_plot, "graphs/test_plot.png")
    mlflow.log_figure(cpa_plot, "graphs/cpa_plot.png")
    mlflow.log_figure(waterfall_plot, "graphs/waterfall_plot.png")
    # Also log the data output
    mlflow.log_artifact("model_priors.csv", "output_data")
    mlflow.log_artifact("model_coeffs.csv", "output_data")
    mlflow.log_artifact("channel_cpa.csv", "output_data")
    mlflow.log_artifact("cpa_comparison_to_expectations.csv", "output_data")
    

    # Log model metrics
    mlflow.log_metrics(model_metrics)
    mlflow.log_metrics(model_diagnostics)
    mlflow.log_metrics(loocv_metrics)  