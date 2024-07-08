import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde
from typing import Dict
from taipy.gui import Markdown
import pymc as pm
#from sample_priors import * 


def draw_samples_from_prior(dist: pm.Distribution, num_samples: int = 50_000,
                            seed=None, **kwargs) -> np.ndarray:
    """
    Draws samples from the prior distribution of a given PyMC distribution.

    This function creates a PyMC model with a single variable, drawn from the specified
    distribution, and then samples from the prior of this distribution.

    Parameters
    ----------
    dist : pm.Distribution
        The PyMC distribution from which to draw samples.
    num_samples : int, optional
        The number of samples to draw from the prior distribution. Default is 10,000.
    seed : int or None, optional
        The seed for the random number generator to ensure reproducibility. If None, 
        the results will vary between runs. Default is None.
    **kwargs
        Additional keyword arguments to pass to the distribution constructor.
        e.g. sigma for Normal, alpha and beta for Beta.
        See PyMC Distributions for more info: https://www.pymc.io/projects/docs/en/stable/api/distributions.html

    Returns
    -------
    np.ndarray
        An array of samples drawn from the specified prior distribution.
    """
    with pm.Model():

        # Define a variable with the given distribution
        my_dist = dist(name = "my_dist", **kwargs)
        
        # Sample from the prior distribution of the model
        draws = pm.draw(my_dist, draws=num_samples, random_seed=seed)
        
    # Return the drawn samples
    return draws

def plot_prior_distribution(draws, nbins=100, opacity=0.1, title="Prior Distribution - Visualised"):
    """
    Plots samples of a prior distribution as a histogram with a KDE (Kernel Density Estimate) overlay
    and a violin plot along the top too with quartile values.
    
    Parameters:
    - draws: numpy array of samples from prior distribution.
    - nbins: int, the number of bins for the histogram.
    - opacity: float, the opacity level for the histogram bars.
    - title: str, the title of the plot.
    """
    # Create the histogram using Plotly Express
    fig = px.histogram(draws, x=draws, nbins=nbins, title=title, 
                       labels={"x": "Value"}, histnorm='probability density', opacity=opacity,
                       marginal="violin", color_discrete_sequence=['#0047AB'])
    
    # Compute the KDE
    kde = gaussian_kde(draws)
    x_range = np.linspace(min(draws), max(draws), 500)
    kde_values = kde(x_range)
    
    # Add the KDE plot to the histogram figure
    fig.add_trace(go.Scatter(x=x_range, y=kde_values, mode='lines',
                              name='KDE', line_color="#DA70D6",
                              opacity=0.8))
    
    # Customize the layout
    fig.update_layout(xaxis_title='Value of Prior', yaxis_title='Density')
    
    # Return the plot
    return fig



def find_optimal_gamma_parameters(lower: float, upper: float, mass: float, 
                                  scaling_factor: int = 100, maxiter: int = 1000) -> Dict[str, float]:
    """
    Finds the optimal parameters for a Gamma distribution given constraints on the lower and upper
    quantiles and a specified mass (confidence level) under those quantiles. Uses multiple initial
    guesses to improve the reliability of finding a suitable solution.

    Parameters:
    - lower (float): The lower bound of the quantile range for the Gamma distribution.
    - upper (float): The upper bound of the quantile range for the Gamma distribution.
    - mass (float): The mass (probability/confidence) between the lower and upper quantiles.
    - scaling_factor (int, optional): A factor used to scale down the lower and upper bounds
      to avoid optimization issues with large values. Defaults to 100.
    - maxiter (int, optional): Maximum number of iterations for the optimizer. Defaults to 1000.

    Returns:
    - Dict[str, float]: A dictionary with the optimal 'alpha' and 'beta' parameters for the
      Gamma distribution.
    """
    # Scale the bounds to mitigate optimization issues with large values
    scaled_lower = lower / scaling_factor
    scaled_upper = upper / scaling_factor

    # Define a set of initial guesses for the optimizer
    initial_guesses = [
        {"alpha": scaled_lower, "beta": 1},
        {"alpha": scaled_upper, "beta": 1},
        {"alpha": (scaled_lower + scaled_upper) / 2, "beta": 1},
    ]

    # Run the optimization for each initial guess and collect the results
    results = []
    for guess in initial_guesses:
        result = pm.find_constrained_prior(
            pm.Gamma, 
            lower=scaled_lower, 
            upper=scaled_upper, 
            mass=mass, 
            init_guess=guess,
            options={"maxiter": maxiter}
        )
        results.append(result)

    # Average the 'alpha' and 'beta' parameters from the results
    avg_result = {
        "alpha": np.mean([r["alpha"] * scaling_factor for r in results]),
        "beta": np.mean([r["beta"] for r in results])
    }
    
    return avg_result



# Function to dynamically render the selected chart
def create_uniform(lower, upper, seed):
    uniformsamples = draw_samples_from_prior(pm.Uniform, lower=lower, upper=upper, seed=seed)
    return plot_prior_distribution(uniformsamples, title='Uniform Distribution Samples')

def create_normal(mu, sigma, seed):
    normalsamples = draw_samples_from_prior(pm.Normal, mu=mu, sigma=sigma, seed=seed)
    return plot_prior_distribution(normalsamples, title='Normal Distribution Samples')

def create_hnormal(sigma, seed):
    hnormalsamples = draw_samples_from_prior(pm.HalfNormal, sigma=sigma, seed=seed)
    return  plot_prior_distribution(hnormalsamples, title='HalfNormal Distribution Samples')

def create_beta(balpha, bbeta, seed):
    betasamples = draw_samples_from_prior(pm.Beta, alpha=balpha, beta=bbeta, seed=seed)
    return plot_prior_distribution(betasamples, title='Beta Distribution Samples')

def create_gamma(gmmalpha, gmmbeta, seed):
    gammasamples = draw_samples_from_prior(pm.Gamma, alpha=gmmalpha, beta=gmmbeta, seed=seed)
    return plot_prior_distribution(gammasamples, title='Gamma Distribution Samples')

def create_poisson(poisonlam, seed):
    psamples = draw_samples_from_prior(pm.Poisson, mu=poisonlam, seed=seed)
    return plot_prior_distribution(psamples, title='Poisson Distribution Samples')

def create_bernoulli(p, seed):
    bernsamples = draw_samples_from_prior(pm.Bernoulli, p=p, seed=seed)
    return plot_prior_distribution(bernsamples, title='Bernoulli Distribution Samples')

def create_expo(lam, seed):
    exposamples = draw_samples_from_prior(pm.Exponential, lam=lam, seed=seed)
    return plot_prior_distribution(exposamples, title='Exponential Distribution Samples')

def create_weibull(walpha, wbeta, seed):
    weibullsamples = draw_samples_from_prior(pm.Weibull, alpha=walpha, beta=wbeta, seed=seed)
    return plot_prior_distribution(weibullsamples, title='Weibull Distribution Samples')

def create_tnormal(tmu, tsigma, tlower, tupper, seed):
    tnsamples = draw_samples_from_prior(pm.TruncatedNormal, mu=tmu, sigma=tsigma, lower=tlower, upper=tupper, seed=seed)
    return plot_prior_distribution(tnsamples, title='TruncatedNormal Distribution Samples')

def create_studentT(nu, stmu, stsigma, seed):
    studentsamples = draw_samples_from_prior(pm.StudentT, nu=nu, mu=stmu, sigma=stsigma, seed=seed)
    return plot_prior_distribution(studentsamples, title='StudentT Distribution Samples')

def create_lnormal(lmu, lsigma, seed):
    lnsamples = draw_samples_from_prior(pm.LogNormal, mu=lmu, sigma=lsigma, seed=seed)
    return plot_prior_distribution(lnsamples, title='LogNormal Distribution Samples')

distribution_md = Markdown("""
## Distribution priors for Bayesian Modelling
<|{dis_show_chart}|toggle|lov={dis_choice_chart}|>
### Distribution
<|part|render={dis_show_chart == 'Uniform'}|
<|layout|columns=1 1|
<|first column
<|container container-styling|
Lower Bound: <|{lower}|> <br/>
<|{lower}|slider|min=0|max=1|step=0.01|on_change=update_distribution|>
|>
|>
<|second column
<|container container-styling|
Upper Bound: <|{upper}|> <br/>
<|{upper}|slider|min=0|max=1|step=0.01|on_change=update_distribution|>
|>
|>
|>
|>

<|part|render={dis_show_chart == 'Normal'}|
<|layout|columns=1 1|
<|first column
<|container container-styling|
mu: <|{mu}|> <br/>
<|{mu}|slider|min=0|max=1|step=0.01|on_change=update_distribution|>
|>
|>
<|second column
<|container container-styling|
sigma: <|{sigma}|> <br/>
<|{sigma}|slider|min=0|max=1|step=0.01|on_change=update_distribution|>
|>
|>
|>
|>

<|part|render={dis_show_chart == 'HalfNormal'}|
sigma: <|{sigma}|> <br/>
<|{sigma}|slider|min=0|max=1|step=0.01|on_change=update_distribution|>
|>

<|part|render={dis_show_chart == 'Beta'}|
<|layout|columns=1 1|
<|first column
<|container container-styling|
alpha: <|{alpha}|> <br/>
<|{alpha}|slider|min=0|max=1|step=0.01|on_change=update_distribution|>
|>
|>
<|second column
<|container container-styling|
beta: <|{beta}|> <br/>
<|{beta}|slider|min=0|max=1|step=0.01|on_change=update_distribution|>
|>
|>
|>
|>

<|part|render={dis_show_chart == 'Gamma'}|
<|layout|columns=1 1|
<|first column
<|container container-styling|
Alpha (Shape): <|{gmmalpha}|> <br/>
<|{gmmalpha}|slider|min=0.01|max=15|step=0.01|on_change=update_distribution|>
|>
|>
<|second column
<|container container-styling|
Beta (rate): <|{gmmbeta}|> <br/>
<|{gmmbeta}|slider|min=0|max=10|step=0.01|on_change=update_distribution|>
|>
|>
|>
|>

<|part|render={dis_show_chart == 'Poisson'}|
Lambda: <|{poisonlam}|> <br/>
<|{poisonlam}|slider|min=0|max=15|step=0.5|on_change=update_distribution|>
|>

<|part|render={dis_show_chart == 'Bernoulli'}|
rate: <|{p}|> <br/>
<|{p}|slider|min=0|max=1|step=0.01|on_change=update_distribution|>
|>

<|part|render={dis_show_chart == 'Exponential'}|
lambda: <|{lam}|> <br/>
<|{lam}|slider|min=0|max=15|step=0.05|on_change=update_distribution|>
|>

<|part|render={dis_show_chart == 'Weibull'}|
<|layout|columns=1 1|
<|first column
<|container container-styling|
Shape (K): <|{walpha}|> <br/>
<|{walpha}|slider|min=0|max=10|step=0.01|on_change=update_distribution|>
|>
|>
<|second column
<|container container-styling|
Scale (lam): <|{wbeta}|> <br/>
<|{wbeta}|slider|min=0|max=10|step=0.01|on_change=update_distribution|>
|>
|>
|>
|>

<|part|render={dis_show_chart == 'TruncatedNormal'}|
<|layout|columns=1 1 1 1|
<|first column
<|container container-styling|
mu: <|{tmu}|> <br/>
<|{tmu}|slider|min=0|max=1|step=0.01|on_change=update_distribution|>
|>
|>
<|second column
<|container container-styling|
sigma: <|{tsigma}|> <br/>
<|{tsigma}|slider|min=0|max=1|step=0.01|on_change=update_distribution|>
|>
|>

<|third column
<|container container-styling|
Lower: <|{tlower}|> <br/>
<|{tlower}|slider|min=-10|max=10|step=0.1|on_change=update_distribution|>
|>
|>

<|fourth column
<|container container-styling|
Upper: <|{tupper}|> <br/>
<|{tupper}|slider|min=0|max=10|step=0.1|on_change=update_distribution|>
|>
|>
|>
|>

<|part|render={dis_show_chart == 'StudentT'}|
<|layout|columns=1 1 1|
<|first column
<|container container-styling|
nu: <|{nu}|> <br/>
<|{nu}|slider|min=0|max=15|step=0.1|on_change=update_distribution|>
|>
|>

<|second column
<|container container-styling|
mu: <|{stmu}|> <br/>
<|{stmu}|slider|min=0|max=1|step=0.01|on_change=update_distribution|>
|>
|>
<|third column
<|container container-styling|
sigma: <|{stsigma}|> <br/>
<|{stsigma}|slider|min=0|max=1|step=0.01|on_change=update_distribution|>
|>
|>
|>
|>

<|part|render={dis_show_chart == 'LogNormal'}|
<|layout|columns=1 1|
<|first column
<|container container-styling|
mu: <|{lmu}|> <br/>
<|{lmu}|slider|min=0|max=1|step=0.01|on_change=update_distribution|>
|>
|>
<|second column
<|container container-styling|
sigma: <|{lsigma}|> <br/>
<|{lsigma}|slider|min=0|max=1|step=0.01|on_change=update_distribution|>
|>
|>
|>
|>
<|chart|figure={get_selected_chart(dis_show_chart, lower, upper, mu, sigma, balpha, bbeta, gmmalpha,gmmbeta, poisonlam, p, lam, walpha, wbeta, tmu, tsigma, tlower, tupper,nu, stmu, stsigma, lmu, lsigma, seed)}|height=500px|on_change=update_distribution|>
""")