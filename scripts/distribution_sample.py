from taipy.gui import Gui, Icon, State, navigate, notify, Markdown
import numpy as np
import matplotlib.pyplot as plt
import taipy.gui.builder as tgb
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde
from typing import Dict
import pymc as pm
from priors_samples import * 

lower = 0.00
upper = 1.00
seed = 42

mu = 0.0
sigma = 1.0
alpha = 1.0
beta = 1.0
p = 0.5
lam = 0.5
nu = 3.0
poisonlam = 0.5

balpha = 0.5
bbeta = 0.6
walpha = 1.5
wbeta = 1.0
pmu = 0.8

gmmalpha = 1.0
gmmbeta = 1.0
tmu = 0.0
tsigma=1.0
tlower=0.0
tupper=2.0

lmu = 0.0
lsigma = 1.0
stmu = 0.0
stsigma = 1.0



# Function to dynamically render the selected chart
def create_uniform(lower, upper):
    uniformsamples = draw_samples_from_prior(pm.Uniform, lower=lower, upper=upper, seed=seed)
    return plot_prior_distribution(uniformsamples, title='Uniform Distribution Samples')

def create_normal(mu, sigma):
    normalsamples = draw_samples_from_prior(pm.Normal, mu=mu, sigma=sigma, seed=seed)
    return plot_prior_distribution(normalsamples, title='Normal Distribution Samples')

def create_hnormal(sigma):
    hnormalsamples = draw_samples_from_prior(pm.HalfNormal, sigma=sigma, seed=seed)
    return  plot_prior_distribution(hnormalsamples, title='HalfNormal Distribution Samples')

def create_beta(balpha, bbeta):
    betasamples = draw_samples_from_prior(pm.Beta, alpha=balpha, beta=bbeta, seed=seed)
    return plot_prior_distribution(betasamples, title='Beta Distribution Samples')

def create_gamma(gmmalpha, gmmbeta):
    gammasamples = draw_samples_from_prior(pm.Gamma, alpha=gmmalpha, beta=gmmbeta, seed=seed)
    return plot_prior_distribution(gammasamples, title='Gamma Distribution Samples')

def create_poisson(poisonlam):
    psamples = draw_samples_from_prior(pm.Poisson, mu=poisonlam, seed=seed)
    return plot_prior_distribution(psamples, title='Poisson Distribution Samples')

def create_bernoulli(p):
    bernsamples = draw_samples_from_prior(pm.Bernoulli, p=p, seed=seed)
    return plot_prior_distribution(bernsamples, title='Bernoulli Distribution Samples')

def create_expo(lam):
    exposamples = draw_samples_from_prior(pm.Exponential, lam=lam, seed=seed)
    return plot_prior_distribution(exposamples, title='Exponential Distribution Samples')

def create_weibull(walpha, wbeta):
    weibullsamples = draw_samples_from_prior(pm.Weibull, alpha=walpha, beta=wbeta, seed=seed)
    return plot_prior_distribution(weibullsamples, title='Weibull Distribution Samples')

def create_tnormal(tmu, tsigma, tlower, tupper):
    tnsamples = draw_samples_from_prior(pm.TruncatedNormal, mu=tmu, sigma=tsigma, lower=tlower, upper=tupper, seed=seed)
    return plot_prior_distribution(tnsamples, title='TruncatedNormal Distribution Samples')

def create_studentT(nu, stmu, stsigma):
    studentsamples = draw_samples_from_prior(pm.StudentT, nu=nu, mu=stmu, sigma=stsigma, seed=seed)
    return plot_prior_distribution(studentsamples, title='StudentT Distribution Samples')

def create_lnormal(lmu, lsigma):
    lnsamples = draw_samples_from_prior(pm.LogNormal, mu=lmu, sigma=lsigma, seed=seed)
    return plot_prior_distribution(lnsamples, title='LogNormal Distribution Samples')
   
def get_selected_chart(dis_show_chart, lower, upper, mu, sigma, balpha, bbeta, gmmalpha, 
                       gmmbeta, poisonlam, p, lam, walpha, wbeta, tmu, tsigma, tlower, tupper,
                       nu, stmu, stsigma, lmu, lsigma):
    if dis_show_chart == "Uniform":
        return create_uniform(lower, upper)
    elif dis_show_chart == "Normal":
        return create_normal(mu, sigma)
    elif dis_show_chart == 'HalfNormal':
        return create_hnormal(sigma)
    elif dis_show_chart == 'Beta':
        return create_beta(balpha, bbeta)
    elif dis_show_chart == 'Gamma':
        return create_gamma(gmmalpha, gmmbeta)
    elif dis_show_chart == 'Poisson':
        return create_poisson(poisonlam)
    elif dis_show_chart == 'Bernoulli':
        return create_bernoulli(p)
    elif dis_show_chart == 'Exponential':
        return create_expo(lam)
    elif dis_show_chart == 'Weibull':
        return create_weibull(walpha, wbeta)
    elif dis_show_chart == 'TruncatedNormal':
        return create_tnormal(tmu, tsigma, tlower, tupper)
    elif dis_show_chart == 'StudentT':
        return create_studentT(nu, stmu, stsigma)
    elif dis_show_chart == 'LogNormal':
        return create_lnormal(lmu, lsigma)


def update_distribution(state):
    lower = float(state.lower)
    upper = float(state.upper)
    mu = float(state.mu)
    sigma = float( state.sigma)
    balpha = float(state.balpha)
    bbeta = float(state.bbeta)
    gmmalpha = float(state.gmmalpha)
    gmmbeta = float(state.gmmbeta)
    poisonlam = float(state.gmmbeta)
    p = float(state.p)
    lam = float(state.lam)
    walpha = float(state.gmmbeta)
    wbeta = float(state.wbeta)
    tmu = float(state.tmu)
    tsigma = float(state.tsigma)
    tlower = float(state.tlower)
    tupper = float(state.tupper)
    nu = float(state.nu)
    stmu = float(state.stmu)
    stsigma = float(state.stsigma)
    lmu = float(state.lmu)
    lsigma = float(state.lsigma)
    get_selected_chart(state.dis_show_chart, lower, upper, mu, sigma, balpha, bbeta, gmmalpha, 
                       gmmbeta, poisonlam, p, lam, walpha, wbeta, tmu, tsigma, tlower, tupper,
                       nu, stmu, stsigma, lmu, lsigma)


dis_choice_chart = ["Uniform", "Normal", "HalfNormal", "Beta", "Poisson", "Bernoulli", "Exponential", "Weibull", "TruncatedNormal", "StudentT", "LogNormal"]
dis_show_chart = dis_choice_chart[0]

uniform = None
normal = None

light_theme = {
    "palette": {
        "background": {
            "default": "#d580ff"  
        },
        "primary": {"main": "#ffffff"}
    }
}

dark_theme = {
    "palette": {
        "background": {
            "default": "#471061"  
        },
        "primary": {"main": "#000000"}
    }
}

card = {
    "palette": {
        "background": {
            "default": "#471061"  
        },
        "primary": {"main": "#000000"}
    }
}


layout = """

<|toggle|theme|>

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


<|chart|figure={get_selected_chart(dis_show_chart, lower, upper, mu, sigma, balpha, bbeta, gmmalpha,gmmbeta, poisonlam, p, lam, walpha, wbeta, tmu, tsigma, tlower, tupper,nu, stmu, stsigma, lmu, lsigma)}|height=500px|on_change=update_distribution|>
"""

if __name__ == "__main__":
    
    Gui(layout).run(use_reloader=True, port=5002)