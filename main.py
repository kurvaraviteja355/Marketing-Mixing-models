import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde
from typing import Dict
from taipy.gui import Gui, Icon, State, navigate, notify, Markdown
import pymc as pm
import warnings
from pages.mmm_functions import *
from pages.distribution.distributions import * 
from pages.adstock.adstock import *
from pages.saturation.saturation import * 



# Starting value for adstock
initial_impact = 100
num_weeks = 47
beta = 0.50
max_peak = 10
shape_par = 0.10
scale_par  = 0.10

# Generate simulated marketing data
seed = np.random.seed(42)
num_points = 500
media_spending = np.linspace(0, 1000, num_points) # x-axis
#initial parameters for saturation 
root_alpha = 0.45
hill_alpha = 0.45
hill_gamma = 100
logistic_lam = 500
tanh_b = 5
tanh_c = 50
mm_alpha = 25
mm_lam = 50

###intial values for the distributions
lower = 0.00
upper = 1.00

mu = 0.0
sigma = 1.0

alpha = 1.0
beta = 1.0

p = 0.5
lam = 0.5
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
nu = 3.0


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


# Function to get selected distribution chart type
def get_selected_chart(dis_show_chart, lower, upper, mu, sigma, balpha, bbeta, gmmalpha, 
                       gmmbeta, poisonlam, p, lam, walpha, wbeta, tmu, tsigma, tlower, tupper,
                       nu, stmu, stsigma, lmu, lsigma, seed):
    if dis_show_chart == "Uniform":
        return create_uniform(lower, upper, seed)
    elif dis_show_chart == "Normal":
        return create_normal(mu, sigma, seed)
    elif dis_show_chart == 'HalfNormal':
        return create_hnormal(sigma, seed)
    elif dis_show_chart == 'Beta':
        return create_beta(balpha, bbeta, seed)
    elif dis_show_chart == 'Gamma':
        return create_gamma(gmmalpha, gmmbeta, seed)
    elif dis_show_chart == 'Poisson':
        return create_poisson(poisonlam, seed)
    elif dis_show_chart == 'Bernoulli':
        return create_bernoulli(p, seed)
    elif dis_show_chart == 'Exponential':
        return create_expo(lam)
    elif dis_show_chart == 'Weibull':
        return create_weibull(walpha, wbeta, seed)
    elif dis_show_chart == 'TruncatedNormal':
        return create_tnormal(tmu, tsigma, tlower, tupper, seed)
    elif dis_show_chart == 'StudentT':
        return create_studentT(nu, stmu, stsigma, seed)
    elif dis_show_chart == 'LogNormal':
        return create_lnormal(lmu, lsigma, seed)
    

## function to deplay chart from toggle selection
def get_selected_adstock(show_Adstock, initial_impact, num_weeks, beta, max_peak, shape_par, scale_par):
    if show_Adstock == 'Geometric':
        return adstock_geometric(initial_impact, num_weeks, beta)
    elif show_Adstock== 'Delayed Geometric':
        return adstock_delay_geometric(initial_impact, num_weeks, beta, max_peak)
    elif show_Adstock == 'Weibull CDF':
        return adstock_weibullCDF(initial_impact, num_weeks, shape_par, scale_par)
    elif show_Adstock == 'Weibull PDF' :
        return adstock_weibullPDF(initial_impact, num_weeks, shape_par, scale_par)
    
# Function to display saturation curves
def display_saturate(display_saturation, media_spending, num_points, root_alpha, hill_alpha, hill_gamma, logistic_lam, tanh_b, tanh_c, mm_alpha, mm_lam):
    if display_saturation == 'Root':
        return root_curve(media_spending, root_alpha, num_points, title_text= display_saturation)
    elif display_saturation == 'Hill':
        return hill_curve(media_spending, hill_alpha, hill_gamma, num_points, title_text=display_saturation)
    elif display_saturation== 'Logistic' :
        return logit_curve(media_spending, logistic_lam, num_points, title_text=display_saturation)
    elif display_saturation == 'Tanh':
        return tanh_curve(media_spending, tanh_b, tanh_c, num_points, title_text=display_saturation)
    elif display_saturation == 'Michaelis-Menten':
        return mm_curve(media_spending, mm_alpha, mm_lam, num_points, title_text=display_saturation)
    
# Update distribution state
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
    seed = state.seed
    get_selected_chart(state.dis_show_chart, lower, upper, mu, sigma, balpha, bbeta, gmmalpha, 
                       gmmbeta, poisonlam, p, lam, walpha, wbeta, tmu, tsigma, tlower, tupper,
                       nu, stmu, stsigma, lmu, lsigma, seed)
    


# Update adstock state
def update_adstock(state):
    num_weeks = state.num_weeks
    beta = float(state.beta)
    max_peak = state.max_peak
    shape_par = float( state.shape_par)
    scale_par = float(state.scale_par)
    get_selected_adstock(state.show_Adstock, initial_impact, num_weeks, beta, max_peak, shape_par, scale_par)


# Update saturation state
def update_saturation(state):
    media_spending = state.media_spending
    root_alpha = float(state.root_alpha)
    hill_alpha = float(state.hill_alpha)
    hill_gamma = state.hill_gamma
    logistic_lam = state.logistic_lam
    tanh_b = state.tanh_b
    tanh_c = state.tanh_c
    mm_alpha =  float(state.mm_alpha)
    mm_lam = state.mm_lam
    num_points = state.num_points
    display_saturate(state.display_saturation, media_spending, num_points, root_alpha, hill_alpha, hill_gamma, logistic_lam, tanh_b, tanh_c, mm_alpha, mm_lam)



# Define choices for distributions, adstock, and saturation
dis_choice_chart = ["Uniform", "Normal", "HalfNormal", "Beta", "Poisson", "Bernoulli", "Exponential", "Weibull", "TruncatedNormal", "StudentT", "LogNormal"]
dis_show_chart = dis_choice_chart[0]

Adstocks_types = ["Geometric", "Delayed Geometric", "Weibull CDF", "Weibull PDF"]
show_Adstock = Adstocks_types[0]

saturation_types = ["Root", "Hill", "Logistic", "Tanh", "Michaelis-Menten"]
display_saturation = saturation_types[0]

# Define menu
menu_lov = [("Distributions", Icon("Images/distributions.jpeg", "Distributions")),
            ("Adstock", Icon("Images/adstock.png", "Adstock")), 
            ("Saturation", Icon("Images/saturation.png", "Saturation"))]

ROOT = """
<|toggle|theme|>
<|menu|label=MMM Modeling|lov={menu_lov}|on_action=menu_fct|>
"""
pages = {
    "/": ROOT,
    'Distributions':distribution_md,
    "Adstock":adstock_md,
    "Saturation":saturation_md,
}

def menu_fct(state, var_name, var_value):
    """Function that is called when there is a change in the menu control."""
    page = var_value["args"][0]
    navigate(state, page)


if __name__=='__main__':
    # Create a Taipy GUI object with the pages dictionary
    tp_app = Gui(pages=pages)
    tp_app.run(title="Bayesian MMM functions", use_reloader=True, port=3636)
    