import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
from taipy.gui import Gui, Icon, State, navigate, notify, Markdown
import taipy.gui.builder as tgb
#import pymc as pm
from mmm_functions import *


# Starting value for adstock
initial_impact = 100
num_weeks = 20
beta = 0.30
max_peak = 10
shape_par = 0.10
scale_par  = 0.10


# Create df to store each adstock in
def adstock_geometric(initial_impact, num_weeks, beta):
    """
    beta : retention rate
    num_weeks: max duration of the carryover effect
    """
    # Get geometric adstock values, decayed over time
    adstock_df = pd.DataFrame({"Week": range(1, (num_weeks + 1)),
                            ## Calculate adstock values
                                "Adstock": geometric_adstock_decay(initial_impact, beta, num_weeks)})
    fig = px.line(adstock_df, x = 'Week',
                y = 'Adstock',
                markers=True)
    # Format plot
    fig.update_layout(title_text="Geometric Adstock Decayed Over Weeks", title_font = dict(size = 30))
    return fig

def adstock_delay_geometric(initial_impact, num_weeks, beta, max_peak):

    """
    beta : retention rate
    max_peak: delay before the peak effect occurs
    num_weeks: max duration of the carryover effect
    """
    # Get geometric adstock values, decayed over time
    adstock_df = pd.DataFrame({"Week": range(1, (num_weeks + 1)),
                            ## Calculate adstock values
                             "Adstock": delayed_geometric_decay(impact = initial_impact,
                                                                       decay_factor = beta,
                                                                        theta = max_peak,
                                                                        L = num_weeks)})
    
    fig = px.line(adstock_df, x = 'Week', y = 'Adstock', markers=True)
    # Format plot
    fig.update_layout(title_text="Delayed Geometric Adstock Decayed Over Weeks", title_font = dict(size = 30))

    return fig

def adstock_weibullCDF(initial_impact, num_weeks, shape_par, scale_par):
   
    # Create df of adstock values, to plot with
    adstock_df_A = pd.DataFrame({"Week": range(1, (num_weeks + 1)),
                                "Adstock": weibull_adstock_decay(initial_impact, shape_par,
                                                  scale_par, num_weeks,
                                                  adstock_type='cdf', normalised=True)})
    fig = px.line(adstock_df_A, x = 'Week',
                y = 'Adstock',
                markers=True)
    
    fig.update_layout(title_text="Weibull CDF Adstock Decayed Over Weeks", 
                    title_font = dict(size = 30))
    return fig 

def adstock_weibullPDF(initial_impact, num_weeks, shape_par, scale_par):
       
    # Create df of adstock values, to plot with
    adstock_df_A = pd.DataFrame({"Week": range(1, (num_weeks + 1)),
                                "Adstock": weibull_adstock_decay(initial_impact, shape_par,
                                                  scale_par, num_weeks,
                                                  adstock_type='pdf', normalised=True)})
    fig = px.line(adstock_df_A, x = 'Week',
                y = 'Adstock',
                markers=True)
    
    fig.update_layout(title_text="Weibull PDF Adstock Decayed Over Weeks", 
                    title_font = dict(size = 30))
    return fig

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
    

# Define the title dynamically
def update_adstock(state):
    num_weeks = state.num_weeks
    beta = float(state.beta)
    max_peak = state.max_peak
    shape_par = float( state.shape_par)
    scale_par = float(state.scale_par)
    get_selected_adstock(state.show_Adstock, initial_impact, num_weeks, beta, max_peak, shape_par, scale_par)



Adstocks_types = ["Geometric", "Delayed Geometric", "Weibull CDF", "Weibull PDF"]
show_Adstock = Adstocks_types[0]


layout = """
<|toggle|theme|>
## Adstock functions for MMM Bayesian Modelling
This webapp visualizes the impact of various adstock transformation on a variable. <br/>
For example, suppose we have a variable that represents the amount spent on a given advertising channel, For digital channels like Facebook, 
assume we acquired 100 impressions in the first week. 
When we see an ad on Facebook, we can either click it or not. 
Channels can be easily tracked, but some channels, such as TV/Radio, may not have an instant impact and may take some time to be seen <br/>
<br/>
<|{show_Adstock}|toggle|lov={Adstocks_types}|>
<|part|render={show_Adstock == 'Geometric'}|
### Geometric Adstock Transformation 
**Typical values for geometric adstock** from [Meta's Analyst's Guide to MMM](https://facebookexperimental.github.io/Robyn/docs/analysts-guide-to-MMM/#feature-engineering) <br/>
<br/>
- **TV:** 0.3 - 0.8 (decays slowly) <br/>
- **OOH/Print/Radio/Bilboards:** 0.1 - 0.4 (decays moderately) <br/>
- **Digital:** 0.0 - 0.3 (decays quickly) 
<br/>
<br/>
<|layout|columns=1 1|
<|first colum
<|container container-styling|
Number of weeks: <|{num_weeks}|> <br/>
<|{num_weeks}|slider|min=0|max=100|on_change=update_adstock|>
|>
|>
<|second column
<|container container-styling|
Beta: <|{beta}|> <br/>
<|{beta}|slider|min=0|max=1|step=0.01|on_change=update_adstock|>
|>
|>
|>
|>

<|part|render={show_Adstock == 'Delayed Geometric'}|
### Delayed Geometric Adstock Transformation 
**Typical values for geometric adstock** from [Meta's Analyst's Guide to MMM](https://facebookexperimental.github.io/Robyn/docs/analysts-guide-to-MMM/#feature-engineering) <br/>
<br/>
- **TV:** 0.3 - 0.8 (decays slowly) <br/>
- **OOH/Print/Radio/Bilboards:** 0.1 - 0.4 (decays moderately) <br/>
- **Digital:** 0.0 - 0.3 (decays quickly) 
<br/>
<br/>
<|layout|columns=1 1 1|
<|first colum
<|container container-styling|
Number of weeks: <|{num_weeks}|> <br/>
<|{num_weeks}|slider|min=0|max=100|on_change=update_adstock|>
|>
|>
<|second column
<|container container-styling|
No. of weeks for max impact: <|{max_peak}|> <br/>
<|{max_peak}|slider|min=0|max=100|on_change=update_adstock|>
|>
|>

<|third column
<|container container-styling|
Beta: <|{beta}|> <br/>
<|{beta}|slider|min=0|max=1|step=0.01|on_change=update_adstock|>
|>
|>
|>
|>

<|part|render={show_Adstock == 'Weibull CDF'}|
### Weibull CDF Adstock Transformation
__The Weibull CDF is a function depending on two variables, &kappa; (known as the **shape**) and &lambda; (known as the **scale**)__. <br/>
The idea is closely related to geometric adstock but with one important difference : the rate of decay (what we called &beta; in the geometric adstock equation) is no longer fixed. Instead **time-dependent**.
<br/>
<br/>
<|layout|columns=1 1 1|
<|first colum
<|container container-styling|
Number of weeks: <|{num_weeks}|> <br/>
<|{num_weeks}|slider|min=0|max=100|on_change=update_adstock|>
|>
|>
<|second column
<|container container-styling|
Shape K: <|{shape_par}|> <br/>
<|{shape_par}|slider|min=0|max=10|step = 0.1|on_change=update_adstock|>
|>
|>

<|third column
<|container container-styling|
Scale lam: <|{scale_par}|> <br/>
<|{scale_par}|slider|min=0|max=1|step=0.01|on_change=update_adstock|>
|>
|>
|>
|>

<|part|render={show_Adstock == 'Weibull PDF'}|
### Weibull PDF Adstock Transformation
__The Weibull PDF is a function depending on two variables, &kappa; (known as the **shape**) and &lambda; (known as the **scale**)__. <br/>
The key difference is that Weibull PDF allows for lagged effects to be taken into account - the **time delay effect**.
<br/>
<br/>
<|layout|columns=1 1 1|
<|first colum
<|container container-styling|
Number of weeks: <|{num_weeks}|> <br/>
<|{num_weeks}|slider|min=0|max=100|on_change=update_adstock|>
|>
|>
<|second column
<|container container-styling|
Shape K: <|{shape_par}|> <br/>
<|{shape_par}|slider|min=0|max=10|step = 0.1|on_change=update_adstock|>
|>
|>

<|third column
<|container container-styling|
Scale lam: <|{scale_par}|> <br/>
<|{scale_par}|slider|min=0|max=1|step=0.01|on_change=update_adstock|>
|>
|>
|>
|>

<|chart|figure={get_selected_adstock(show_Adstock, initial_impact, num_weeks, beta, max_peak, shape_par, scale_par)}|height=500px|on_change=update_adstock|>


"""

if __name__ == "__main__":
    
    Gui(layout).run(use_reloader=True, port=5003)
